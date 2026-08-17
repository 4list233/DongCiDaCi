"""Stage 1b -- split the drum kit into per-instrument stems.

This is the architectural change. Classifying a mixed drum track into N classes
asks one model to answer "which of these overlapping things did I just hear",
which is exactly where 5-class models lose toms, collapse ride into crash, and
flatten ghost notes. Separating first turns that into N easy questions: each
stem contains one instrument, so onset detection is nearly trivial and the
stem's own RMS envelope gives a real velocity.

Two 2025-26 papers converge on this. `Enhanced Automatic Drum Transcription via
Drum Stem Source Separation` (arXiv 2509.24853) uses exactly this to expand
ADTOF from five classes to seven *and* recover MIDI velocities. `Separate-and-
Detect` (arXiv 2608.01093) builds a five-stem latent-diffusion separator and
then runs a fixed onset detector per stem.

The separator here is DrumSep (MDX23C / TFC-TDF-Net-v3, by jarredou and aufr33),
run through ZFTurbo's Music-Source-Separation-Training inference script. It is
optional: without it the pipeline falls back to whole-kit classification.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# DrumSep's stem names -> our lane keys.
#
# `cymbals` covers both ride and crash; splitting them is done afterwards from
# the stem itself (see onsets.split_cymbals), because a ride pattern and a crash
# accent look very different in time even when they share a stem.
STEM_TO_LANE: dict[str, str] = {
    "kick": "bd",
    "snare": "sd",
    "toms": "mt",        # refined to lt/mt/ht by pitch, see onsets.py
    "hh": "hh",
    "hihat": "hh",
    "hi-hat": "hh",      # the v0.1 model spells it this way
    "cymbals": "rd",     # undifferentiated: refined to rd/cc, see onsets.py
    "ride": "rd",        # already separated by the 6-stem model
    "crash": "cc",
}

# The 6-stem model separates ride from crash itself, at SDR 10.8. The 5-stem one
# emits a single "cymbals" stem. Guessing ride-vs-crash from timing is a decent
# fallback but it is strictly worse than a model that was trained to tell them
# apart, so it must not run on top of a real ride stem.
UNDIFFERENTIATED_CYMBALS = "cymbals"

# Where the checkpoint and config live once installed.
MODEL_DIR = Path(os.environ.get("DCDC_MODEL_DIR", Path.home() / ".cache" / "dcdc" / "models"))
MSST_DIR = Path(os.environ.get("DCDC_MSST_DIR", Path.home() / ".cache" / "dcdc" / "msst"))


@dataclass
class DrumStems:
    """Paths to whichever per-instrument stems the separator produced."""
    stems: dict[str, Path] = field(default_factory=dict)
    model: str = ""

    def __bool__(self) -> bool:
        return bool(self.stems)

    def lane_stems(self) -> dict[str, Path]:
        """Stems keyed by our lane names rather than the model's stem names."""
        out: dict[str, Path] = {}
        for name, path in self.stems.items():
            lane = STEM_TO_LANE.get(name.lower())
            if lane:
                out[lane] = path
        return out

    @property
    def cymbals_need_splitting(self) -> bool:
        """True when the model gave one cymbal stem rather than ride and crash.

        Decided from what the separator actually produced, so installing a
        different checkpoint changes this without any code change.
        """
        names = {n.lower() for n in self.stems}
        if "ride" in names or "crash" in names:
            return False
        return UNDIFFERENTIATED_CYMBALS in names


# Published checkpoints, best first.
#
# Weights for these models get rehosted and the original GitHub release for the
# v0.1 model is already gone, so each file lists mirrors rather than one URL.
# Filenames are consistent across the registries that index them, which is what
# makes mirroring viable at all.
_HF = "https://huggingface.co"
_POLITREES = f"{_HF}/Politrees/UVR_resources/resolve/main/models/MDX23C"
_POLITREES_OLD = f"{_HF}/Politrees/UVR_resources/resolve/main/MDX23C_models"
_LAINLIVES = f"{_HF}/lainlives/audio-separator-models/resolve/main"
_JARREDOU = "https://github.com/jarredou/models/releases/download"

CHECKPOINTS: list[dict] = [
    {
        "name": "MDX23C DrumSep (aufr33 & jarredou)",
        # What it actually produces is read from the config after download
        # rather than trusted from here.
        "stems": "kick, snare, toms, hi-hat, and cymbals (ride/crash per version)",
        "ckpt": [
            f"{_POLITREES}/MDX23C-DrumSep-aufr33-jarredou.ckpt",
            f"{_POLITREES_OLD}/MDX23C-DrumSep-aufr33-jarredou.ckpt",
            f"{_LAINLIVES}/MDX23C-DrumSep-aufr33-jarredou.ckpt",
        ],
        "config": [
            f"{_POLITREES}/config_drumsep_mdx23c.yaml",
            f"{_POLITREES_OLD}/config_drumsep_mdx23c.yaml",
            f"{_JARREDOU}/DrumSep/config_mdx23c.yaml",
        ],
    },
]


def config_instruments(config_path: Path) -> list[str]:
    """The stem names a config declares, in order. Empty if unreadable.

    Authoritative: it is what the model was trained to output, so it beats
    inferring stems from whatever filenames the inference script happens to
    write.
    """
    try:
        import yaml
    except ImportError:
        log.warning("pyyaml is not installed, so stem names fall back to guesswork")
        return []
    try:
        with Path(config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception as exc:
        # A config we cannot parse is not fatal; stem matching falls back to
        # every name we know. Say so rather than degrading in silence.
        log.warning("could not parse %s: %s", config_path, exc)
        return []

    training = config.get("training") or {}
    instruments = training.get("instruments") or config.get("instruments") or []
    return [str(name) for name in instruments]


def _fetch(urls: list[str], dest: Path) -> bool:
    """Download the first URL that works. True if `dest` ends up present."""
    import urllib.error
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return True

    # Download beside the target and rename, so an interrupted download cannot
    # leave a truncated file that later looks installed.
    partial = dest.with_suffix(dest.suffix + ".part")
    for url in urls:
        host = url.split("/")[2]
        try:
            print(f"  fetching {dest.name} from {host} ...", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "dcdc"})
            with urllib.request.urlopen(request) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            print(f"    HTTP {exc.code}, trying the next mirror")
            continue
        except OSError as exc:
            partial.unlink(missing_ok=True)
            print(f"    {exc}, trying the next mirror")
            continue
        partial.rename(dest)
        print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return True

    print(f"  every mirror failed for {dest.name}")
    return False


def download(index: int = 0) -> bool:
    """Fetch a checkpoint and its config into MODEL_DIR. True on success."""
    choice = CHECKPOINTS[index]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {choice['name']}")

    config_urls, ckpt_urls = choice["config"], choice["ckpt"]
    config_dest = MODEL_DIR / config_urls[0].rsplit("/", 1)[-1]
    ckpt_dest = MODEL_DIR / ckpt_urls[0].rsplit("/", 1)[-1]

    # Config first: it is small, so a dead mirror set fails in a second rather
    # than after pulling 438MB.
    if not _fetch(config_urls, config_dest):
        return False
    if not _fetch(ckpt_urls, ckpt_dest):
        return False

    instruments = config_instruments(config_dest)
    if instruments:
        print(f"  stems: {', '.join(instruments)}")
        unmapped = [i for i in instruments if i.lower() not in STEM_TO_LANE]
        if unmapped:
            print(f"  warning: no lane mapped for {unmapped}; those stems are ignored")
    return is_available()


def is_available() -> bool:
    """True when both the inference code and a checkpoint are present."""
    return (MSST_DIR / "inference.py").exists() and bool(_find_checkpoint())


def check() -> tuple[bool, list[str]]:
    """Report whether separation can actually run, and what is missing.

    Having the files is not the same as being able to run them: the inference
    script has its own dependency list, and if it is unmet the run fails at
    import time. That failure is caught and the pipeline degrades to whole-kit
    transcription, which produces a chart with no toms -- indistinguishable from
    a bad model unless the real reason is surfaced here.
    """
    problems: list[str] = []

    if not (MSST_DIR / "inference.py").exists():
        problems.append(
            "inference code missing -- git clone "
            f"https://github.com/ZFTurbo/Music-Source-Separation-Training {MSST_DIR}"
        )
    if not _find_checkpoint():
        problems.append("no checkpoint -- run: dcdc install-drumsep")
    if not _find_config():
        problems.append("no model config -- run: dcdc install-drumsep")

    missing = _missing_imports()
    if missing:
        problems.append(
            f"python packages missing: {', '.join(missing)} -- run: "
            f"pip install -r {MSST_DIR / 'requirements.txt'}"
        )

    return not problems, problems


# What MSST's mdx23c inference path imports beyond what we already depend on.
_MSST_IMPORTS = ("torch", "yaml", "omegaconf", "ml_collections", "einops", "tqdm")


def _missing_imports() -> list[str]:
    import importlib.util

    return [name for name in _MSST_IMPORTS if importlib.util.find_spec(name) is None]


def describe_stems() -> str:
    """The stems the installed checkpoint declares, for `dcdc doctor`."""
    config = _find_config()
    if not config:
        return ""
    instruments = config_instruments(config)
    if not instruments:
        return "unknown (could not read the config)"
    lanes = [i for i in instruments if i.lower() in STEM_TO_LANE]
    text = ", ".join(instruments)
    if len(lanes) < len(instruments):
        unmapped = [i for i in instruments if i.lower() not in STEM_TO_LANE]
        text += f"  (unmapped, will be ignored: {', '.join(unmapped)})"
    return text


def _find(suffixes: tuple[str, ...]) -> Path | None:
    """Newest matching file in MODEL_DIR, preferring names mentioning drumsep.

    Matching is case-insensitive because the published checkpoint is spelled
    `...DrumSep_model...`, and a case-sensitive glob silently finds nothing --
    which looks exactly like the model not being installed.
    """
    if not MODEL_DIR.exists():
        return None
    candidates = [
        p for p in MODEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    # Prefer an explicitly-named drumsep file over any other model dropped here.
    named = [p for p in candidates if "drumsep" in p.name.lower()]
    return sorted(named or candidates)[-1]


def _find_checkpoint() -> Path | None:
    return _find((".ckpt", ".chpt", ".th"))


def _find_config() -> Path | None:
    return _find((".yaml", ".yml"))


def separate(drums_path: Path, out_dir: Path, device: str | None = None) -> DrumStems:
    """Split an isolated drum track into per-instrument stems.

    Returns an empty DrumStems if the separator is not installed, so callers can
    degrade rather than fail.
    """
    if not is_available():
        log.info("drumsep not installed; skipping per-instrument separation")
        return DrumStems()

    checkpoint = _find_checkpoint()
    config = _find_config()
    if checkpoint is None or config is None:
        log.warning("drumsep checkpoint or config missing under %s", MODEL_DIR)
        return DrumStems()

    out_dir = Path(out_dir) / "kit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # MSST's inference script works on a folder, so give it one containing only
    # this track.
    staging = out_dir / "_in"
    staging.mkdir(exist_ok=True)
    staged = staging / drums_path.name
    if not staged.exists():
        shutil.copy2(drums_path, staged)

    device = device or _pick_device()
    cmd = [
        sys.executable, str(MSST_DIR / "inference.py"),
        "--model_type", "mdx23c",
        "--config_path", str(config),
        "--start_check_point", str(checkpoint),
        "--input_folder", str(staging),
        "--store_dir", str(out_dir),
        "--device_ids", "0" if device == "cuda" else "0",
    ]
    if device == "cpu":
        cmd.append("--force_cpu")

    log.info("drumsep: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        log.warning("drumsep failed (exit %d):\n%s", proc.returncode, "\n".join(tail))
        return DrumStems()

    shutil.rmtree(staging, ignore_errors=True)

    # Match against the stem names the config declares, falling back to every
    # name we know how to map. MSST names outputs "<track>_<stem>.wav".
    expected = config_instruments(config) or list(STEM_TO_LANE)
    stems: dict[str, Path] = {}
    for wav in sorted(out_dir.glob("*.wav")):
        for name in expected:
            if wav.stem.lower().endswith(f"_{name.lower()}"):
                stems[name] = wav
                break

    if not stems:
        log.warning("drumsep produced no recognisable stems in %s", out_dir)

    return DrumStems(stems=stems, model=checkpoint.name)


def _pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def install_hint() -> str:
    lines = [
        "DrumSep is not installed. It splits the kit into per-instrument stems, "
        "which is what gives you toms, ride vs crash, and real velocities.",
    ]
    if not (MSST_DIR / "inference.py").exists():
        lines.append(
            "  git clone https://github.com/ZFTurbo/Music-Source-Separation-Training "
            f"{MSST_DIR}"
        )
    if not _find_checkpoint() or not _find_config():
        lines.append("  dcdc install-drumsep")
    return "\n".join(lines)
