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
    "cymbals": "rd",     # refined to rd/cc, see onsets.py
    "ride": "rd",
    "crash": "cc",
}

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


def is_available() -> bool:
    """True when both the inference code and a checkpoint are present."""
    return (MSST_DIR / "inference.py").exists() and bool(_find_checkpoint())


def _find_checkpoint() -> Path | None:
    if not MODEL_DIR.exists():
        return None
    for pattern in ("*drumsep*.ckpt", "*drumsep*.chpt", "*drumsep*.th", "*.ckpt"):
        found = sorted(MODEL_DIR.glob(pattern))
        if found:
            return found[0]
    return None


def _find_config() -> Path | None:
    if not MODEL_DIR.exists():
        return None
    found = sorted(MODEL_DIR.glob("*drumsep*.yaml")) or sorted(MODEL_DIR.glob("*.yaml"))
    return found[0] if found else None


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

    # MSST names outputs "<track>_<stem>.wav".
    stems: dict[str, Path] = {}
    for wav in sorted(out_dir.glob("*.wav")):
        for name in STEM_TO_LANE:
            if wav.stem.lower().endswith(f"_{name}"):
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
    return (
        "DrumSep is not installed. It splits the kit into per-instrument stems, "
        "which is what gives you toms, ride-vs-crash, and real velocities.\n"
        f"  git clone https://github.com/ZFTurbo/Music-Source-Separation-Training {MSST_DIR}\n"
        f"  mkdir -p {MODEL_DIR}\n"
        "  # then place the DrumSep MDX23C checkpoint (.ckpt) and its config (.yaml)\n"
        f"  # from github.com/jarredou/models releases into {MODEL_DIR}"
    )
