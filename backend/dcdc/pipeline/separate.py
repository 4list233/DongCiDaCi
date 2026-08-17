"""Stage 1 -- isolate the drums.

Demucs v4 (htdemucs). This is the highest-leverage stage in the chain: every
downstream model gets better because the detector stops fighting the mix.

Runs Demucs as a subprocess rather than importing it. Demucs pins torch versions
fairly aggressively and we do not want that constraint leaking into the API
process -- this way the separator can live in its own environment if it has to.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL = "htdemucs"


@dataclass
class SeparationResult:
    drums: Path
    # Everything except the drums. Free from the same pass, and it is what the
    # play-along modes need: the song with a drum-shaped hole in it.
    no_drums: Path | None
    model: str
    device: str


def pick_device() -> str:
    """Prefer Apple Silicon's MPS, then CUDA, then CPU.

    On an M-series Mac Studio this is the difference between roughly real-time
    and several minutes per song.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate(
    audio: Path,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
) -> SeparationResult:
    """Split `audio` and return the path to the isolated drum stem.

    Uses --two-stems=drums, which is roughly twice as fast as a full four-stem
    split and produces the same drum track. We never look at the other stems.
    """
    audio = audio.resolve()
    out_dir = out_dir.resolve()
    if not audio.exists():
        raise FileNotFoundError(audio)

    device = device or pick_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "drums",
        "-n", model,
        "-d", device,
        "-o", str(out_dir),
        str(audio),
    ]
    log.info("separating on %s: %s", device, " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError(
            "demucs failed (exit %d):\n%s" % (proc.returncode, "\n".join(tail))
        )

    # Demucs writes to <out>/<model>/<track name>/{drums,no_drums}.wav
    produced = out_dir / model / audio.stem

    def take(name: str) -> Path | None:
        src = produced / f"{name}.wav"
        if not src.exists():
            candidates = list(out_dir.rglob(f"{name}.wav"))
            if not candidates:
                return None
            src = candidates[0]
        dest = out_dir / f"{name}.wav"
        if src != dest:
            shutil.move(str(src), str(dest))
        return dest

    drums = take("drums")
    no_drums = take("no_drums")
    shutil.rmtree(out_dir / model, ignore_errors=True)

    if drums is None:
        raise RuntimeError(f"demucs reported success but produced no drums.wav under {out_dir}")

    return SeparationResult(drums=drums, no_drums=no_drums, model=model, device=device)


def is_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False
