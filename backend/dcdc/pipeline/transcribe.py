"""Stage 3 -- detect onsets and classify which piece of the kit was hit.

Backends are pluggable because this is the stage most likely to change. The
research moves fast, and the good models have dependency trees that break on
Apple Silicon (madmom and TensorFlow are the usual culprits).

  spectral  -- always works. Onset detection plus band-energy classification.
               Pre-deep-learning technique. Gets a rock beat roughly right,
               misses ghost notes, cannot tell a ride from a crash.

  adtof     -- the real model. 5 classes, trained on 359 hours of real music.
               Prefer the ADTOF-pytorch variant on a Mac: same accuracy to
               within 0.2% F-measure, without TensorFlow or madmom.

Both return the same thing, so the rest of the pipeline does not care which ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# What a 5-class model can actually distinguish. Anything finer -- ride vs crash,
# open vs closed hat, which tom -- is yours to add by ear in the editor.
ADT_CLASSES = ("bd", "sd", "hh", "tt", "cy")


@dataclass
class Onset:
    time: float          # seconds
    lane: str            # a key from chart.LANES
    velocity: float      # 0..1
    confidence: float = 1.0
    # Set during quantization when neighbouring onsets collapse into one gesture.
    rudiment: str | None = None


@dataclass
class Transcription:
    onsets: list[Onset]
    backend: str
    warnings: list[str]


def transcribe(drums_path: Path, backend: str = "auto") -> Transcription:
    if backend in ("auto", "adtof"):
        try:
            return _adtof(drums_path)
        except ImportError:
            if backend == "adtof":
                raise
            log.info("adtof not installed, falling back to spectral")
        except Exception as exc:
            if backend == "adtof":
                raise
            log.warning("adtof failed (%s), falling back to spectral", exc)
    return _spectral(drums_path)


# --- the real model ---------------------------------------------------------

def _adtof(drums_path: Path) -> Transcription:
    """Bridge to ADTOF.

    ADTOF ships as a notebook rather than a library API, so this adapts whichever
    entry point is present. Its 5 classes map onto our lanes with a deliberate
    loss of detail: `tt` lands on the mid tom and `cy` on the ride, because those
    are the likelier reading, and you correct them in the editor.
    """
    from adtof.model.model import Model  # type: ignore

    model = Model.modelFactory(modelName="crnn-ADTOF")[0]
    model.load()
    prediction = model.predictOneTrack(str(drums_path))

    lane_of = {"bd": "bd", "sd": "sd", "hh": "hh", "tt": "mt", "cy": "rd"}
    onsets: list[Onset] = []
    for cls, times in zip(ADT_CLASSES, prediction):
        for t in np.atleast_1d(times):
            onsets.append(Onset(time=float(t), lane=lane_of[cls], velocity=0.72))

    onsets.sort(key=lambda o: o.time)
    return Transcription(
        onsets=onsets,
        backend="adtof",
        warnings=[
            "ADTOF resolves 5 classes: toms default to mid tom and cymbals to ride",
            "velocities are not predicted, so ghost notes need marking by ear",
        ],
    )


# --- the fallback -----------------------------------------------------------

# Frequency bands that separate the three things a spectral method can actually
# separate. Deliberately narrow: overlapping bands produce phantom simultaneous
# hits on every onset.
BANDS = {
    "bd": (30, 130),
    "sd": (180, 900),
    "hh": (6000, 14000),
}


def _spectral(drums_path: Path) -> Transcription:
    """Onset detection plus band-energy classification.

    Honest about what it is: this decides kick/snare/hat by asking which
    frequency band dominates at each onset. It will produce a usable skeleton
    for a straight rock beat and will embarrass itself on anything busier.
    """
    import librosa

    y, sr = librosa.load(str(drums_path), mono=True)
    if len(y) == 0:
        return Transcription([], "spectral", ["audio was empty"])

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=True, units="frames",
        pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=0.12, wait=2,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_idx = {
        lane: np.where((freqs >= lo) & (freqs < hi))[0] for lane, (lo, hi) in BANDS.items()
    }

    # A hit smears forward, so measure a short window after the onset rather
    # than a single frame.
    window = max(1, int(0.03 * sr / hop))

    raw: list[tuple[float, str, float]] = []
    for t in onset_times:
        f = int(librosa.time_to_frames(t, sr=sr, hop_length=hop))
        seg = S[:, f : f + window]
        if seg.size == 0:
            continue
        energy = {lane: float(seg[idx].sum()) for lane, idx in band_idx.items()}
        total = sum(energy.values()) or 1.0

        # Multiple lanes can fire on one onset -- kick and hat together is the
        # single most common event in popular music.
        for lane, e in energy.items():
            share = e / total
            if share > 0.22:
                raw.append((float(t), lane, e))

    if not raw:
        return Transcription([], "spectral", ["no onsets detected"])

    # Normalize loudness per lane. Absolute energy is meaningless across lanes
    # (a kick dwarfs a hat), but relative dynamics within a lane are what tell
    # an accent from a ghost note.
    onsets: list[Onset] = []
    for lane in BANDS:
        vals = [e for _, l, e in raw if l == lane]
        if not vals:
            continue
        ref = float(np.percentile(vals, 90)) or 1.0
        for t, l, e in raw:
            if l == lane:
                onsets.append(
                    Onset(time=t, lane=lane, velocity=float(np.clip(e / ref, 0.05, 1.0)),
                          confidence=0.5)
                )

    onsets.sort(key=lambda o: o.time)
    return Transcription(
        onsets=onsets,
        backend="spectral",
        warnings=[
            "spectral fallback: kick/snare/hi-hat only",
            "no toms, no cymbal distinction, ghost notes are unreliable",
            "install ADTOF for a real transcription",
        ],
    )


def is_adtof_available() -> bool:
    try:
        import adtof  # noqa: F401
        return True
    except ImportError:
        return False
