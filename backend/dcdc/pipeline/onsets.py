"""Per-stem onset detection and velocity.

When each stem contains one instrument, transcription stops being a
classification problem and becomes an onset-picking problem, which is far more
reliable. Two things fall out of it for free:

  velocity   the stem's own loudness at the hit, so a ghost note is genuinely
             quieter *than other snare hits* rather than quieter than a kick
  refinement toms split into floor/mid/high by pitch, and the cymbal stem splits
             into ride and crash by how the hits are spaced and how long they
             ring -- neither of which is recoverable from a mixed drum track

Velocity follows the method in arXiv 2509.24853: peak loudness in a short window
centred on the onset, normalised against that stem's own dynamic range.
"""

from __future__ import annotations

import logging

import numpy as np

from .transcribe import Onset

log = logging.getLogger(__name__)

# Window centred on the onset used to measure how hard the hit was.
VELOCITY_WINDOW_S = 0.05

# A stem is "silent" if its peak never gets near full scale; separating a kit
# that has no toms still produces a toms stem, full of bleed.
SILENT_STEM_PEAK = 0.02

# Bleed from neighbouring instruments shows up as low-level onsets. Anything
# this far below the stem's own loud hits is discarded rather than notated.
BLEED_FLOOR = 0.10


def detect_stem(
    y: np.ndarray,
    sr: int,
    lane: str,
    delta: float = 0.07,
    min_gap_s: float = 0.025,
) -> list[Onset]:
    """Find hits in a single-instrument stem, with velocities.

    `delta` is deliberately lower than a whole-kit detector would use: the stem
    only contains one instrument, so a quiet peak is a quiet hit rather than a
    different drum, and that is precisely the ghost note we do not want to miss.
    """
    import librosa

    if y.size == 0 or float(np.max(np.abs(y))) < SILENT_STEM_PEAK:
        return []

    hop = 256                      # ~5.8ms at 44.1k: fine enough for flams
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        backtrack=True, delta=delta,
        wait=max(1, int(min_gap_s * sr / hop)),
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(times) == 0:
        return []

    loudness = _peak_loudness(y, sr, times)
    if loudness.size == 0:
        return []

    # Normalise against this stem's own loud hits, not against absolute scale.
    # The 90th percentile rather than the max so a single crash does not make
    # every other hit look like a ghost note.
    reference = float(np.percentile(loudness, 90)) or 1.0
    normalised = np.clip(loudness / reference, 0.0, 1.0)

    onsets = []
    for time, velocity in zip(times, normalised):
        if velocity < BLEED_FLOOR:
            continue
        onsets.append(Onset(time=float(time), lane=lane, velocity=float(velocity),
                            confidence=0.9))
    return onsets


def _peak_loudness(y: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """Peak absolute amplitude in a window centred on each onset."""
    half = int(VELOCITY_WINDOW_S * sr / 2)
    out = []
    for t in times:
        centre = int(t * sr)
        lo = max(0, centre - half)
        hi = min(len(y), centre + half)
        segment = y[lo:hi]
        out.append(float(np.max(np.abs(segment))) if segment.size else 0.0)
    return np.asarray(out, dtype=float)


def split_toms(y: np.ndarray, sr: int, onsets: list[Onset]) -> list[Onset]:
    """Assign tom hits to floor / mid / high by fundamental pitch.

    The separator gives one toms stem. Which tom is which is recoverable because
    they are pitched: a floor tom sits around 80-110Hz and a rack tom well above
    that. Fills descend, so getting this wrong puts every fill upside down.
    """
    import librosa

    if not onsets:
        return []

    pitches = []
    for onset in onsets:
        centre = int(onset.time * sr)
        segment = y[centre : centre + int(0.12 * sr)]
        if segment.size < 512:
            pitches.append(np.nan)
            continue
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        freqs = np.fft.rfftfreq(len(segment), 1 / sr)
        band = (freqs > 60) & (freqs < 400)
        pitches.append(float(freqs[band][np.argmax(spectrum[band])]) if band.any() else np.nan)

    valid = np.asarray([p for p in pitches if not np.isnan(p)])
    if valid.size == 0:
        return onsets

    # Split on the observed range rather than fixed frequencies -- every kit is
    # tuned differently, and what matters is which tom is lower than which.
    low, high = float(np.percentile(valid, 15)), float(np.percentile(valid, 85))
    span = max(high - low, 1.0)

    for onset, pitch in zip(onsets, pitches):
        if np.isnan(pitch):
            continue
        position = (pitch - low) / span
        onset.lane = "lt" if position < 0.33 else "ht" if position > 0.66 else "mt"
    return onsets


def split_cymbals(y: np.ndarray, sr: int, onsets: list[Onset]) -> list[Onset]:
    """Separate ride pattern from crash accents within one cymbal stem.

    A ride is played repeatedly and evenly; a crash is loud, isolated, and rings
    much longer. Neither is recoverable from a mixed drum track, which is why
    5-class models collapse them into one voice.
    """
    if len(onsets) < 3:
        for onset in onsets:
            onset.lane = "cc"          # isolated cymbal hits read as crashes
        return onsets

    times = np.asarray([o.time for o in onsets])
    velocities = np.asarray([o.velocity for o in onsets])
    gaps = np.diff(times, prepend=times[0] - 1.0)
    typical_gap = float(np.median(gaps[gaps > 0])) or 1.0

    for onset, gap, velocity in zip(onsets, gaps, velocities):
        # Loud, and preceded by a longer-than-usual gap: a crash.
        isolated = gap > typical_gap * 1.8
        loud = velocity > float(np.percentile(velocities, 80))
        onset.lane = "cc" if (isolated and loud) else "rd"
    return onsets


def transcribe_stems(lane_stems: dict, sr_target: int = 44100) -> tuple[list[Onset], list[str]]:
    """Run detection across every stem and return one merged, sorted list."""
    import librosa

    onsets: list[Onset] = []
    warnings: list[str] = []

    for lane, path in sorted(lane_stems.items()):
        try:
            y, sr = librosa.load(str(path), mono=True, sr=sr_target)
        except Exception as exc:
            warnings.append(f"could not read the {lane} stem: {exc}")
            continue

        # Cymbals and hats need a higher threshold: they are noisy and a low
        # delta turns a single crash's decay into a stream of phantom hits.
        delta = 0.10 if lane in ("rd", "cc", "hh") else 0.06
        found = detect_stem(y, sr, lane, delta=delta)

        if lane == "mt":
            found = split_toms(y, sr, found)
        elif lane == "rd":
            found = split_cymbals(y, sr, found)

        if not found:
            warnings.append(f"no hits found in the {lane} stem")
        onsets.extend(found)

    onsets.sort(key=lambda o: o.time)
    return onsets, warnings
