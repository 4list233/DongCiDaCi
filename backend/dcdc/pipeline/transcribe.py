"""Stage 3 -- detect onsets and classify which piece of the kit was hit.

Backends are pluggable because this is the stage most likely to change. The
research moves fast, and the good models have dependency trees that break on
Apple Silicon (madmom and TensorFlow are the usual culprits).

  spectral  -- always works. Onset detection plus band-energy classification.
               Pre-deep-learning technique. Gets a rock beat roughly right,
               misses ghost notes, cannot tell a ride from a crash.

  adtof     -- the real model. 5 classes, trained on 359 hours of real music.
               On a Mac use ADTOF-pytorch (github.com/xavriley/ADTOF-pytorch):
               same model to within 0.2% F-measure, and it needs only torch,
               librosa and pretty_midi -- no TensorFlow, no madmom, which are
               exactly what will not build on Apple Silicon.

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

# General MIDI channel-10 note numbers -> our lanes.
#
# The model resolves 5 classes, so most of this map is about reading its output
# sensibly rather than recovering detail it never had. Where a class could be
# several instruments we take the likelier one and let you correct it: a tom
# lands on the mid tom, an unspecified cymbal on the ride.
GM_TO_LANE: dict[int, str] = {
    35: "bd", 36: "bd",                          # acoustic / electric kick
    37: "sd", 38: "sd", 40: "sd",                # side stick, snare, e-snare
    39: "sd",                                    # hand clap reads as a backbeat
    42: "hh", 44: "hf", 46: "hh",                # closed, pedal, open hat
    41: "lt", 43: "lt",                          # floor toms
    45: "mt", 47: "mt",                          # low / low-mid toms
    48: "ht", 50: "ht",                          # hi-mid / high toms
    49: "cc", 52: "cc", 55: "cc", 57: "cc",      # crash, china, splash
    51: "rd", 53: "rd", 59: "rd",                # ride, bell, ride 2
}

# Hat notes that mean the hat was open. Carried through so the quantizer can
# mark them rather than flattening every hat to closed.
GM_OPEN_HAT = frozenset({46})


def _adtof(drums_path: Path) -> Transcription:
    """Bridge to ADTOF-pytorch.

    Writes MIDI to a temp file and reads it back rather than reaching into the
    model, because MIDI is the documented output and it carries velocity --
    which is what makes ghost-note detection downstream possible at all.
    """
    import tempfile

    from adtof_pytorch import transcribe_to_midi  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        midi_path = Path(tmp) / "drums.mid"
        try:
            transcribe_to_midi(str(drums_path), str(midi_path), device=_torch_device())
        except TypeError:
            # Older signatures take only the two paths.
            transcribe_to_midi(str(drums_path), str(midi_path))
        onsets, unmapped = _onsets_from_midi(midi_path)

    warnings = [
        "ADTOF resolves 5 classes: toms default to mid tom, cymbals to ride",
        "open vs closed hi-hat and ride vs crash still need your ear",
    ]
    if unmapped:
        warnings.append(
            f"ignored {len(unmapped)} hits on unmapped MIDI notes: {sorted(unmapped)}"
        )

    return Transcription(onsets=onsets, backend="adtof", warnings=warnings)


def _onsets_from_midi(midi_path: Path) -> tuple[list[Onset], set[int]]:
    """Read a General MIDI drum file into onsets. Returns (onsets, unmapped notes)."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    onsets: list[Onset] = []
    unmapped: set[int] = set()

    for instrument in midi.instruments:
        for note in instrument.notes:
            lane = GM_TO_LANE.get(note.pitch)
            if lane is None:
                unmapped.add(note.pitch)
                continue
            onsets.append(
                Onset(
                    time=float(note.start),
                    lane=lane,
                    velocity=float(note.velocity) / 127.0,
                    confidence=1.0,
                )
            )

    onsets.sort(key=lambda o: o.time)
    return onsets, unmapped


def _torch_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


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
        import adtof_pytorch  # noqa: F401
        return True
    except ImportError:
        return False
