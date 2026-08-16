"""Stage 2 -- find the grid.

Beats, downbeats, tempo. Without this you have timestamps, not bars, and
timestamps do not engrave. A downbeat off by one puts the entire chart on the
"and", which is the single most destructive failure in the pipeline.

Two backends:

  beat_this  -- ISMIR 2024 transformer, accurate, gives real downbeats
  librosa    -- always installed, gives beats but *infers* downbeats naively

The librosa path exists so the pipeline runs on day one. Its downbeat inference
assumes the first strong beat is bar one, which is wrong often enough that the
editor lets you nudge the whole grid by a beat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Grid:
    beats: np.ndarray          # seconds, every beat
    downbeats: np.ndarray      # seconds, subset of beats that start a bar
    tempo: float               # median bpm
    beats_per_bar: int = 4
    backend: str = "librosa"
    confidence: float = 0.0    # 0..1, how regular the beat spacing is
    warnings: list[str] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return max(len(self.downbeats) - 1, 0)

    def bar_bounds(self, i: int) -> tuple[float, float]:
        """Start and end time of bar `i` (0-indexed)."""
        return float(self.downbeats[i]), float(self.downbeats[i + 1])

    def local_tempo(self, i: int) -> float:
        start, end = self.bar_bounds(i)
        span = end - start
        return self.beats_per_bar * 60.0 / span if span > 0 else self.tempo

    def shift_downbeats(self, by: int) -> "Grid":
        """Rotate which beat counts as the downbeat.

        The most common correction a human makes to a machine grid, so it gets
        to be a first-class operation rather than a manual edit.
        """
        if len(self.beats) == 0:
            return self
        idx = int(np.searchsorted(self.beats, self.downbeats[0])) if len(self.downbeats) else 0
        start = (idx + by) % self.beats_per_bar
        return Grid(
            beats=self.beats,
            downbeats=self.beats[start::self.beats_per_bar],
            tempo=self.tempo,
            beats_per_bar=self.beats_per_bar,
            backend=self.backend,
            confidence=self.confidence,
            warnings=self.warnings + [f"downbeats shifted by {by}"],
        )


def detect(audio_path, beats_per_bar: int = 4, backend: str = "auto") -> Grid:
    """Detect the beat grid. Falls back to librosa if beat_this is unavailable."""
    if backend in ("auto", "beat_this"):
        try:
            return _beat_this(audio_path, beats_per_bar)
        except ImportError:
            if backend == "beat_this":
                raise
            log.info("beat_this not installed, falling back to librosa")
        except Exception as exc:
            if backend == "beat_this":
                raise
            log.warning("beat_this failed (%s), falling back to librosa", exc)
    return _librosa(audio_path, beats_per_bar)


def _beat_this(audio_path, beats_per_bar: int) -> Grid:
    from beat_this.inference import File2Beats

    f2b = File2Beats(device=_torch_device())
    beats, downbeats = f2b(str(audio_path))
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)

    tempo, conf = _tempo_and_confidence(beats)
    warnings = []
    if len(downbeats) < 2:
        warnings.append("beat_this found fewer than two downbeats; bar detection unreliable")

    return Grid(
        beats=beats,
        downbeats=downbeats,
        tempo=tempo,
        beats_per_bar=beats_per_bar,
        backend="beat_this",
        confidence=conf,
        warnings=warnings,
    )


def _librosa(audio_path, beats_per_bar: int) -> Grid:
    import librosa

    y, sr = librosa.load(str(audio_path), mono=True)
    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr)

    # Naive downbeat inference: score each of the `beats_per_bar` possible
    # phases by the onset strength landing on it, and take the strongest. This
    # is right more often than chance and wrong often enough to need the editor.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    phase_scores = [
        float(strengths[p::beats_per_bar].mean()) if len(strengths[p::beats_per_bar]) else 0.0
        for p in range(beats_per_bar)
    ]
    phase = int(np.argmax(phase_scores))

    tempo, conf = _tempo_and_confidence(beats)
    best, second = sorted(phase_scores, reverse=True)[:2] if len(phase_scores) > 1 else (1.0, 0.0)
    warnings = ["downbeats inferred by librosa; verify bar 1 before trusting the chart"]
    if second > 0 and best / second < 1.15:
        warnings.append(
            "downbeat phase was ambiguous -- if the chart feels displaced, shift the grid"
        )

    return Grid(
        beats=beats,
        downbeats=beats[phase::beats_per_bar],
        tempo=float(tempo_est if np.isscalar(tempo_est) else np.atleast_1d(tempo_est)[0]),
        beats_per_bar=beats_per_bar,
        backend="librosa",
        confidence=conf,
        warnings=warnings,
    )


def _tempo_and_confidence(beats: np.ndarray) -> tuple[float, float]:
    """Median tempo, plus how regular the spacing is (1.0 = metronomic)."""
    if len(beats) < 2:
        return 120.0, 0.0
    intervals = np.diff(beats)
    median = float(np.median(intervals))
    if median <= 0:
        return 120.0, 0.0
    spread = float(np.std(intervals) / median)
    return 60.0 / median, float(max(0.0, 1.0 - spread * 4))


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
