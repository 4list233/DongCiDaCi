"""Cached analysis: the expensive half of the pipeline, stored.

Separation and detection take a minute and never change for a given recording.
Quantization takes milliseconds and is full of judgement calls you will want to
revisit -- which beat is bar one, is this sixteenths or triplets, is the grid a
few milliseconds late.

Keeping them apart means fixing an alignment problem is instant instead of a
full re-run, which matters because alignment is the single most common thing
wrong with a first draft.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import Grid
from .transcribe import Onset, Transcription


@dataclass
class Analysis:
    onsets: list[Onset]
    grid: Grid
    adt_backend: str
    grid_backend: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "adt_backend": self.adt_backend,
            "grid_backend": self.grid_backend,
            "warnings": self.warnings,
            "grid": {
                "beats": [round(float(t), 5) for t in self.grid.beats],
                "downbeats": [round(float(t), 5) for t in self.grid.downbeats],
                "tempo": self.grid.tempo,
                "beats_per_bar": self.grid.beats_per_bar,
                "confidence": self.grid.confidence,
            },
            # level is stored alongside velocity because the two are not
            # interchangeable: velocity is relative to one instrument, level is
            # absolute and is the only thing comparable across stems. Dropping it
            # here would make a reread behave differently from the first run.
            "onsets": [
                [round(o.time, 5), o.lane, round(o.velocity, 4), round(o.level, 5)]
                for o in self.onsets
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Analysis":
        g = raw["grid"]
        grid = Grid(
            beats=np.asarray(g["beats"], dtype=float),
            downbeats=np.asarray(g["downbeats"], dtype=float),
            tempo=g["tempo"],
            beats_per_bar=g.get("beats_per_bar", 4),
            backend=raw.get("grid_backend", ""),
            confidence=g.get("confidence", 0.0),
        )
        return cls(
            # Three-element rows come from before level was recorded.
            onsets=[
                Onset(time=row[0], lane=row[1], velocity=row[2],
                      level=row[3] if len(row) > 3 else 0.0)
                for row in raw["onsets"]
            ],
            grid=grid,
            adt_backend=raw.get("adt_backend", ""),
            grid_backend=raw.get("grid_backend", ""),
            warnings=raw.get("warnings", []),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Analysis":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def transcription(self) -> Transcription:
        return Transcription(onsets=list(self.onsets), backend=self.adt_backend,
                             warnings=list(self.warnings))

    def shifted(self, offset_beats: int = 0, offset_ms: float = 0.0) -> "Analysis":
        """Re-align the grid without re-running any model.

        `offset_beats` rotates which beat counts as the downbeat -- the fix when
        the whole chart is displaced by a beat. `offset_ms` nudges every onset,
        for when detection sits consistently early or late against the grid.
        """
        grid = self.grid.shift_downbeats(offset_beats) if offset_beats else self.grid
        if offset_ms:
            shift = offset_ms / 1000.0
            onsets = [Onset(time=o.time + shift, lane=o.lane, velocity=o.velocity,
                            confidence=o.confidence) for o in self.onsets]
        else:
            onsets = list(self.onsets)

        return Analysis(
            onsets=onsets, grid=grid,
            adt_backend=self.adt_backend, grid_backend=self.grid_backend,
            warnings=list(self.warnings),
        )
