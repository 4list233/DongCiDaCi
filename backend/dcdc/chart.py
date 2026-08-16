"""The chart format.

A chart is stored as ``chart.json`` and is the canonical, hand-editable source of
truth for a song. Everything else -- AlphaTex for the browser, LilyPond for print,
MIDI for a DAW -- is generated from it and can be thrown away.

The format borrows its vocabulary from drum tab, because drummers already read it
and because it makes ``git diff`` legible::

    {"n": 5, "section": "verse", "lanes": {
        "hh": "x-x-x-x-x-x-x-x-",
        "sd": "----o---g---o---",
        "bd": "o-------o--o----"
    }}

One character per subdivision, ``res`` characters per bar. You can see the groove
in the diff, and ``grep`` finds patterns across the whole fill library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

# --- lanes ------------------------------------------------------------------
#
# Ordered top-to-bottom as they sit on the staff. `midi` is the General MIDI
# channel-10 note; `alphatex` is alphaTab's percussion articulation name.

@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    midi: int
    alphatex: str


# The `alphatex` names are not free-form -- alphaTab validates them against a
# fixed vocabulary and rejects the whole score if one is wrong. These are taken
# from that vocabulary, not invented; see frontend/test/alphatex.test.mjs, which
# parses generated output through alphaTab itself to keep them honest.
LANES: tuple[Lane, ...] = (
    Lane("cc", "Crash",      49, "CrashHighHit"),
    Lane("rd", "Ride",       51, "RideMiddle"),
    Lane("hh", "Hi-hat",     42, "HiHatClosed"),
    Lane("ht", "High tom",   48, "HighTomHit"),
    Lane("mt", "Mid tom",    45, "MidTomHit"),
    Lane("lt", "Floor tom",  41, "LowFloorTomHit"),
    Lane("sd", "Snare",      38, "SnareHit"),
    Lane("bd", "Kick",       36, "KickHit"),
    Lane("hf", "Hat foot",   44, "PedalHiHatHit"),
)

LANE_BY_KEY: dict[str, Lane] = {lane.key: lane for lane in LANES}

# --- articulations ----------------------------------------------------------
#
# The character vocabulary. `velocity` is 0..1 and survives into MIDI export;
# `open` and `rudiment` are the annotations a transcription model cannot recover
# and you have to supply by ear.

@dataclass(frozen=True)
class Articulation:
    char: str
    name: str
    velocity: float
    open: bool = False
    rudiment: str | None = None


ARTICULATIONS: tuple[Articulation, ...] = (
    Articulation("-", "rest",   0.0),
    Articulation("x", "hit",    0.72),
    Articulation("X", "accent", 1.0),
    Articulation("o", "hit",    0.72),
    Articulation("O", "accent", 1.0),
    Articulation("g", "ghost",  0.28),
    Articulation("+", "open",   0.80, open=True),
    Articulation("f", "flam",   0.80, rudiment="flam"),
    Articulation("d", "drag",   0.70, rudiment="drag"),
    Articulation("b", "buzz",   0.60, rudiment="buzz"),
)

ART_BY_CHAR: dict[str, Articulation] = {a.char: a for a in ARTICULATIONS}
REST = "-"

# `x` and `o` mean the same thing mechanically; the split is conventional, so
# cymbals read as x-noteheads and drums as ordinary noteheads.
CYMBAL_LANES = frozenset({"cc", "rd", "hh", "hf"})


class ChartError(ValueError):
    """Raised when a chart file is structurally invalid."""


@dataclass
class Bar:
    n: int
    lanes: dict[str, str] = field(default_factory=dict)
    section: str | None = None
    # Free-text, per bar. This is the annotation layer -- "ghosts are the point
    # here", "he's playing this on the ride bell", "I play it as a single".
    note: str | None = None

    def validate(self, res: int) -> None:
        for key, pattern in self.lanes.items():
            if key not in LANE_BY_KEY:
                raise ChartError(f"bar {self.n}: unknown lane {key!r}")
            if len(pattern) != res:
                raise ChartError(
                    f"bar {self.n}: lane {key!r} has {len(pattern)} slots, expected {res}"
                )
            for i, ch in enumerate(pattern):
                if ch not in ART_BY_CHAR:
                    raise ChartError(
                        f"bar {self.n}: lane {key!r} slot {i}: unknown articulation {ch!r}"
                    )

    def hits(self) -> Iterator[tuple[int, str, Articulation]]:
        """Yield (slot, lane_key, articulation) for every sounding hit, in time order."""
        for slot in range(self._res()):
            for lane in LANES:
                pattern = self.lanes.get(lane.key)
                if not pattern or pattern[slot] == REST:
                    continue
                yield slot, lane.key, ART_BY_CHAR[pattern[slot]]

    def is_empty(self) -> bool:
        return all(set(p) <= {REST} for p in self.lanes.values())

    def _res(self) -> int:
        return len(next(iter(self.lanes.values()))) if self.lanes else 0


@dataclass
class SyncPoint:
    """Anchors a bar to a wall-clock position in the original recording.

    These come out of the beat tracker for free and are what let alphaTab move
    its cursor along a real performance instead of a synthesized one.
    """
    bar: int
    time: float       # seconds into the source audio
    tempo: float      # bpm at this point


@dataclass
class Chart:
    title: str
    artist: str = ""
    res: int = 16                  # subdivisions per bar
    time_signature: tuple[int, int] = (4, 4)
    tempo: float = 120.0
    bars: list[Bar] = field(default_factory=list)
    sync: list[SyncPoint] = field(default_factory=list)
    source: dict = field(default_factory=dict)   # provenance: audio file, model versions
    notes: str = ""                              # song-level annotation

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        if self.res <= 0:
            raise ChartError(f"res must be positive, got {self.res}")
        beats, unit = self.time_signature
        if self.res % beats != 0:
            raise ChartError(
                f"res {self.res} does not divide evenly into {beats} beats "
                f"of {beats}/{unit} -- pick a res that is a multiple of {beats}"
            )
        for bar in self.bars:
            bar.validate(self.res)

    @property
    def slots_per_beat(self) -> int:
        return self.res // self.time_signature[0]

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "artist": self.artist,
            "res": self.res,
            "time_signature": list(self.time_signature),
            "tempo": self.tempo,
            "notes": self.notes,
            "source": self.source,
            "sync": [asdict(s) for s in self.sync],
            "bars": [
                {
                    k: v
                    for k, v in (
                        ("n", b.n),
                        ("section", b.section),
                        ("note", b.note),
                        ("lanes", b.lanes),
                    )
                    if v is not None
                }
                for b in self.bars
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Chart":
        try:
            chart = cls(
                title=raw["title"],
                artist=raw.get("artist", ""),
                res=raw.get("res", 16),
                time_signature=tuple(raw.get("time_signature", (4, 4))),  # type: ignore[arg-type]
                tempo=raw.get("tempo", 120.0),
                notes=raw.get("notes", ""),
                source=raw.get("source", {}),
                sync=[SyncPoint(**s) for s in raw.get("sync", [])],
                bars=[
                    Bar(
                        n=b["n"],
                        lanes=b.get("lanes", {}),
                        section=b.get("section"),
                        note=b.get("note"),
                    )
                    for b in raw.get("bars", [])
                ],
            )
        except (KeyError, TypeError) as exc:
            raise ChartError(f"malformed chart: {exc}") from exc
        chart.validate()
        return chart

    def dumps(self) -> str:
        """Serialize with one bar per line.

        json.dumps with indent would put every lane on its own line and make the
        groove unreadable in a diff. The whole point of the format is that a bar
        is one glanceable row, so bars are emitted compactly and stacked.
        """
        d = self.to_dict()
        bars = d.pop("bars")
        head = json.dumps(d, indent=2, ensure_ascii=False)[:-2].rstrip().rstrip(",")
        lines = [head + ",", '  "bars": [']
        for i, bar in enumerate(bars):
            comma = "," if i < len(bars) - 1 else ""
            lines.append("    " + json.dumps(bar, ensure_ascii=False) + comma)
        lines.append("  ]")
        lines.append("}")
        return "\n".join(lines) + "\n"

    @classmethod
    def loads(cls, text: str) -> "Chart":
        return cls.from_dict(json.loads(text))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "Chart":
        return cls.loads(Path(path).read_text(encoding="utf-8"))

    # -- timing -------------------------------------------------------------

    def bar_start_time(self, bar_n: int) -> float:
        """Wall-clock seconds where a bar begins in the source recording.

        Interpolates between sync points, which is what keeps the cursor honest
        across a human-played record that drifts. Falls back to a fixed tempo
        when there are no sync points (a hand-written chart, say).
        """
        if not self.sync:
            beats, _ = self.time_signature
            return (bar_n - 1) * beats * 60.0 / self.tempo

        exact = next((s for s in self.sync if s.bar == bar_n), None)
        if exact:
            return exact.time

        before = [s for s in self.sync if s.bar < bar_n]
        after = [s for s in self.sync if s.bar > bar_n]
        beats, _ = self.time_signature

        if not before:                      # before the first anchor
            first = after[0]
            return first.time - (first.bar - bar_n) * beats * 60.0 / first.tempo
        if not after:                       # past the last anchor
            last = before[-1]
            return last.time + (bar_n - last.bar) * beats * 60.0 / last.tempo

        lo, hi = before[-1], after[0]
        span = hi.bar - lo.bar
        return lo.time + (hi.time - lo.time) * (bar_n - lo.bar) / span
