"""Stage 4 -- turn onsets and a grid into readable notation.

This is the stage with no off-the-shelf answer, and it is where the chart stops
being data and starts being music. Getting *a* chart out is easy. Getting one
that reads like something a drummer would hand you is the whole problem.

Three judgements live here:

  resolution  Is this a 16th-note groove or an 8th-note triplet groove? Guessing
              too fine produces a thicket of 32nds nobody can read; too coarse
              silently deletes notes.
  dynamics    A hit quieter than its neighbours in the same lane is a ghost note,
              and ghost notes are the groove. Velocity is the only handle we have.
  rudiments   Two snare onsets 30ms apart are a flam, not two 64th notes. Fixing
              this after quantization is impossible, so it happens before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..chart import Bar, Chart, SyncPoint, CYMBAL_LANES
from .grid import Grid
from . import musical
from .transcribe import Onset, Transcription

log = logging.getLogger(__name__)

# Candidate resolutions, as subdivisions per 4/4 bar.
#   8  = eighths      16 = sixteenths     32 = thirty-seconds
#   12 = eighth triplets                  24 = sixteenth triplets
CANDIDATE_RES = (8, 12, 16, 24, 32)

# Prefer the simpler reading. Without this the fitter always picks 32, because a
# finer grid trivially fits everything better.
COMPLEXITY_PENALTY = 0.055

# An onset further than this from any slot centre cannot be represented at that
# resolution -- quantizing it does not round it, it *moves* it somewhere it was
# never played.
MISPLACED_THRESHOLD = 0.3

# Weighted hard, because mean error alone hides exactly the failure that matters:
# a groove of eighth notes with one sixteenth-note kick averages out to a tiny
# error, and the coarse grid wins while quietly relocating that kick.
MISPLACED_WEIGHT = 2.0

# Two hits in one lane closer than this are one rudiment, not two notes.
FLAM_WINDOW_S = 0.045

# Offbeats sitting this far past the midpoint of a beat are audibly swung.
# Full triplet swing puts the "&" 1/6 (0.167) of a beat late, so the threshold
# has to sit below that or the most common case never fires.
SWING_WARN = 0.10

# Ghost and accent thresholds, relative to the lane's own reference level.
GHOST_RATIO = 0.55
ACCENT_RATIO = 1.25

# Which percentile counts as a "normal" hit for a lane. Deliberately not the
# median: in a bar full of ghost notes the median sits between the ghosts and
# the backbeats, and every backbeat then reads as an accent. The 75th percentile
# tracks the loud cluster, which is what a drummer means by "normal".
REFERENCE_PERCENTILE = 75


@dataclass
class QuantizeReport:
    res: int
    fit_error: float          # mean |offset| as a fraction of a slot, 0 = perfect
    swing: float              # 0 = straight, ~0.33 = triplet swing
    dropped: int              # onsets that collided into an occupied slot
    flams: int
    warnings: list[str]


def quantize(
    transcription: Transcription,
    grid: Grid,
    title: str,
    artist: str = "",
    res: int | None = None,
    sensitivity: float = 0.5,
) -> tuple[Chart, QuantizeReport]:
    onsets = sorted(transcription.onsets, key=lambda o: o.time)
    warnings = list(transcription.warnings) + list(grid.warnings)

    if grid.n_bars == 0:
        raise ValueError("grid has no complete bars; beat detection probably failed")

    # Reduce what was heard to what could have been played, before quantizing.
    # Doing it here rather than during detection means sensitivity can be retuned
    # from cached analysis in milliseconds, instead of re-running separation to
    # find out whether a lower setting reads better.
    onsets, discarded = musical.clean(onsets, sensitivity=sensitivity)
    warnings.extend(musical.describe(discarded, len(onsets)))

    onsets, flam_count = _merge_flams(onsets)

    if res is None:
        res, fit_error = _choose_resolution(onsets, grid)
    else:
        fit_error = _fit_error(onsets, grid, res)

    swing = _estimate_swing(onsets, grid)
    if swing > SWING_WARN:
        warnings.append(
            f"swing feel detected (offbeats sit {swing:.0%} late) -- "
            "notated straight; mark the chart as swung rather than quantizing to triplets"
        )

    thresholds = _lane_thresholds(onsets)

    bars: list[Bar] = []
    sync: list[SyncPoint] = []
    dropped = 0

    for i in range(grid.n_bars):
        start, end = grid.bar_bounds(i)
        span = end - start
        if span <= 0:
            continue

        sync.append(SyncPoint(bar=i + 1, time=round(start, 4), tempo=round(grid.local_tempo(i), 2)))

        lanes: dict[str, list[str]] = {}
        in_bar = [o for o in onsets if start <= o.time < end]

        for onset in in_bar:
            slot = int(round((onset.time - start) / span * res))
            if slot >= res:      # rounded up into the next bar
                continue
            pattern = lanes.setdefault(onset.lane, ["-"] * res)
            char = _articulation_char(onset, thresholds.get(onset.lane, 0.72))
            if pattern[slot] != "-":
                dropped += 1
                # Keep the louder of the two -- a collision usually means a
                # missed subdivision, and the accent is the one that matters.
                if _rank(char) <= _rank(pattern[slot]):
                    continue
            pattern[slot] = char

        bars.append(Bar(n=i + 1, lanes={k: "".join(v) for k, v in sorted(lanes.items())}))

    chart = Chart(
        title=title,
        artist=artist,
        res=res,
        tempo=round(grid.tempo, 2),
        bars=bars,
        sync=sync,
        source={
            "adt_backend": transcription.backend,
            "grid_backend": grid.backend,
            "grid_confidence": round(grid.confidence, 3),
        },
    )
    chart.validate()

    if fit_error > 0.22:
        warnings.append(
            f"onsets fit the {res}-slot grid poorly (mean offset {fit_error:.0%} of a slot) -- "
            "the tempo map or the downbeat phase is probably wrong"
        )
    if grid.confidence < 0.5:
        warnings.append(
            f"beat spacing is irregular (confidence {grid.confidence:.0%}) -- "
            "expect the chart to drift against the recording"
        )

    report = QuantizeReport(
        res=res,
        fit_error=round(fit_error, 4),
        swing=round(swing, 3),
        dropped=dropped,
        flams=flam_count,
        warnings=warnings,
    )
    return chart, report


# --- resolution -------------------------------------------------------------

def _choose_resolution(onsets: list[Onset], grid: Grid) -> tuple[int, float]:
    """Pick the coarsest grid that still explains where every note is.

    Scored on three terms: how tightly the notes sit on the grid, how many notes
    the grid cannot represent at all, and a bias toward the simpler reading.
    """
    if not onsets:
        return 16, 0.0

    scored = []
    for res in CANDIDATE_RES:
        offsets = np.abs(_slot_offsets(onsets, grid, res))
        if not len(offsets):
            continue
        err = float(np.mean(offsets))
        misplaced = float(np.mean(offsets > MISPLACED_THRESHOLD))
        penalty = COMPLEXITY_PENALTY * np.log2(res / CANDIDATE_RES[0])
        scored.append((err + MISPLACED_WEIGHT * misplaced + penalty, err, res))

    if not scored:
        return 16, 0.0

    scored.sort()
    _, err, res = scored[0]
    log.info("chose res=%d (mean offset %.3f of a slot)", res, err)
    return res, err


def _fit_error(onsets: list[Onset], grid: Grid, res: int) -> float:
    """Mean distance from each onset to its nearest slot, as a fraction of a slot."""
    offsets = _slot_offsets(onsets, grid, res)
    return float(np.mean(np.abs(offsets))) if len(offsets) else 0.0


def _slot_offsets(onsets: list[Onset], grid: Grid, res: int) -> np.ndarray:
    """Signed offsets in [-0.5, 0.5] slot units, one per onset that lands in a bar."""
    out = []
    for i in range(grid.n_bars):
        start, end = grid.bar_bounds(i)
        span = end - start
        if span <= 0:
            continue
        for o in onsets:
            if not (start <= o.time < end):
                continue
            pos = (o.time - start) / span * res
            out.append(pos - round(pos))
    return np.asarray(out, dtype=float)


# --- feel -------------------------------------------------------------------

def _estimate_swing(onsets: list[Onset], grid: Grid) -> float:
    """How late the offbeat eighths sit, as a fraction of a beat.

    0.0 is dead straight. Full triplet swing puts the "&" two thirds of the way
    through the beat instead of halfway, which is 1/6 (~0.167) of a beat late.

    Reported, never applied. Swing is a performance instruction -- you write
    "swung" at the top of the chart and keep the notation straight, rather than
    littering it with triplet brackets.
    """
    offsets = _slot_offsets(onsets, grid, 8)
    if len(offsets) < 8:
        return 0.0

    # Only the odd slots (the "&"s) carry swing; even slots are the beats.
    odd = []
    for i in range(grid.n_bars):
        start, end = grid.bar_bounds(i)
        span = end - start
        if span <= 0:
            continue
        for o in onsets:
            if not (start <= o.time < end):
                continue
            pos = (o.time - start) / span * 8
            if round(pos) % 2 == 1:
                odd.append(pos - round(pos))

    if len(odd) < 4:
        return 0.0
    # An eighth slot is half a beat, so a +0.5 slot offset is +0.25 of a beat.
    return float(np.clip(np.median(odd) * 0.5, 0.0, 0.5))


# --- dynamics ---------------------------------------------------------------

def _lane_thresholds(onsets: list[Onset]) -> dict[str, float]:
    """Reference 'normal' velocity per lane."""
    by_lane: dict[str, list[float]] = {}
    for o in onsets:
        by_lane.setdefault(o.lane, []).append(o.velocity)
    return {
        lane: float(np.percentile(v, REFERENCE_PERCENTILE))
        for lane, v in by_lane.items() if v
    }


def _articulation_char(onset: Onset, reference: float) -> str:
    base_hit, base_accent = ("x", "X") if onset.lane in CYMBAL_LANES else ("o", "O")

    if onset.rudiment == "flam":
        return "f"
    if reference > 0:
        if onset.velocity > reference * ACCENT_RATIO:
            return base_accent
        # Ghost notes only make sense on the snare. A quiet kick is just a quiet
        # kick, and notating it as a ghost would be a lie.
        if onset.lane == "sd" and onset.velocity < reference * GHOST_RATIO:
            return "g"
    return base_hit


_RANK = {"-": 0, "g": 1, "x": 2, "o": 2, "+": 3, "d": 3, "b": 3, "f": 4, "X": 5, "O": 5}


def _rank(char: str) -> int:
    return _RANK.get(char, 2)


# --- rudiments --------------------------------------------------------------

def _merge_flams(onsets: list[Onset]) -> tuple[list[Onset], int]:
    """Collapse near-simultaneous same-lane onsets into a single flam.

    Has to happen before quantization: once both hits snap to a grid they either
    land on the same slot (and one is silently dropped) or on adjacent slots (and
    the chart claims a 32nd note that nobody played).
    """
    if not onsets:
        return [], 0

    merged: list[Onset] = []
    count = 0
    last_by_lane: dict[str, Onset] = {}

    for onset in onsets:
        prev = last_by_lane.get(onset.lane)
        if prev is not None and onset.time - prev.time < FLAM_WINDOW_S:
            # Keep the louder hit's position; a flam's grace note leads the beat.
            prev.velocity = max(prev.velocity, onset.velocity)
            prev.rudiment = "flam"
            count += 1
            continue
        merged.append(onset)
        last_by_lane[onset.lane] = onset

    return merged, count
