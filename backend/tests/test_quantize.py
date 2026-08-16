"""Quantizer tests.

These are the important ones. Stage 4 is the part with no off-the-shelf answer,
so it is the part most likely to be silently wrong -- and a silently wrong
quantizer produces a chart that looks plausible and is not what was played.

Everything here builds synthetic onsets with known answers, so no audio, no
models, and no GPU are involved.
"""

import numpy as np
import pytest

from dcdc.pipeline.grid import Grid
from dcdc.pipeline.quantize import quantize
from dcdc.pipeline.transcribe import Onset, Transcription


def make_grid(n_bars=4, bpm=120.0, beats_per_bar=4):
    spb = 60.0 / bpm
    beats = np.arange(n_bars * beats_per_bar + 1) * spb
    return Grid(
        beats=beats,
        downbeats=beats[::beats_per_bar],
        tempo=bpm,
        beats_per_bar=beats_per_bar,
        backend="test",
        confidence=1.0,
    )


def onsets_from_pattern(pattern, lane, grid, res=16, velocity=0.72):
    """Place onsets exactly on grid slots from a tab string, per bar."""
    out = []
    for i in range(grid.n_bars):
        start, end = grid.bar_bounds(i)
        span = end - start
        for slot, ch in enumerate(pattern):
            if ch != "-":
                out.append(Onset(time=start + span * slot / res, lane=lane, velocity=velocity))
    return out


def transcription(onsets):
    return Transcription(onsets=sorted(onsets, key=lambda o: o.time),
                         backend="test", warnings=[])


class TestReconstruction:
    def test_recovers_a_straight_rock_beat(self):
        """A groove with a sixteenth-note kick genuinely needs a 16-slot grid."""
        grid = make_grid()
        onsets = (
            onsets_from_pattern("x-x-x-x-x-x-x-x-", "hh", grid)
            + onsets_from_pattern("----o-------o---", "sd", grid)
            + onsets_from_pattern("o--o----o-------", "bd", grid)
        )
        chart, report = quantize(transcription(onsets), grid, title="T")

        assert report.res == 16
        assert len(chart.bars) == 4
        assert chart.bars[0].lanes["hh"] == "x-x-x-x-x-x-x-x-"
        assert chart.bars[0].lanes["sd"] == "----o-------o---"
        assert chart.bars[0].lanes["bd"] == "o--o----o-------"
        assert report.fit_error < 0.01

    def test_survives_human_timing_jitter(self):
        """Real onsets never land exactly on the grid. +/-15ms must not move a
        note into the wrong slot at 120bpm, where a 16th is 125ms."""
        grid = make_grid()
        rng = np.random.default_rng(7)
        onsets = onsets_from_pattern("o-o-o-o-o-o-o-o-", "hh", grid)
        for o in onsets:
            o.time += float(rng.uniform(-0.015, 0.015))

        chart, report = quantize(transcription(onsets), grid, title="T", res=16)
        assert chart.bars[0].lanes["hh"] == "x-x-x-x-x-x-x-x-"
        assert report.fit_error < 0.2


class TestResolution:
    def test_prefers_eighths_when_eighths_explain_it(self):
        """A grid finer than the music is the classic unreadable-chart failure."""
        grid = make_grid()
        onsets = onsets_from_pattern("x-x-x-x-x-x-x-x-", "hh", grid)
        _, report = quantize(transcription(onsets), grid, title="T")
        assert report.res == 8

    def test_uses_sixteenths_when_needed(self):
        grid = make_grid()
        onsets = onsets_from_pattern("xxxxxxxxxxxxxxxx", "hh", grid)
        _, report = quantize(transcription(onsets), grid, title="T")
        assert report.res == 16

    def test_detects_triplet_subdivision(self):
        grid = make_grid()
        onsets = onsets_from_pattern("xxxxxxxxxxxx", "hh", grid, res=12)
        _, report = quantize(transcription(onsets), grid, title="T")
        assert report.res == 12

    def test_one_sixteenth_note_forces_the_finer_grid(self):
        """Regression: a single unrepresentable note must not be averaged away.

        An eighth-note groove with one sixteenth-note kick has a tiny *mean*
        offset at res 8, so a mean-error metric picks the coarse grid -- and
        then silently relocates the one note that made the groove interesting.
        """
        grid = make_grid()
        onsets = (
            onsets_from_pattern("x-x-x-x-x-x-x-x-", "hh", grid)
            + onsets_from_pattern("----o-------o---", "sd", grid)
            + onsets_from_pattern("o--o----o-------", "bd", grid)   # the "o" on slot 3
        )
        chart, report = quantize(transcription(onsets), grid, title="T")

        assert report.res == 16
        assert chart.bars[0].lanes["bd"][3] == "o"

    def test_forced_resolution_is_respected(self):
        grid = make_grid()
        onsets = onsets_from_pattern("x-x-x-x-x-x-x-x-", "hh", grid)
        chart, report = quantize(transcription(onsets), grid, title="T", res=32)
        assert report.res == 32
        assert chart.res == 32
        assert len(chart.bars[0].lanes["hh"]) == 32


class TestDynamics:
    def test_quiet_snare_hits_become_ghost_notes(self):
        grid = make_grid()
        loud = [o for o in onsets_from_pattern("----o-------o---", "sd", grid)]
        quiet = [o for o in onsets_from_pattern("--o-------o-----", "sd", grid, velocity=0.2)]
        chart, _ = quantize(transcription(loud + quiet), grid, title="T", res=16)

        pattern = chart.bars[0].lanes["sd"]
        assert pattern[4] == "o", "backbeat should be a normal hit"
        assert pattern[2] == "g", "the quiet hit should read as a ghost note"

    def test_loud_hits_become_accents(self):
        grid = make_grid()
        normal = onsets_from_pattern("-x-x-x-x-x-x-x-x", "hh", grid, velocity=0.5)
        loud = onsets_from_pattern("x-------x-------", "hh", grid, velocity=1.0)
        chart, _ = quantize(transcription(normal + loud), grid, title="T", res=16)

        pattern = chart.bars[0].lanes["hh"]
        assert pattern[0] == "X"
        assert pattern[1] == "x"

    def test_quiet_kick_is_not_a_ghost_note(self):
        """Ghost notes are a snare idiom. Notating a soft kick as one is a lie."""
        grid = make_grid()
        onsets = (
            onsets_from_pattern("o-------o-------", "bd", grid, velocity=0.7)
            + onsets_from_pattern("--o-------------", "bd", grid, velocity=0.15)
        )
        chart, _ = quantize(transcription(onsets), grid, title="T")
        assert "g" not in chart.bars[0].lanes["bd"]


class TestRudiments:
    def test_close_pairs_collapse_into_a_flam(self):
        grid = make_grid()
        onsets = onsets_from_pattern("----o-------o---", "sd", grid)
        graces = [Onset(time=o.time - 0.02, lane="sd", velocity=0.4) for o in list(onsets)]
        chart, report = quantize(transcription(onsets + graces), grid, title="T")

        assert report.flams > 0
        assert "f" in chart.bars[0].lanes["sd"]
        # Without flam merging these would have become spurious 32nds.
        assert chart.bars[0].lanes["sd"].count("f") == 2

    def test_distinct_hits_are_not_merged(self):
        grid = make_grid()   # 16ths at 120bpm are 125ms apart, far above the window
        onsets = onsets_from_pattern("oooo------------", "sd", grid)
        chart, report = quantize(transcription(onsets), grid, title="T")
        assert report.flams == 0
        assert chart.bars[0].lanes["sd"].startswith("oooo")


class TestSwing:
    def test_straight_playing_reports_no_swing(self):
        grid = make_grid()
        onsets = onsets_from_pattern("xxxxxxxx", "hh", grid, res=8)
        _, report = quantize(transcription(onsets), grid, title="T")
        assert report.swing < 0.05

    def test_late_offbeats_are_reported_as_swing(self):
        grid = make_grid()
        onsets = onsets_from_pattern("xxxxxxxx", "hh", grid, res=8)
        spb = 60.0 / 120.0
        for i, o in enumerate(sorted(onsets, key=lambda x: x.time)):
            if i % 2 == 1:
                o.time += spb / 6.0        # push the "&" toward a triplet feel
        _, report = quantize(transcription(onsets), grid, title="T")

        assert report.swing > 0.1
        assert any("swing" in w for w in report.warnings)


class TestSyncPoints:
    def test_one_sync_point_per_bar(self):
        grid = make_grid(n_bars=6)
        onsets = onsets_from_pattern("o-------o-------", "bd", grid)
        chart, _ = quantize(transcription(onsets), grid, title="T")

        assert len(chart.sync) == 6
        assert [s.bar for s in chart.sync] == [1, 2, 3, 4, 5, 6]
        assert chart.sync[0].time == 0.0
        assert chart.sync[1].time == pytest.approx(2.0)

    def test_sync_points_track_a_drifting_tempo(self):
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.45, 2.9, 3.35, 3.8])
        grid = Grid(beats=beats, downbeats=beats[::4], tempo=120.0,
                    backend="test", confidence=0.9)
        onsets = [Onset(time=float(t), lane="bd", velocity=0.7) for t in beats[::4]]
        chart, _ = quantize(transcription(onsets), grid, title="T")

        assert chart.sync[0].tempo == pytest.approx(120.0, abs=1.0)
        assert chart.sync[1].tempo > 125.0


class TestFailureReporting:
    def test_empty_grid_raises(self):
        grid = Grid(beats=np.array([0.0]), downbeats=np.array([0.0]),
                    tempo=120.0, backend="test")
        with pytest.raises(ValueError, match="no complete bars"):
            quantize(transcription([]), grid, title="T")

    def test_low_grid_confidence_is_surfaced(self):
        grid = make_grid()
        grid.confidence = 0.2
        onsets = onsets_from_pattern("o-------o-------", "bd", grid)
        _, report = quantize(transcription(onsets), grid, title="T")
        assert any("irregular" in w for w in report.warnings)

    def test_collisions_are_counted_not_hidden(self):
        grid = make_grid()
        onsets = onsets_from_pattern("o---------------", "sd", grid)
        # A second hit 5ms later, outside the flam window at this granularity
        # but inside the same slot.
        extra = [Onset(time=o.time + 0.05, lane="sd", velocity=0.7) for o in list(onsets)]
        _, report = quantize(transcription(onsets + extra), grid, title="T", res=8)
        assert report.dropped > 0

    def test_transcription_warnings_propagate(self):
        grid = make_grid()
        onsets = onsets_from_pattern("o-------o-------", "bd", grid)
        t = Transcription(onsets=onsets, backend="spectral", warnings=["fallback in use"])
        _, report = quantize(t, grid, title="T")
        assert "fallback in use" in report.warnings


def test_produced_chart_always_validates():
    """Whatever the quantizer emits must be loadable by the editor."""
    grid = make_grid(n_bars=8, bpm=143.0)
    onsets = (
        onsets_from_pattern("x-x-x-x-x-x-x-x-", "hh", grid)
        + onsets_from_pattern("----o---g---o--g", "sd", grid)
        + onsets_from_pattern("o--o----o-------", "bd", grid)
    )
    chart, _ = quantize(transcription(onsets), grid, title="T", artist="A")
    chart.validate()

    from dcdc.chart import Chart
    Chart.loads(chart.dumps())
