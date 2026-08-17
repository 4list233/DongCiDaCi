"""Tests for cached analysis and grid re-alignment.

Alignment is the most common thing wrong with a first draft -- "the whole chart
is displaced by a beat" -- and fixing it must not require re-running a
four-minute separation.
"""

import numpy as np
import pytest

from dcdc.pipeline.analysis import Analysis
from dcdc.pipeline.grid import Grid
from dcdc.pipeline.transcribe import Onset
from dcdc.pipeline import requantize


def make_analysis(bpm=120.0, n_bars=4):
    spb = 60.0 / bpm
    beats = np.arange(n_bars * 4 + 1) * spb
    grid = Grid(beats=beats, downbeats=beats[::4], tempo=bpm,
                beats_per_bar=4, backend="test", confidence=1.0)
    # Kick on 1 and 3 of every bar.
    onsets = []
    for bar in range(n_bars):
        start = bar * 4 * spb
        onsets.append(Onset(time=start, lane="bd", velocity=0.8))
        onsets.append(Onset(time=start + 2 * spb, lane="bd", velocity=0.8))
    return Analysis(onsets=onsets, grid=grid, adt_backend="test",
                    grid_backend="test", warnings=[])


class TestRoundTrip:
    def test_survives_save_and_load(self, tmp_path):
        record = make_analysis()
        path = tmp_path / "analysis.json"
        record.save(path)
        restored = Analysis.load(path)

        assert len(restored.onsets) == len(record.onsets)
        assert restored.onsets[0].lane == "bd"
        assert restored.onsets[0].velocity == pytest.approx(0.8)
        assert restored.grid.tempo == 120.0
        assert len(restored.grid.downbeats) == len(record.grid.downbeats)

    def test_rebuilds_the_same_chart(self, tmp_path):
        record = make_analysis()
        record.save(tmp_path / "a.json")
        before, _ = requantize(record, title="T")
        after, _ = requantize(Analysis.load(tmp_path / "a.json"), title="T")
        assert before.dumps() == after.dumps()


class TestAlignment:
    def test_beat_offset_moves_the_barline(self):
        """The fix for a chart displaced by a beat."""
        record = make_analysis()
        straight, _ = requantize(record, title="T", res=16)
        shifted, _ = requantize(record, title="T", res=16, offset_beats=1)

        assert straight.bars[0].lanes["bd"] == "o-------o-------"
        # Rotating the downbeat by one beat moves the kick pattern by 4 slots.
        assert shifted.bars[0].lanes["bd"] != straight.bars[0].lanes["bd"]

    def test_offset_is_recorded_on_the_chart(self):
        record = make_analysis()
        chart, _ = requantize(record, title="T", offset_beats=2, offset_ms=-15.0)
        assert chart.source["offset_beats"] == 2
        assert chart.source["offset_ms"] == -15.0

    def test_millisecond_nudge_shifts_onsets(self):
        """Detection that sits consistently late should be correctable without
        re-running anything."""
        record = make_analysis()
        # Push every onset a full sixteenth late, then pull it back.
        late = record.shifted(offset_ms=125.0)
        assert late.onsets[0].time == pytest.approx(record.onsets[0].time + 0.125)

        recovered = late.shifted(offset_ms=-125.0)
        assert recovered.onsets[0].time == pytest.approx(record.onsets[0].time)

    def test_shifting_does_not_mutate_the_original(self):
        record = make_analysis()
        before = record.onsets[0].time
        record.shifted(offset_ms=500.0)
        assert record.onsets[0].time == before

    def test_requantize_can_change_resolution(self):
        record = make_analysis()
        coarse, report_c = requantize(record, title="T", res=8)
        fine, report_f = requantize(record, title="T", res=32)

        assert report_c.res == 8 and len(coarse.bars[0].lanes["bd"]) == 8
        assert report_f.res == 32 and len(fine.bars[0].lanes["bd"]) == 32


class TestBackendIsRecorded:
    def test_chart_records_which_model_ran(self):
        """A chart with no toms because it silently fell back looks identical to
        a chart from a bad model. They need different fixes, so the backend that
        actually ran is written onto the chart."""
        record = make_analysis()
        record.adt_backend = "spectral"
        chart, _ = requantize(record, title="T")
        assert chart.source["adt_backend"] == "spectral"
