import json

import pytest

from dcdc.chart import Bar, Chart, ChartError, SyncPoint


def rock_bar(n=1):
    return Bar(n=n, lanes={
        "hh": "x-x-x-x-x-x-x-x-",
        "sd": "----o-------o---",
        "bd": "o-------o-------",
    })


def test_roundtrip_preserves_everything():
    chart = Chart(
        title="Test", artist="Nobody", tempo=96.0,
        bars=[rock_bar(1), rock_bar(2)],
        sync=[SyncPoint(bar=1, time=0.5, tempo=96.0)],
    )
    restored = Chart.loads(chart.dumps())

    assert restored.title == "Test"
    assert restored.tempo == 96.0
    assert len(restored.bars) == 2
    assert restored.bars[0].lanes["hh"] == "x-x-x-x-x-x-x-x-"
    assert restored.sync[0].time == 0.5


def test_dumps_puts_one_bar_per_line():
    """The format only earns its keep if a bar is one glanceable row in a diff."""
    chart = Chart(title="T", bars=[rock_bar(1), rock_bar(2), rock_bar(3)])
    bar_lines = [ln for ln in chart.dumps().splitlines() if '"n":' in ln]
    assert len(bar_lines) == 3
    assert '"hh": "x-x-x-x-x-x-x-x-"' in bar_lines[0]


def test_dumps_is_valid_json():
    chart = Chart(title="T", bars=[rock_bar(1)], notes="quote \" and unicode ü")
    json.loads(chart.dumps())


def test_wrong_length_lane_is_rejected():
    chart = Chart(title="T", res=16, bars=[Bar(n=1, lanes={"hh": "x-x-"})])
    with pytest.raises(ChartError, match="4 slots, expected 16"):
        chart.validate()


def test_unknown_articulation_is_rejected():
    chart = Chart(title="T", res=4, bars=[Bar(n=1, lanes={"sd": "o?o-"})])
    with pytest.raises(ChartError, match="unknown articulation"):
        chart.validate()


def test_unknown_lane_is_rejected():
    chart = Chart(title="T", res=4, bars=[Bar(n=1, lanes={"cowbell": "o-o-"})])
    with pytest.raises(ChartError, match="unknown lane"):
        chart.validate()


def test_res_must_divide_into_beats():
    with pytest.raises(ChartError, match="does not divide evenly"):
        Chart(title="T", res=10, time_signature=(4, 4)).validate()


def test_hits_are_yielded_in_time_order():
    bar = rock_bar()
    slots = [slot for slot, _, _ in bar.hits()]
    assert slots == sorted(slots)
    assert (0, "hh") in [(s, l) for s, l, _ in bar.hits()]
    assert (0, "bd") in [(s, l) for s, l, _ in bar.hits()]


def test_ghost_note_carries_low_velocity():
    bar = Bar(n=1, lanes={"sd": "g---o---O---g---"})
    velocities = {slot: art.velocity for slot, lane, art in bar.hits() if lane == "sd"}
    assert velocities[0] < velocities[4] < velocities[8]


class TestBarTiming:
    def test_uses_fixed_tempo_without_sync_points(self):
        chart = Chart(title="T", tempo=120.0)
        assert chart.bar_start_time(1) == 0.0
        assert chart.bar_start_time(3) == pytest.approx(4.0)   # 2 bars at 2s each

    def test_interpolates_between_sync_points(self):
        chart = Chart(title="T", sync=[
            SyncPoint(bar=1, time=0.0, tempo=120.0),
            SyncPoint(bar=5, time=8.0, tempo=120.0),
        ])
        assert chart.bar_start_time(3) == pytest.approx(4.0)

    def test_extrapolates_past_the_last_sync_point(self):
        chart = Chart(title="T", tempo=120.0, sync=[SyncPoint(bar=1, time=1.0, tempo=120.0)])
        # One bar of 4/4 at 120bpm is 2 seconds.
        assert chart.bar_start_time(2) == pytest.approx(3.0)

    def test_extrapolates_before_the_first_sync_point(self):
        chart = Chart(title="T", sync=[SyncPoint(bar=3, time=5.0, tempo=120.0)])
        assert chart.bar_start_time(1) == pytest.approx(1.0)

    def test_handles_tempo_drift(self):
        """Sync points are the whole reason a chart stays locked to a human
        performance that speeds up."""
        chart = Chart(title="T", sync=[
            SyncPoint(bar=1, time=0.0, tempo=100.0),
            SyncPoint(bar=9, time=18.0, tempo=110.0),
        ])
        mid = chart.bar_start_time(5)
        assert 8.5 < mid < 9.5
