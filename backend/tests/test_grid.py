"""Tests for beat-grid detection quality signals.

The trackers themselves need audio and weights, so what is covered here is the
part that decides what the chart *says about itself*: a bad confidence number or
a misdirected warning sends you debugging the wrong thing entirely.
"""

import numpy as np

from dcdc.pipeline import grid


class TestDroppedBeats:
    """A tracker that skips beats leaves double-length gaps. Saying so points at
    the tracker; saying "spacing is irregular" points at the performance, which
    is the wrong place to look."""

    def test_a_clean_grid_has_none(self):
        beats = np.arange(0, 20, 0.5)
        assert grid.dropped_beats(beats) == 0

    def test_a_single_gap_is_one_missed_beat(self):
        beats = np.array([0.0, 0.5, 1.5, 2.0, 2.5])     # 1.0 gap where 0.5 expected
        assert grid.dropped_beats(beats) == 1

    def test_a_triple_gap_is_two(self):
        beats = np.array([0.0, 0.5, 2.0, 2.5, 3.0])
        assert grid.dropped_beats(beats) == 2

    def test_a_gradual_tempo_change_is_not_a_missed_beat(self):
        """Accelerating is not skipping."""
        beats = np.cumsum([0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44])
        assert grid.dropped_beats(beats) == 0

    def test_too_few_beats_to_judge(self):
        assert grid.dropped_beats(np.array([0.0, 0.5])) == 0


class TestConfidenceIsRobust:
    def test_a_metronomic_grid_scores_high(self):
        _, conf = grid._tempo_and_confidence(np.arange(0, 20, 0.5))
        assert conf > 0.95

    def test_a_few_missed_beats_do_not_zero_the_score(self):
        """The old standard-deviation measure reported 0% confidence for a grid
        that was fine apart from three dropped beats, which reads as 'this
        recording is unusable' rather than 'the tracker slipped three times'."""
        beats = list(np.arange(0, 20, 0.5))
        for index in (10, 20, 30):
            del beats[index]
        _, conf = grid._tempo_and_confidence(np.asarray(beats))
        assert conf > 0.5, conf

    def test_genuinely_ragged_spacing_still_scores_low(self):
        rng = np.random.default_rng(0)
        beats = np.cumsum(rng.uniform(0.2, 0.9, 60))
        _, conf = grid._tempo_and_confidence(beats)
        assert conf < 0.5, conf

    def test_tempo_survives_missed_beats(self):
        """Median spacing, so a gap does not halve the reported tempo."""
        beats = list(np.arange(0, 30, 0.5))     # 120bpm
        del beats[15]
        tempo, _ = grid._tempo_and_confidence(np.asarray(beats))
        assert 118 < tempo < 122, tempo


class TestRepairBeats:
    """Filling holes matters more than reporting them: bars are built by counting
    beats from a downbeat, so one missing beat shortens a bar and displaces every
    bar after it."""

    def test_a_clean_grid_is_untouched(self):
        beats = np.arange(0, 10, 0.5)
        repaired, added = grid.repair_beats(beats)
        assert added == 0
        assert np.allclose(repaired, beats)

    def test_a_single_hole_is_filled_at_the_right_time(self):
        beats = np.array([0.0, 0.5, 1.5, 2.0])      # 1.0 missing
        repaired, added = grid.repair_beats(beats)
        assert added == 1
        assert np.allclose(repaired, [0.0, 0.5, 1.0, 1.5, 2.0])

    def test_two_consecutive_holes_are_both_filled(self):
        beats = np.array([0.0, 0.5, 2.0, 2.5])      # 1.0 and 1.5 missing
        repaired, added = grid.repair_beats(beats)
        assert added == 2
        assert np.allclose(repaired, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    def test_repair_makes_the_grid_regular(self):
        """The point of the exercise: afterwards every interval is the same."""
        beats = list(np.arange(0, 30, 0.5))
        for index in (10, 25, 40):
            del beats[index]
        repaired, added = grid.repair_beats(np.asarray(beats))
        assert added == 3
        assert np.allclose(np.diff(repaired), 0.5)
        assert grid.dropped_beats(repaired) == 0

    def test_a_tempo_change_is_not_treated_as_a_hole(self):
        """Halving the tempo genuinely doubles the interval, but every interval
        after it doubles too, so the median moves rather than a gap appearing."""
        beats = np.concatenate([np.arange(0, 5, 0.5), np.arange(5, 15, 1.0)])
        _, added = grid.repair_beats(beats)
        assert added <= 1, added

    def test_an_ambiguous_gap_is_left_alone(self):
        """1.7x the median is not a clean multiple, so guessing would be wrong."""
        beats = np.array([0.0, 0.5, 1.35, 1.85, 2.35])
        _, added = grid.repair_beats(beats)
        assert added == 0

    def test_too_few_beats_to_judge(self):
        beats = np.array([0.0, 2.0])
        repaired, added = grid.repair_beats(beats)
        assert added == 0 and len(repaired) == 2

    def test_repair_raises_confidence(self):
        beats = list(np.arange(0, 30, 0.5))
        for index in (10, 25, 40):
            del beats[index]
        before = grid._tempo_and_confidence(np.asarray(beats))[1]
        repaired, _ = grid.repair_beats(np.asarray(beats))
        after = grid._tempo_and_confidence(repaired)[1]
        assert after >= before
        assert after > 0.95
