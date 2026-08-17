"""Tests for reducing detected sound to a playable part.

Detection answers "what made a noise". A chart answers "what was the drummer
doing". These tests cover the gap, which is physical rather than acoustic: stems
leak, a body has four limbs, and a hit nobody could hear is not part of the part.
"""

import pytest

from dcdc.pipeline import musical
from dcdc.pipeline.transcribe import Onset


def hit(time, lane, velocity=0.8):
    return Onset(time=time, lane=lane, velocity=velocity)


class TestBleedBetweenStems:
    """Separation is not clean: the toms stem carries kick and snare leakage.
    Detected at a low threshold that leakage becomes notes -- faithful to the
    audio and wrong about the performance."""

    def test_a_quiet_tom_under_a_loud_kick_is_dropped(self):
        onsets = [hit(1.0, "bd", 0.95), hit(1.005, "mt", 0.12)]
        kept, removed = musical.clean(onsets, sensitivity=1.0)
        assert [o.lane for o in kept] == ["bd"]
        assert removed["bleed"] == 1

    def test_a_real_kick_and_crash_together_both_survive(self):
        """Both are struck with intent, so both land hard."""
        onsets = [hit(1.0, "bd", 0.9), hit(1.004, "cc", 0.85)]
        kept, _ = musical.clean(onsets, sensitivity=1.0)
        assert {o.lane for o in kept} == {"bd", "cc"}

    def test_hits_far_apart_are_never_compared(self):
        """A quiet tom on its own beat is a real note, not leakage."""
        onsets = [hit(1.0, "bd", 0.95), hit(1.4, "mt", 0.2)]
        kept, removed = musical.clean(onsets, sensitivity=1.0)
        assert len(kept) == 2 and removed["bleed"] == 0

    def test_the_loudest_voice_of_an_instant_always_survives(self):
        onsets = [hit(2.0, "sd", 0.3), hit(2.002, "hh", 0.28)]
        kept, _ = musical.clean(onsets, sensitivity=1.0)
        assert "sd" in {o.lane for o in kept}


class TestLimbs:
    """A drummer has two hands and two feet. Five simultaneous voices is not a
    hard part, it is an impossible one -- so it is evidence of over-detection."""

    def test_a_third_hand_is_refused(self):
        onsets = [hit(1.0, "sd", 0.9), hit(1.0, "hh", 0.9),
                  hit(1.0, "mt", 0.9), hit(1.0, "ht", 0.9)]
        kept, removed = musical.clean(onsets, sensitivity=1.0)
        hands = [o for o in kept if o.lane not in musical.FOOT_LANES]
        assert len(hands) == 2
        assert removed["limbs"] == 2

    def test_the_backbeat_outranks_the_hat(self):
        """When a hand has to choose, the snare is the part."""
        onsets = [hit(1.0, "hh", 0.9), hit(1.0, "mt", 0.9), hit(1.0, "sd", 0.9)]
        kept, _ = musical.clean(onsets, sensitivity=1.0)
        assert "sd" in {o.lane for o in kept}

    def test_both_feet_may_play_at_once(self):
        """Kick and hi-hat pedal are different feet, so this is playable."""
        onsets = [hit(1.0, "bd", 0.9), hit(1.0, "hf", 0.85),
                  hit(1.0, "sd", 0.9), hit(1.0, "hh", 0.9)]
        kept, _ = musical.clean(onsets, sensitivity=1.0)
        assert {o.lane for o in kept} == {"bd", "hf", "sd", "hh"}

    def test_one_foot_cannot_play_twice_at_once(self):
        onsets = [hit(1.0, "bd", 0.5), hit(1.004, "bd", 0.9)]
        kept, removed = musical.clean(onsets, sensitivity=1.0)
        assert len(kept) == 1
        assert kept[0].velocity == pytest.approx(0.9)   # keeps the louder
        assert removed["limbs"] == 1

    def test_a_full_groove_is_left_intact(self):
        """Kick, snare and hat together is the most common thing in drumming and
        must never be touched."""
        onsets = [hit(1.0, "bd"), hit(1.0, "hh"), hit(1.5, "sd"), hit(1.5, "hh")]
        kept, removed = musical.clean(onsets, sensitivity=1.0)
        assert len(kept) == 4
        assert sum(removed.values()) == 0


class TestSensitivity:
    def test_high_sensitivity_keeps_quiet_hits(self):
        onsets = [hit(1.0, "sd", 1.0), hit(2.0, "sd", 0.18)]
        kept, _ = musical.clean(onsets, sensitivity=1.0)
        assert len(kept) == 2

    def test_low_sensitivity_keeps_only_the_clear_hits(self):
        onsets = [hit(1.0, "sd", 1.0), hit(2.0, "sd", 0.18)]
        kept, removed = musical.clean(onsets, sensitivity=0.0)
        assert len(kept) == 1 and removed["quiet"] == 1

    def test_the_default_keeps_ghost_notes(self):
        """A ghost note is around a third of a backbeat and is the groove, so the
        default must not remove it."""
        onsets = [hit(1.0, "sd", 1.0), hit(1.5, "sd", 0.35), hit(2.0, "sd", 0.95)]
        kept, _ = musical.clean(onsets, sensitivity=0.5)
        assert len(kept) == 3

    def test_the_floor_is_per_lane(self):
        """Loudness only means something against the same drum: a quiet hat is
        normal, and must not be judged against a kick."""
        onsets = [hit(1.0, "bd", 1.0), hit(3.0, "hh", 0.22), hit(3.5, "hh", 0.25)]
        kept, _ = musical.clean(onsets, sensitivity=0.5)
        assert {o.lane for o in kept} == {"bd", "hh"}

    def test_out_of_range_values_are_clamped(self):
        onsets = [hit(1.0, "sd", 0.9)]
        assert len(musical.clean(onsets, sensitivity=5.0)[0]) == 1
        assert len(musical.clean(onsets, sensitivity=-3.0)[0]) == 1


class TestReporting:
    def test_it_admits_what_it_removed(self):
        notes = musical.describe({"bleed": 12, "limbs": 3, "quiet": 40}, kept=200)
        joined = " ".join(notes)
        assert "12" in joined and "3" in joined and "40" in joined

    def test_it_says_so_when_it_removed_more_than_it_kept(self):
        notes = musical.describe({"bleed": 90, "limbs": 10, "quiet": 5}, kept=40)
        assert any("lower sensitivity" in n for n in notes)

    def test_a_clean_pass_says_nothing(self):
        assert musical.describe({"bleed": 0, "limbs": 0, "quiet": 0}, kept=100) == []


class TestEmptyInput:
    def test_no_onsets_is_safe(self):
        kept, removed = musical.clean([])
        assert kept == [] and sum(removed.values()) == 0


class TestBleedOnlyLanes:
    """The flaw that made phantom toms survive everything else.

    detect_stem normalises velocity against each stem's own peak, so a stem
    holding nothing but leakage normalises that leakage to 1.0. Judged on
    velocity alone the lane looks like a drum being hit hard all track.
    """

    def test_a_lane_of_pure_bleed_is_removed_entirely(self):
        onsets = [hit(i * 0.5, "bd", 0.9) for i in range(8)]
        for o in onsets:
            o.level = 0.9
        # A toms stem with only leakage: velocity near full, level tiny.
        for i in range(8):
            ghost = hit(i * 0.5 + 0.2, "mt", 1.0)
            ghost.level = 0.03
            onsets.append(ghost)

        kept, removed = musical.clean(sorted(onsets, key=lambda o: o.time), sensitivity=1.0)
        assert {o.lane for o in kept} == {"bd"}
        assert removed["lanes"] == 8

    def test_a_genuinely_quiet_but_real_lane_survives(self):
        """A ride played softly under a loud mix is still being played."""
        onsets = []
        for i in range(8):
            k = hit(i * 0.5, "bd", 0.9); k.level = 0.9
            r = hit(i * 0.5 + 0.25, "rd", 0.8); r.level = 0.3
            onsets += [k, r]

        kept, removed = musical.clean(sorted(onsets, key=lambda o: o.time), sensitivity=1.0)
        assert {o.lane for o in kept} == {"bd", "rd"}
        assert removed["lanes"] == 0

    def test_bleed_within_an_instant_uses_absolute_level(self):
        """Per-lane velocity would rate the leak as loud as the kick."""
        kick = hit(1.0, "bd", 0.9); kick.level = 0.9
        leak = hit(1.004, "mt", 0.95); leak.level = 0.1
        # A second real tom hit elsewhere, so the lane is not dropped wholesale.
        real = hit(3.0, "mt", 1.0); real.level = 0.6

        kept, removed = musical.clean([kick, leak, real], sensitivity=1.0)
        assert removed["bleed"] == 1
        assert sorted(o.time for o in kept) == [1.0, 3.0]

    def test_velocity_only_input_still_works(self):
        """The ADTOF path reports MIDI velocity and no absolute level."""
        onsets = [hit(1.0, "bd", 0.9), hit(1.5, "sd", 0.8), hit(2.0, "mt", 0.7)]
        kept, removed = musical.clean(onsets, sensitivity=1.0)
        assert len(kept) == 3 and removed["lanes"] == 0
