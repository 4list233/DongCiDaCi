"""Tests for the transcription layer's mapping logic.

The models themselves need a GPU and large downloads, so what is tested here is
the part that is pure logic and the part most likely to be quietly wrong: how a
model's General MIDI output becomes lanes and velocities.
"""

import pytest

from dcdc.chart import LANE_BY_KEY
from dcdc.pipeline.transcribe import GM_TO_LANE, GM_OPEN_HAT, ADT_CLASSES


def test_every_mapped_lane_exists_in_the_chart_vocabulary():
    """A typo here would silently drop a whole voice from every transcription."""
    for note, lane in GM_TO_LANE.items():
        assert lane in LANE_BY_KEY, f"MIDI {note} maps to unknown lane {lane!r}"


def test_the_five_core_voices_are_reachable():
    mapped = set(GM_TO_LANE.values())
    for lane in ("bd", "sd", "hh"):
        assert lane in mapped, f"{lane} is unreachable from MIDI"
    assert mapped & {"lt", "mt", "ht"}, "no tom is reachable"
    assert mapped & {"cc", "rd"}, "no cymbal is reachable"


def test_standard_gm_drum_notes_map_where_a_drummer_expects():
    assert GM_TO_LANE[36] == "bd"     # bass drum 1
    assert GM_TO_LANE[38] == "sd"     # acoustic snare
    assert GM_TO_LANE[42] == "hh"     # closed hi-hat
    assert GM_TO_LANE[46] == "hh"     # open hi-hat is still the hi-hat lane
    assert GM_TO_LANE[44] == "hf"     # pedal hi-hat is the foot, not the hand
    assert GM_TO_LANE[49] == "cc"     # crash 1
    assert GM_TO_LANE[51] == "rd"     # ride 1


def test_toms_are_ordered_low_to_high():
    """41/43 are floor toms and 48/50 are the high toms. Getting this backwards
    puts every fill upside down on the staff."""
    assert GM_TO_LANE[41] == "lt"
    assert GM_TO_LANE[43] == "lt"
    assert GM_TO_LANE[48] == "ht"
    assert GM_TO_LANE[50] == "ht"


def test_open_hat_notes_are_flagged():
    assert GM_OPEN_HAT <= set(GM_TO_LANE), "open-hat notes must also be mapped"
    for note in GM_OPEN_HAT:
        assert GM_TO_LANE[note] == "hh"


def test_adt_classes_are_the_documented_five():
    assert ADT_CLASSES == ("bd", "sd", "hh", "tt", "cy")


class TestMidiVelocities:
    """Velocity is the only handle the quantizer has on ghost notes, so it has
    to survive the MIDI round trip intact."""

    def test_velocity_is_normalised_to_unit_range(self):
        pretty_midi = pytest.importorskip("pretty_midi")
        from dcdc.pipeline.transcribe import _onsets_from_midi

        midi = pretty_midi.PrettyMIDI()
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        for i, velocity in enumerate((127, 90, 25)):
            drums.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=38, start=i * 0.5, end=i * 0.5 + 0.1)
            )
        midi.instruments.append(drums)

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.mid"
            midi.write(str(path))
            onsets, unmapped = _onsets_from_midi(path)

        assert not unmapped
        assert len(onsets) == 3
        assert all(o.lane == "sd" for o in onsets)
        assert onsets[0].velocity == pytest.approx(1.0, abs=0.01)
        assert onsets[2].velocity < 0.25          # the ghost stays quiet
        assert onsets[0].time < onsets[1].time < onsets[2].time

    def test_unmapped_notes_are_reported_not_silently_dropped(self):
        pretty_midi = pytest.importorskip("pretty_midi")
        from dcdc.pipeline.transcribe import _onsets_from_midi

        midi = pretty_midi.PrettyMIDI()
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(velocity=90, pitch=38, start=0.0, end=0.1))
        drums.notes.append(pretty_midi.Note(velocity=90, pitch=70, start=0.5, end=0.6))  # maracas
        midi.instruments.append(drums)

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.mid"
            midi.write(str(path))
            onsets, unmapped = _onsets_from_midi(path)

        assert len(onsets) == 1
        assert unmapped == {70}
