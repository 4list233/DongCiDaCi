"""Tests for kit separation and per-stem detection."""

import numpy as np
import pytest

from dcdc.pipeline import drumsep
from dcdc.pipeline.onsets import detect_stem, split_cymbals, split_toms
from dcdc.pipeline.transcribe import Onset


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(drumsep, "MODEL_DIR", tmp_path)
    return tmp_path


class TestCheckpointDiscovery:
    def test_finds_the_published_capitalised_filename(self, model_dir):
        """The real checkpoint is spelled `...DrumSep_model...`. A case-sensitive
        glob finds nothing, which looks exactly like it not being installed."""
        (model_dir / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt").write_bytes(b"x")
        (model_dir / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml").write_text("x")

        assert drumsep._find_checkpoint() is not None
        assert drumsep._find_config() is not None

    def test_ignores_empty_files(self, model_dir):
        """A zero-byte file is an interrupted download, not an install."""
        (model_dir / "drumsep.ckpt").write_bytes(b"")
        assert drumsep._find_checkpoint() is None

    def test_ignores_partial_downloads(self, model_dir):
        (model_dir / "drumsep.ckpt.part").write_bytes(b"xxxx")
        assert drumsep._find_checkpoint() is None

    def test_prefers_a_drumsep_named_file(self, model_dir):
        (model_dir / "some_other_model.ckpt").write_bytes(b"x")
        (model_dir / "MDX23C-DrumSep-aufr33.ckpt").write_bytes(b"x")
        assert "DrumSep" in drumsep._find_checkpoint().name

    def test_missing_dir_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(drumsep, "MODEL_DIR", tmp_path / "nope")
        assert drumsep._find_checkpoint() is None
        assert not drumsep.is_available()


class TestStemNaming:
    def test_every_published_stem_name_maps(self):
        """Both published models must map cleanly, hyphen included."""
        for name in ("kick", "snare", "toms", "hi-hat", "ride", "crash", "cymbals"):
            assert name in drumsep.STEM_TO_LANE, f"{name!r} has no lane"

    def test_six_stem_model_needs_no_cymbal_guessing(self, tmp_path):
        stems = drumsep.DrumStems(stems={
            "kick": tmp_path / "k.wav", "ride": tmp_path / "r.wav",
            "crash": tmp_path / "c.wav",
        })
        assert not stems.cymbals_need_splitting
        lanes = stems.lane_stems()
        assert lanes["rd"].name == "r.wav" and lanes["cc"].name == "c.wav"

    def test_five_stem_model_does(self, tmp_path):
        stems = drumsep.DrumStems(stems={"cymbals": tmp_path / "cy.wav"})
        assert stems.cymbals_need_splitting

    def test_empty_result_is_falsy(self):
        assert not drumsep.DrumStems()


class TestCymbalSplit:
    def test_even_repeated_hits_read_as_a_ride(self):
        onsets = [Onset(time=i * 0.25, lane="rd", velocity=0.6) for i in range(16)]
        split_cymbals(onsets)
        assert all(o.lane == "rd" for o in onsets)

    def test_a_loud_isolated_hit_reads_as_a_crash(self):
        onsets = [Onset(time=i * 0.25, lane="rd", velocity=0.5) for i in range(8)]
        onsets.append(Onset(time=6.0, lane="rd", velocity=1.0))   # loud, after a gap
        split_cymbals(onsets)
        assert onsets[-1].lane == "cc"

    def test_too_few_hits_to_judge_are_left_alone(self):
        onsets = [Onset(time=0.0, lane="rd", velocity=0.9)]
        split_cymbals(onsets)
        assert onsets[0].lane == "cc"        # a lone cymbal reads as a crash


class TestVelocity:
    def _click_track(self, sr, times, amps):
        y = np.zeros(int(sr * (max(times) + 1.0)), dtype=np.float32)
        for t, amp in zip(times, amps):
            i = int(t * sr)
            decay = np.exp(-np.linspace(0, 12, int(sr * 0.08)))
            y[i:i + len(decay)] += (amp * decay * np.random.default_rng(0).standard_normal(len(decay))).astype(np.float32)
        return y

    def test_a_quiet_hit_gets_a_lower_velocity(self):
        """A ghost note is quiet relative to other snare hits. This is the whole
        reason detection runs per stem rather than on a mixed kit."""
        pytest.importorskip("librosa")
        sr = 22050
        times = [0.5, 1.0, 1.5, 2.0]
        y = self._click_track(sr, times, [1.0, 0.15, 1.0, 0.15])

        found = detect_stem(y, sr, "sd")
        assert len(found) >= 4

        loud = [o.velocity for o in found if abs(o.time - 0.5) < 0.05 or abs(o.time - 1.5) < 0.05]
        quiet = [o.velocity for o in found if abs(o.time - 1.0) < 0.05 or abs(o.time - 2.0) < 0.05]
        assert loud and quiet
        assert min(loud) > max(quiet)

    def test_a_silent_stem_yields_nothing(self):
        """Separating a kit with no toms still produces a toms stem full of bleed."""
        pytest.importorskip("librosa")
        assert detect_stem(np.zeros(22050, dtype=np.float32), 22050, "mt") == []


class TestTomSplit:
    def test_pitch_orders_the_toms(self):
        """Fills descend, so getting this backwards turns every fill upside down."""
        pytest.importorskip("librosa")
        sr = 22050
        times, freqs = [0.2, 0.6, 1.0], [90.0, 160.0, 260.0]
        y = np.zeros(int(sr * 1.5), dtype=np.float32)
        for t, f in zip(times, freqs):
            i = int(t * sr)
            n = int(sr * 0.15)
            env = np.exp(-np.linspace(0, 8, n))
            y[i:i + n] += (np.sin(2 * np.pi * f * np.arange(n) / sr) * env).astype(np.float32)

        onsets = [Onset(time=t, lane="mt", velocity=0.8) for t in times]
        split_toms(y, sr, onsets)
        assert onsets[0].lane == "lt"      # lowest -> floor tom
        assert onsets[2].lane == "ht"      # highest -> high tom

    def test_no_onsets_is_safe(self):
        assert split_toms(np.zeros(100, dtype=np.float32), 22050, []) == []


class TestCheckpointCatalogue:
    def test_the_default_is_the_six_stem_model(self):
        best = drumsep.CHECKPOINTS[0]
        assert "ride" in best["stems"] and "crash" in best["stems"]

    def test_every_entry_has_both_urls(self):
        for choice in drumsep.CHECKPOINTS:
            assert choice["ckpt"].endswith(".ckpt")
            assert choice["config"].endswith((".yaml", ".yml"))
