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
    def test_every_entry_is_named_and_has_both_files(self):
        for choice in drumsep.CHECKPOINTS:
            assert choice["name"]
            assert choice["ckpt"] and choice["config"]


class TestConfigDrivenStems:
    """What the model outputs is read from its config, not guessed."""

    def _write(self, path, instruments):
        path.write_text(
            "audio:\n  sample_rate: 44100\n"
            "training:\n  instruments:\n"
            + "".join(f"    - {name}\n" for name in instruments),
            encoding="utf-8",
        )

    def test_reads_the_instrument_list(self, tmp_path):
        pytest.importorskip("yaml")
        config = tmp_path / "c.yaml"
        self._write(config, ["kick", "snare", "toms", "hi-hat", "ride", "crash"])
        assert drumsep.config_instruments(config) == [
            "kick", "snare", "toms", "hi-hat", "ride", "crash"
        ]

    def test_every_declared_stem_of_both_versions_maps(self, tmp_path):
        pytest.importorskip("yaml")
        for instruments in (
            ["kick", "snare", "toms", "hi-hat", "ride", "crash"],
            ["kick", "snare", "toms", "hi-hat", "cymbals"],
        ):
            config = tmp_path / "c.yaml"
            self._write(config, instruments)
            for name in drumsep.config_instruments(config):
                assert name.lower() in drumsep.STEM_TO_LANE, name

    def test_unreadable_config_is_not_fatal(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        assert drumsep.config_instruments(missing) == []
        junk = tmp_path / "junk.yaml"
        junk.write_text("\x00\x01 not: [valid", encoding="utf-8")
        assert drumsep.config_instruments(junk) == []

    def test_a_real_ride_stem_is_never_second_guessed(self, tmp_path):
        """The heuristic must not demote genuine ride hits when the separator
        already told ride from crash."""
        six = drumsep.DrumStems(stems={n: tmp_path / f"{n}.wav" for n in
                                       ("kick", "snare", "toms", "hi-hat", "ride", "crash")})
        assert not six.cymbals_need_splitting

        five = drumsep.DrumStems(stems={n: tmp_path / f"{n}.wav" for n in
                                        ("kick", "snare", "toms", "hi-hat", "cymbals")})
        assert five.cymbals_need_splitting


class TestMirrors:
    def test_every_file_lists_more_than_one_mirror(self):
        """The original GitHub release for this model is already gone."""
        for choice in drumsep.CHECKPOINTS:
            assert len(choice["ckpt"]) > 1
            assert len(choice["config"]) > 1

    def test_urls_point_at_the_right_file_types(self):
        for choice in drumsep.CHECKPOINTS:
            assert all(u.endswith(".ckpt") for u in choice["ckpt"])
            assert all(u.endswith((".yaml", ".yml")) for u in choice["config"])

    def test_falls_through_to_a_working_mirror(self, model_dir, monkeypatch):
        import urllib.error

        attempted = []

        def fake_urlopen(request, *a, **kw):
            url = request.full_url if hasattr(request, "full_url") else request
            attempted.append(url)
            if "dead" in url:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            import io
            return io.BytesIO(b"payload")

        monkeypatch.setattr("urllib.request.urlopen",
                            lambda r, *a, **k: _ctx(fake_urlopen(r)))
        dest = model_dir / "model.ckpt"
        ok = drumsep._fetch(
            ["https://dead.example/a.ckpt", "https://live.example/a.ckpt"], dest
        )
        assert ok and dest.read_bytes() == b"payload"
        assert len(attempted) == 2

    def test_leaves_no_partial_file_when_every_mirror_fails(self, model_dir, monkeypatch):
        import urllib.error

        def always_404(request, *a, **kw):
            url = request.full_url if hasattr(request, "full_url") else request
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", always_404)
        dest = model_dir / "model.ckpt"
        assert not drumsep._fetch(["https://a.example/x.ckpt"], dest)
        assert not dest.exists()
        assert not list(model_dir.glob("*.part"))


class _ctx:
    """Minimal context manager around a stub response."""
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        return self._obj

    def __exit__(self, *exc):
        return False


class TestReadiness:
    """Having the files is not the same as being able to run them."""

    def test_reports_each_missing_piece(self, model_dir, monkeypatch):
        monkeypatch.setattr(drumsep, "MSST_DIR", model_dir / "no-msst")
        ready, problems = drumsep.check()
        assert not ready
        joined = " ".join(problems)
        assert "inference code missing" in joined
        assert "install-drumsep" in joined

    def test_unmet_python_deps_are_named(self, model_dir, monkeypatch):
        monkeypatch.setattr(drumsep, "_MSST_IMPORTS", ("torch", "definitely_not_installed"))
        _, problems = drumsep.check()
        joined = " ".join(problems)
        assert "definitely_not_installed" in joined
        assert "[stems]" in joined

    def test_never_recommends_msst_requirements(self, model_dir, monkeypatch):
        """That file is a training manifest: it fails to build on macOS and
        downgrades librosa and demucs on the way past."""
        monkeypatch.setattr(drumsep, "_MSST_IMPORTS", ("definitely_not_installed",))
        _, problems = drumsep.check()
        assert "requirements.txt" not in " ".join(problems)

    def test_einops_is_not_required(self):
        """Only the roformer models import it; the mdx23c path never does."""
        assert "einops" not in drumsep._MSST_IMPORTS

    def test_ready_when_everything_is_present(self, model_dir, monkeypatch):
        msst = model_dir / "msst"
        msst.mkdir()
        (msst / "inference.py").write_text("x")
        monkeypatch.setattr(drumsep, "MSST_DIR", msst)
        monkeypatch.setattr(drumsep, "_MSST_IMPORTS", ())
        (model_dir / "drumsep.ckpt").write_bytes(b"x")
        (model_dir / "config_drumsep.yaml").write_text("training:\n  instruments:\n    - kick\n")

        ready, problems = drumsep.check()
        assert ready and problems == []

    def test_describe_stems_flags_unmapped_names(self, model_dir):
        pytest.importorskip("yaml")
        (model_dir / "config_drumsep.yaml").write_text(
            "training:\n  instruments:\n    - kick\n    - gong\n", encoding="utf-8"
        )
        described = drumsep.describe_stems()
        assert "kick" in described and "gong" in described
        assert "unmapped" in described

    def test_describe_stems_admits_when_it_cannot_read(self, model_dir):
        (model_dir / "config_drumsep.yaml").write_text("\x00 [broken", encoding="utf-8")
        assert "unknown" in drumsep.describe_stems()

    def test_no_config_describes_nothing(self, model_dir):
        assert drumsep.describe_stems() == ""
