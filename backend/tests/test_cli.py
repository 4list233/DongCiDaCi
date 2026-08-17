"""Tests for the command line.

Covers the failure that is hardest to notice: the server serving an interface
build older than the code that was pulled, so a change appears not to work.
"""

import os
import time
import types

from dcdc import cli


class TestStaleFrontend:
    """The server serves frontend/dist, which is gitignored. Pulling new
    interface code and serving the previous build looks exactly like the change
    not working, which is the worst way for it to fail."""

    def _frontend(self, tmp_path, monkeypatch):
        root = tmp_path / "repo" / "frontend"
        (root / "src").mkdir(parents=True)
        (root / "dist").mkdir()
        (root / "index.html").write_text("x")
        (root / "src" / "main.js").write_text("x")
        # cli resolves the frontend relative to its own file.
        monkeypatch.setattr(cli, "__file__", str(tmp_path / "repo" / "backend" / "dcdc" / "cli.py"))
        return root

    def test_a_current_build_is_left_alone(self, tmp_path, monkeypatch, capsys):
        root = self._frontend(tmp_path, monkeypatch)
        built = root / "dist" / "index.html"
        built.write_text("built")
        os.utime(built, (time.time() + 10, time.time() + 10))

        cli._ensure_frontend_built()
        assert "rebuild" not in capsys.readouterr().err.lower()

    def test_a_missing_build_is_noticed(self, tmp_path, monkeypatch, capsys):
        self._frontend(tmp_path, monkeypatch)
        monkeypatch.setattr(cli.shutil, "which", lambda _: None)

        cli._ensure_frontend_built()
        assert "no build found" in capsys.readouterr().err

    def test_a_stale_build_is_noticed(self, tmp_path, monkeypatch, capsys):
        root = self._frontend(tmp_path, monkeypatch)
        built = root / "dist" / "index.html"
        built.write_text("built")
        os.utime(built, (time.time() - 600, time.time() - 600))
        monkeypatch.setattr(cli.shutil, "which", lambda _: None)

        cli._ensure_frontend_built()
        assert "has changed" in capsys.readouterr().err

    def test_a_failed_rebuild_does_not_stop_the_server(self, tmp_path, monkeypatch, capsys):
        """Serving the old page beats refusing to start."""
        self._frontend(tmp_path, monkeypatch)
        monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/npm")
        monkeypatch.setattr(
            cli.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )

        cli._ensure_frontend_built()          # must not raise
        assert "rebuild failed" in capsys.readouterr().err
