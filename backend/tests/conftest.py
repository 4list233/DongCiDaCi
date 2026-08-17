import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient over a throwaway songs directory.

    main.py builds its Store at import time, so the module is reloaded after the
    env var is set. Reloading also gives each test a fresh JobRunner, which
    keeps a job in one test from leaking into the next.
    """
    monkeypatch.setenv("DCDC_SONGS_DIR", str(tmp_path / "songs"))
    from dcdc import main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as c:
        yield c
