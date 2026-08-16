"""API smoke tests.

No audio and no models -- these check that the HTTP surface, the store, and
chart validation hold together. The pipeline itself is covered by
test_quantize.py.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DCDC_SONGS_DIR", str(tmp_path / "songs"))
    # Import after the env var is set: main.py builds its Store at import time.
    import importlib
    from dcdc import main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as c:
        yield c


def make_chart(title="Test"):
    return {
        "title": title,
        "artist": "Nobody",
        "res": 16,
        "time_signature": [4, 4],
        "tempo": 120.0,
        "sync": [{"bar": 1, "time": 0.0, "tempo": 120.0}],
        "bars": [{"n": 1, "lanes": {
            "hh": "x-x-x-x-x-x-x-x-",
            "sd": "----o-------o---",
            "bd": "o-------o-------",
        }}],
    }


def test_health_reports_capabilities(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    # These are False in CI and that is the point -- the UI must be able to say so.
    assert "demucs" in body and "adtof" in body
    assert body["device"] in {"cpu", "mps", "cuda"}


def test_vocabulary_matches_the_backend_model(client):
    from dcdc.chart import LANES
    body = client.get("/api/vocabulary").json()
    assert [l["key"] for l in body["lanes"]] == [l.key for l in LANES]
    assert any(a["char"] == "g" for a in body["articulations"])


def test_song_lifecycle(client):
    created = client.post("/api/songs", json={"title": "Tom Sawyer", "artist": "Rush"})
    assert created.status_code == 201
    slug = created.json()["slug"]
    assert slug == "rush-tom-sawyer"

    assert client.get(f"/api/songs/{slug}").json()["title"] == "Tom Sawyer"
    assert len(client.get("/api/songs").json()) == 1

    assert client.delete(f"/api/songs/{slug}").status_code == 204
    assert client.get(f"/api/songs/{slug}").status_code == 404


def test_slugs_do_not_collide(client):
    a = client.post("/api/songs", json={"title": "Song"}).json()["slug"]
    b = client.post("/api/songs", json={"title": "Song"}).json()["slug"]
    assert a != b


def test_chart_save_and_load_roundtrip(client):
    slug = client.post("/api/songs", json={"title": "T"}).json()["slug"]

    saved = client.put(f"/api/songs/{slug}/chart", json=make_chart())
    assert saved.status_code == 200
    assert saved.json()["bars"] == 1

    loaded = client.get(f"/api/songs/{slug}/chart").json()
    assert loaded["bars"][0]["lanes"]["hh"] == "x-x-x-x-x-x-x-x-"


def test_invalid_chart_is_refused_not_repaired(client):
    """Persisting a malformed chart would corrupt the one file that matters."""
    slug = client.post("/api/songs", json={"title": "T"}).json()["slug"]
    bad = make_chart()
    bad["bars"][0]["lanes"]["hh"] = "x-x-"      # wrong length for res 16

    res = client.put(f"/api/songs/{slug}/chart", json=bad)
    assert res.status_code == 422
    assert "slots" in res.json()["detail"]
    assert client.get(f"/api/songs/{slug}/chart").status_code == 404


def test_transcribe_without_audio_is_rejected(client):
    slug = client.post("/api/songs", json={"title": "T"}).json()["slug"]
    res = client.post(f"/api/songs/{slug}/transcribe", json={})
    assert res.status_code == 400
    assert "upload audio" in res.json()["detail"]


def test_unsupported_audio_format_is_rejected(client):
    slug = client.post("/api/songs", json={"title": "T"}).json()["slug"]
    res = client.post(
        f"/api/songs/{slug}/audio",
        files={"file": ("drums.txt", b"not audio", "text/plain")},
    )
    assert res.status_code == 400


def test_missing_chart_is_a_404_not_an_empty_chart(client):
    slug = client.post("/api/songs", json={"title": "T"}).json()["slug"]
    assert client.get(f"/api/songs/{slug}/chart").status_code == 404
    assert client.get(f"/api/songs/{slug}/raw").status_code == 404


def test_raw_chart_is_never_overwritten_by_edits(client, tmp_path):
    """raw.json is the baseline every diff is measured against."""
    from dcdc.chart import Chart
    from dcdc.store import Store

    store = Store(tmp_path / "songs")
    song = store.create("T")
    machine = Chart.from_dict(make_chart())
    store.save_chart(song.slug, machine, is_raw=True)

    edited = Chart.from_dict(make_chart())
    edited.bars[0].lanes["sd"] = "----o---g---o---"
    store.save_chart(song.slug, edited, is_raw=True)

    assert Chart.load(store.raw_path(song.slug)).bars[0].lanes["sd"] == "----o-------o---"
    assert Chart.load(store.chart_path(song.slug)).bars[0].lanes["sd"] == "----o---g---o---"
