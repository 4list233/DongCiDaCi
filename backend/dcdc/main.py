"""HTTP API.

Local-only, single user, no auth. Bound to 127.0.0.1 by default -- if you ever
change that, add authentication first, because every endpoint here will happily
read and write files for anyone who can reach it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chart import Chart, ChartError, LANES, ARTICULATIONS
from .jobs import JobRunner
from .store import Store, default_root
from . import pipeline
from .pipeline import drumsep, fetch, separate, transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SONGS_DIR = default_root()
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

store = Store(SONGS_DIR)
runner = JobRunner(store)

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    runner.shutdown()


app = FastAPI(title="DongCiDaCi", version="0.1.0", lifespan=lifespan)

# The Vite dev server runs on another port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- models -----------------------------------------------------------------

class CreateSong(BaseModel):
    title: str
    artist: str = ""
    source_url: str = ""


class TranscribeOptions(BaseModel):
    adt_backend: str = "auto"
    grid_backend: str = "auto"
    res: int | None = None


class RequantizeRequest(BaseModel):
    # Rotate which beat is the downbeat. This is the "the chart is displaced by
    # a beat" control, and it is the most common correction there is.
    offset_beats: int = 0
    # Nudge every onset, for detection that sits consistently early or late.
    offset_ms: float = 0.0
    res: int | None = None


class FetchRequest(BaseModel):
    url: str
    title: str = ""
    artist: str = ""
    transcribe: bool = True
    # Set to keep an existing attempt at the same link and start a separate one,
    # for comparing two transcriptions of the same recording side by side.
    duplicate: bool = False


# --- meta -------------------------------------------------------------------

@app.get("/api/health")
def health():
    """What the machine can actually do right now.

    The frontend uses this to say "spectral fallback" out loud rather than
    letting you believe you got a real transcription.
    """
    return {
        "ok": True,
        "songs_dir": str(SONGS_DIR),
        "demucs": separate.is_available(),
        "adtof": transcribe.is_adtof_available(),
        "drumsep": drumsep.is_available(),
        "ytdlp": fetch.is_available(),
        "device": separate.pick_device(),
    }


@app.get("/api/vocabulary")
def vocabulary():
    """Lanes and articulation characters, so the editor never hardcodes them."""
    return {
        "lanes": [
            {"key": l.key, "label": l.label, "midi": l.midi, "alphatex": l.alphatex}
            for l in LANES
        ],
        "articulations": [
            {"char": a.char, "name": a.name, "velocity": a.velocity,
             "open": a.open, "rudiment": a.rudiment}
            for a in ARTICULATIONS
        ],
    }


# --- songs ------------------------------------------------------------------

@app.get("/api/songs")
def list_songs():
    return [s.__dict__ for s in store.list()]


@app.post("/api/songs", status_code=201)
def create_song(body: CreateSong):
    if not body.title.strip():
        raise HTTPException(400, "title is required")
    return store.create(body.title.strip(), body.artist.strip(), body.source_url.strip()).__dict__


@app.get("/api/songs/{slug}")
def get_song(slug: str):
    song = store.get(slug)
    if not song:
        raise HTTPException(404, f"no song {slug!r}")
    return {
        **song.__dict__,
        "has_chart": store.chart_path(slug).exists(),
        "has_audio": store.audio_path(slug) is not None,
        "running": runner.is_running(slug),
    }


@app.delete("/api/songs/{slug}", status_code=204)
def delete_song(slug: str):
    if not store.get(slug):
        raise HTTPException(404, f"no song {slug!r}")
    if runner.is_running(slug):
        raise HTTPException(409, "a transcription is still running for this song")
    store.delete(slug)


@app.post("/api/songs/{slug}/audio")
async def upload_audio(slug: str, file: UploadFile = File(...)):
    if not store.get(slug):
        raise HTTPException(404, f"no song {slug!r}")
    try:
        name = store.store_audio(slug, file.filename or "source.mp3", await file.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"audio_file": name}


@app.post("/api/songs/prune")
def prune_songs(failed_only: bool = True):
    """Delete songs in bulk. By default only the ones that failed.

    Retrying a link used to leave a dead `untitled-N` behind every time, and
    clearing those one button at a time is not work anyone should do.
    """
    removed = []
    for song in store.list():
        if failed_only and song.status != "failed":
            continue
        store.delete(song.slug)
        removed.append(song.slug)
    return {"removed": removed, "count": len(removed)}


@app.post("/api/songs/from-url", status_code=202)
def create_song_from_url(body: FetchRequest):
    """Create a song from a link, download its audio, and transcribe it.

    Validates the URL synchronously so an unusable link (Spotify, Apple Music)
    fails immediately with an explanation, rather than becoming a background job
    that fails a second later where nobody is looking.
    """
    try:
        fetch.check_url(body.url)
    except fetch.FetchError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not fetch.is_available():
        raise HTTPException(
            501,
            "yt-dlp is not installed on this machine. Run: pip install -e '.[fetch]'",
        )

    url = body.url.strip()

    # Re-fetching a link means "do that again", not "leave another copy behind".
    existing = store.find_by_url(url)
    if existing and not body.duplicate:
        store.update(existing.slug, status="queued", error="")
        runner.submit_fetch(existing.slug, url, then_transcribe=body.transcribe)
        return {"slug": existing.slug, "status": "queued", "reused": True}

    song = store.create(
        body.title.strip() or "Untitled",
        body.artist.strip(),
        url,
    )
    runner.submit_fetch(song.slug, url, then_transcribe=body.transcribe)
    return {"slug": song.slug, "status": "queued", "reused": False}


@app.post("/api/songs/{slug}/fetch", status_code=202)
def fetch_audio_for_song(slug: str, body: FetchRequest):
    """Attach audio to an existing song from a link."""
    if not store.get(slug):
        raise HTTPException(404, f"no song {slug!r}")
    try:
        fetch.check_url(body.url)
    except fetch.FetchError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not fetch.is_available():
        raise HTTPException(
            501, "yt-dlp is not installed. Run: pip install -e '.[fetch]'"
        )

    if not runner.submit_fetch(slug, body.url.strip(), then_transcribe=body.transcribe):
        raise HTTPException(409, "a job is already running for this song")
    return {"status": "queued"}


@app.get("/api/songs/{slug}/audio")
def get_audio(slug: str):
    """Serve the source audio for playback. Range requests work, which is what
    lets alphaTab scrub the backing track."""
    path = store.audio_path(slug)
    if not path:
        raise HTTPException(404, "no audio uploaded for this song")
    return FileResponse(path)


@app.get("/api/songs/{slug}/stem/{name}")
def get_stem(slug: str, name: str):
    """Serve a separated stem.

    `no_drums` is the play-along track: the song with the drums removed, which
    Demucs produces in the same pass as the drum stem.
    """
    if name not in ("drums", "no_drums"):
        raise HTTPException(400, "stem must be 'drums' or 'no_drums'")
    path = store.stem_path(slug, name)
    if not path:
        raise HTTPException(404, f"no {name} stem -- run a transcription first")
    return FileResponse(path)


@app.post("/api/songs/{slug}/requantize")
def requantize(slug: str, body: RequantizeRequest):
    """Rebuild the chart from cached analysis with a different alignment.

    This is the fix for "the whole chart is displaced by a beat". It re-runs
    only the quantizer, so it returns immediately rather than re-separating and
    re-transcribing a four-minute song.
    """
    song = store.get(slug)
    if not song:
        raise HTTPException(404, f"no song {slug!r}")

    path = store.analysis_path(slug)
    if not path.exists():
        raise HTTPException(
            409,
            "no cached analysis for this song -- it predates analysis caching, "
            "so run a transcription once more",
        )

    record = pipeline.analysis.Analysis.load(path)
    chart, report = pipeline.requantize(
        record, title=song.title, artist=song.artist,
        res=body.res, offset_beats=body.offset_beats, offset_ms=body.offset_ms,
    )
    store.save_chart(slug, chart)
    return {
        "bars": len(chart.bars),
        "res": report.res,
        "fit_error": report.fit_error,
        "warnings": report.warnings,
    }


# --- transcription ----------------------------------------------------------

@app.post("/api/songs/{slug}/transcribe", status_code=202)
def start_transcription(slug: str, options: TranscribeOptions | None = None):
    song = store.get(slug)
    if not song:
        raise HTTPException(404, f"no song {slug!r}")
    if store.audio_path(slug) is None:
        raise HTTPException(400, "upload audio before transcribing")

    opts = (options or TranscribeOptions()).model_dump()
    if not runner.submit(slug, **opts):
        raise HTTPException(409, "a transcription is already running for this song")
    return {"status": "queued"}


# --- charts -----------------------------------------------------------------

@app.get("/api/songs/{slug}/chart")
def get_chart(slug: str):
    chart = store.load_chart(slug)
    if not chart:
        raise HTTPException(404, "no chart yet -- run a transcription first")
    return chart.to_dict()


@app.put("/api/songs/{slug}/chart")
def put_chart(slug: str, body: dict):
    """Save an edited chart.

    Validates before writing. A chart that fails validation is a bug in the
    editor, and silently persisting it would corrupt the file you actually care
    about, so this refuses rather than repairs.
    """
    if not store.get(slug):
        raise HTTPException(404, f"no song {slug!r}")
    try:
        chart = Chart.from_dict(body)
    except ChartError as exc:
        raise HTTPException(422, f"invalid chart: {exc}") from exc

    store.save_chart(slug, chart)
    return {"saved": True, "bars": len(chart.bars)}


@app.get("/api/songs/{slug}/raw")
def get_raw_chart(slug: str):
    """The machine's untouched first draft, for diffing against your edits."""
    path = store.raw_path(slug)
    if not path.exists():
        raise HTTPException(404, "no raw chart recorded")
    return Chart.load(path).to_dict()


@app.get("/api/songs/{slug}/notes", response_class=PlainTextResponse)
def get_notes(slug: str):
    path = store.notes_path(slug)
    return path.read_text(encoding="utf-8") if path.exists() else ""


@app.put("/api/songs/{slug}/notes")
def put_notes(slug: str, text: str = Form("")):
    if not store.get(slug):
        raise HTTPException(404, f"no song {slug!r}")
    store.notes_path(slug).write_text(text, encoding="utf-8")
    return {"saved": True}


# --- frontend ---------------------------------------------------------------

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend():
        return {
            "message": "frontend not built",
            "fix": "cd frontend && npm install && npm run build",
            "dev": "or run `npm run dev` and open http://localhost:5173",
        }
