"""Filesystem-backed song store.

There is no database. A song is a directory, which means the app's storage layer
and the repo layout are the same thing: charts are committed, audio is ignored,
and `git log songs/<slug>/chart.json` is the history of how your reading of a
song changed.

    songs/<slug>/
        song.json      title, artist, source link, job state
        source.<ext>   the audio you supplied (gitignored)
        stems/         demucs output (gitignored)
        chart.json     the chart -- committed, the source of truth
        raw.json       the machine's untouched first draft -- committed, never edited
        notes.md       free-form annotation
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from .chart import Chart

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text or "untitled"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Song:
    slug: str
    title: str
    artist: str = ""
    source_url: str = ""          # Spotify/YouTube link, metadata only
    audio_file: str = ""          # filename inside the song dir
    status: str = "new"           # new | queued | running | ready | failed
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    report: dict = field(default_factory=dict)
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------

    def dir(self, slug: str) -> Path:
        return self.root / slug

    def chart_path(self, slug: str) -> Path:
        return self.dir(slug) / "chart.json"

    def raw_path(self, slug: str) -> Path:
        return self.dir(slug) / "raw.json"

    def notes_path(self, slug: str) -> Path:
        return self.dir(slug) / "notes.md"

    def analysis_path(self, slug: str) -> Path:
        return self.dir(slug) / "analysis.json"

    def stem_path(self, slug: str, name: str) -> Path | None:
        """A separated stem, if it exists. `name` is 'drums' or 'no_drums'."""
        path = self.dir(slug) / "stems" / f"{name}.wav"
        return path if path.exists() else None

    def audio_path(self, slug: str) -> Path | None:
        song = self.get(slug)
        if not song or not song.audio_file:
            return None
        path = self.dir(slug) / song.audio_file
        return path if path.exists() else None

    # -- songs --------------------------------------------------------------

    def create(self, title: str, artist: str = "", source_url: str = "") -> Song:
        slug = self._unique_slug(slugify(f"{artist} {title}" if artist else title))
        self.dir(slug).mkdir(parents=True, exist_ok=True)
        song = Song(slug=slug, title=title, artist=artist, source_url=source_url)
        self.save(song)
        return song

    def get(self, slug: str) -> Song | None:
        meta = self.dir(slug) / "song.json"
        if not meta.exists():
            return None
        return Song(**json.loads(meta.read_text(encoding="utf-8")))

    def list(self) -> list[Song]:
        songs = [s for d in sorted(self.root.iterdir()) if d.is_dir()
                 for s in [self.get(d.name)] if s]
        return sorted(songs, key=lambda s: s.updated, reverse=True)

    def save(self, song: Song) -> Song:
        song.updated = _now()
        path = self.dir(song.slug) / "song.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(song), indent=2) + "\n", encoding="utf-8")
        return song

    def update(self, slug: str, **fields) -> Song:
        song = self.get(slug)
        if not song:
            raise KeyError(slug)
        for k, v in fields.items():
            setattr(song, k, v)
        return self.save(song)

    def delete(self, slug: str) -> None:
        import shutil
        shutil.rmtree(self.dir(slug), ignore_errors=True)

    # -- charts -------------------------------------------------------------

    def load_chart(self, slug: str) -> Chart | None:
        path = self.chart_path(slug)
        return Chart.load(path) if path.exists() else None

    def save_chart(self, slug: str, chart: Chart, is_raw: bool = False) -> None:
        chart.save(self.chart_path(slug))
        # raw.json is written exactly once, the first time the machine produces
        # a chart. It is the baseline every later diff is measured against, so
        # it must never be overwritten by an edit.
        if is_raw and not self.raw_path(slug).exists():
            chart.save(self.raw_path(slug))

    def store_audio(self, slug: str, filename: str, data: bytes) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise ValueError(
                f"unsupported audio format {ext!r}; expected one of {sorted(AUDIO_EXTS)}"
            )
        name = f"source{ext}"
        (self.dir(slug) / name).write_bytes(data)
        self.update(slug, audio_file=name)
        return name

    def attach_audio(self, slug: str, path: Path) -> str:
        """Register an audio file already sitting in the song directory.

        Used by the fetch stage, which writes straight into the song folder
        rather than handing bytes back through the API process.
        """
        path = Path(path)
        if path.parent.resolve() != self.dir(slug).resolve():
            raise ValueError(f"{path} is not inside the song directory for {slug!r}")
        self.update(slug, audio_file=path.name)
        return path.name

    def _unique_slug(self, base: str) -> str:
        slug, n = base, 2
        while self.dir(slug).exists():
            slug, n = f"{base}-{n}", n + 1
        return slug
