"""In-process job runner.

Single user on one machine means no Redis, no Celery, no broker. A thread pool
with one worker is exactly right: the pipeline is GPU-bound, so running two at
once would only make both slower.

Jobs are fire-and-forget; progress is written back to the song's song.json so a
browser refresh (or a crash) never loses track of what is running.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import pipeline
from .store import Store

log = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, store: Store, max_workers: int = 1):
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dcdc")
        self._running: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, slug: str, **options) -> bool:
        """Queue a transcription. Returns False if one is already in flight."""
        with self._lock:
            if slug in self._running:
                return False
            self._running.add(slug)

        self.store.update(slug, status="queued", stage="waiting", progress=0.0, error="")
        self._pool.submit(self._run, slug, options)
        return True

    def submit_fetch(self, slug: str, url: str, then_transcribe: bool = True, **options) -> bool:
        """Queue a download, optionally chaining straight into transcription.

        Chained by default because fetching and then waiting for a second click
        is a pointless interruption -- the whole point of pasting a link is that
        you want a chart at the end of it.
        """
        with self._lock:
            if slug in self._running:
                return False
            self._running.add(slug)

        self.store.update(slug, status="queued", stage="waiting", progress=0.0, error="")
        self._pool.submit(self._run_fetch, slug, url, then_transcribe, options)
        return True

    def is_running(self, slug: str) -> bool:
        with self._lock:
            return slug in self._running

    def _run_fetch(self, slug: str, url: str, then_transcribe: bool, options: dict) -> None:
        try:
            from .pipeline import fetch as fetch_stage

            def progress(stage: str, frac: float) -> None:
                # Downloading occupies the first fifth of the bar when a
                # transcription follows it, so the progress bar stays honest.
                scaled = frac * 0.2 if then_transcribe else frac
                self.store.update(slug, status="running", stage=stage, progress=scaled)

            result = fetch_stage.fetch(url, self.store.dir(slug), progress=progress)
            self.store.attach_audio(slug, result.path)

            song = self.store.get(slug)
            updates: dict = {"source_url": result.source_url}
            # Only adopt the fetched metadata where the user left a field blank.
            if song and not song.artist and result.artist:
                updates["artist"] = result.artist
            if song and song.title.strip().lower() in ("", "untitled") and result.title:
                updates["title"] = result.title
            self.store.update(slug, **updates)

        except Exception as exc:
            log.exception("fetch failed for %s", slug)
            self.store.update(slug, status="failed", stage="", error=str(exc))
            with self._lock:
                self._running.discard(slug)
            return

        if not then_transcribe:
            self.store.update(slug, status="new", stage="", progress=0.0, error="")
            with self._lock:
                self._running.discard(slug)
            return

        # Already holding the slot, so run the transcription inline rather than
        # resubmitting -- resubmitting would deadlock against our own guard.
        self._transcribe(slug, options, progress_floor=0.2)
        with self._lock:
            self._running.discard(slug)

    def _run(self, slug: str, options: dict) -> None:
        try:
            self._transcribe(slug, options)
        finally:
            with self._lock:
                self._running.discard(slug)

    def _transcribe(self, slug: str, options: dict, progress_floor: float = 0.0) -> None:
        """Run the pipeline. Caller owns the running-slot bookkeeping."""
        try:
            song = self.store.get(slug)
            if song is None:
                raise KeyError(f"song {slug!r} disappeared")

            audio = self.store.audio_path(slug)
            if audio is None:
                raise FileNotFoundError(
                    "no audio for this song -- add a file or a link before transcribing"
                )

            span = 1.0 - progress_floor

            def progress(stage: str, frac: float) -> None:
                self.store.update(
                    slug, status="running", stage=stage,
                    progress=progress_floor + frac * span,
                )

            chart, report = pipeline.run(
                audio_path=audio,
                work_dir=self.store.dir(slug),
                title=song.title,
                artist=song.artist,
                progress=progress,
                **options,
            )

            # First machine output for this song also becomes the immutable
            # raw.json baseline.
            is_first = not self.store.raw_path(slug).exists()
            self.store.save_chart(slug, chart, is_raw=is_first)

            self.store.update(
                slug,
                status="ready",
                stage="done",
                progress=1.0,
                error="",
                report={
                    "res": report.res,
                    "fit_error": report.fit_error,
                    "swing": report.swing,
                    "dropped": report.dropped,
                    "flams": report.flams,
                    "bars": len(chart.bars),
                    "warnings": report.warnings,
                },
            )
            log.info("transcribed %s: %d bars at res %d", slug, len(chart.bars), report.res)

        except Exception as exc:
            log.exception("job failed for %s", slug)
            self.store.update(
                slug,
                status="failed",
                stage="",
                error=f"{type(exc).__name__}: {exc}",
                report={"traceback": traceback.format_exc()[-4000:]},
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
