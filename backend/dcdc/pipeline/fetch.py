"""Stage 0 -- get the audio.

The pipeline needs a waveform, and the two ways to get one are very different:

  a file you already have    always works
  a YouTube link             works via yt-dlp
  a Spotify link             cannot work, at all, ever

Spotify is not a missing feature. The Audio Analysis and Audio Features
endpoints were closed to new applications in November 2024, and the Web Playback
SDK streams through Encrypted Media Extensions specifically so that applications
cannot reach the decoded samples. There is no API tier, key, or workaround that
changes this -- so a Spotify URL is detected and explained rather than attempted.

Note that downloading from YouTube is against its terms of service, whatever the
purpose. This is a local single-user tool and that is the user's call to make.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SPOTIFY_RE = re.compile(r"(open\.)?spotify\.com|spotify:", re.I)
APPLE_MUSIC_RE = re.compile(r"music\.apple\.com", re.I)


class FetchError(RuntimeError):
    """Raised with a message meant to be shown to the user verbatim."""


@dataclass
class FetchResult:
    path: Path
    title: str
    artist: str
    duration: float | None
    source_url: str


def is_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def check_url(url: str) -> None:
    """Reject links that cannot possibly yield audio, with the reason."""
    url = url.strip()
    if not url:
        raise FetchError("No link given.")

    if SPOTIFY_RE.search(url):
        raise FetchError(
            "Spotify links cannot provide audio. The Audio Analysis API was closed "
            "to new apps in November 2024, and the Web Playback SDK is DRM-protected "
            "so the samples are unreachable. Find the song on YouTube, or upload a "
            "file you already have."
        )
    if APPLE_MUSIC_RE.search(url):
        raise FetchError(
            "Apple Music links are DRM-protected and cannot provide audio. "
            "Try YouTube, or upload a file."
        )
    if not url.startswith(("http://", "https://")):
        raise FetchError(f"That does not look like a link: {url!r}")


def fetch(url: str, dest_dir: Path, progress=None) -> FetchResult:
    """Download the best available audio stream and convert it to WAV.

    WAV rather than MP3 because everything downstream decodes it anyway, and a
    lossy round trip before source separation is a pointless quality loss on a
    file that gets deleted after transcription.
    """
    check_url(url)

    if not is_available():
        raise FetchError(
            "yt-dlp is not installed. Run:  pip install -e '.[fetch]'"
        )

    import yt_dlp

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(status: dict) -> None:
        if not progress:
            return
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            done = status.get("downloaded_bytes", 0)
            frac = (done / total) if total else 0.0
            progress("downloading audio", min(frac, 0.99))
        elif status.get("status") == "finished":
            progress("converting audio", 1.0)

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "noplaylist": True,       # a link into a playlist should fetch one song
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:                      # yt_dlp raises many types
        raise FetchError(_explain(exc)) from exc

    if info is None:
        raise FetchError("Nothing was downloaded from that link.")
    if "entries" in info:                          # a playlist slipped through
        info = info["entries"][0]

    path = dest_dir / "source.wav"
    if not path.exists():
        found = sorted(dest_dir.glob("source.*"))
        if not found:
            raise FetchError("The download reported success but produced no file.")
        path = found[0]

    return FetchResult(
        path=path,
        title=(info.get("track") or info.get("title") or "Untitled").strip(),
        # `artist` is populated for YouTube Music entries; `uploader` is the
        # channel, which is usually the right guess for everything else.
        artist=(info.get("artist") or info.get("uploader") or "").strip(),
        duration=info.get("duration"),
        source_url=info.get("webpage_url") or url,
    )


def _explain(exc: Exception) -> str:
    """Turn yt-dlp's internal errors into something worth reading."""
    text = str(exc)
    lowered = text.lower()

    if "ffmpeg" in lowered or "ffprobe" in lowered:
        return "ffmpeg is missing, which yt-dlp needs to extract audio. Run: brew install ffmpeg"
    if "private video" in lowered:
        return "That video is private."
    if "video unavailable" in lowered or "removed" in lowered:
        return "That video is unavailable or has been removed."
    if "sign in" in lowered or "age" in lowered and "restrict" in lowered:
        return "That video is age-restricted and cannot be fetched without signing in."
    if "unsupported url" in lowered:
        return "yt-dlp does not recognise that site."
    if "http error 429" in lowered or "too many requests" in lowered:
        return "Rate-limited by the host. Wait a few minutes and try again."
    if "unable to download" in lowered or "urlopen" in lowered:
        return f"Network error while downloading: {text}"
    # yt-dlp is frequently broken by upstream site changes; say so, because
    # "upgrade yt-dlp" genuinely is the fix most of the time.
    return (
        f"{text}\n\nIf this looks like a parsing failure, yt-dlp is often a version "
        "behind a site change: pip install -U yt-dlp"
    )
