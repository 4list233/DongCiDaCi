"""Tests for link handling.

No network. What is tested is the part that must be right before any network
call happens: refusing links that cannot possibly work, and saying why.
"""

import types

import pytest

from dcdc.pipeline import fetch


class TestUnusableLinks:
    """Spotify and Apple Music are not 'not yet supported' -- they are
    impossible, and the error has to say so or it reads as a bug."""

    @pytest.mark.parametrize("url", [
        "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT",
        "http://open.spotify.com/album/1234",
        "spotify:track:4cOdK2wGLETKBW3PvgPWqT",
        "https://OPEN.SPOTIFY.COM/track/abc",
    ])
    def test_spotify_is_rejected_with_a_reason(self, url):
        with pytest.raises(fetch.FetchError) as exc:
            fetch.check_url(url)
        message = str(exc.value)
        assert "DRM" in message or "Audio Analysis" in message
        assert "YouTube" in message, "the error should say what to do instead"

    def test_apple_music_is_rejected(self):
        with pytest.raises(fetch.FetchError, match="DRM"):
            fetch.check_url("https://music.apple.com/us/album/scarlet/123456")

    def test_empty_url_is_rejected(self):
        with pytest.raises(fetch.FetchError, match="No link"):
            fetch.check_url("   ")

    def test_non_url_text_is_rejected(self):
        with pytest.raises(fetch.FetchError, match="does not look like a link"):
            fetch.check_url("scarlet by periphery")


class TestUsableLinks:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://soundcloud.com/artist/track",
        "https://bandcamp.com/track/whatever",
    ])
    def test_accepted(self, url):
        fetch.check_url(url)      # must not raise


class TestErrorMessages:
    """yt-dlp's own errors are unreadable; these get surfaced to the user."""

    def test_missing_ffmpeg_is_named(self):
        assert "ffmpeg" in fetch._explain(Exception("ffprobe/ffmpeg not found")).lower()

    def test_private_video(self):
        assert "private" in fetch._explain(Exception("ERROR: Private video")).lower()

    def test_unavailable_video(self):
        assert "unavailable" in fetch._explain(Exception("Video unavailable")).lower()

    def test_rate_limit(self):
        assert "rate-limited" in fetch._explain(Exception("HTTP Error 429")).lower()

    def test_unknown_errors_suggest_upgrading(self):
        """yt-dlp is usually broken by an upstream site change, and upgrading
        genuinely is the fix, so an unrecognised error should say so."""
        assert "pip install -U yt-dlp" in fetch._explain(Exception("some parser exploded"))


class TestApiSurface:
    def test_spotify_url_fails_fast_with_400(self, client):
        res = client.post("/api/songs/from-url", json={
            "url": "https://open.spotify.com/track/abc",
        })
        assert res.status_code == 400
        assert "DRM" in res.json()["detail"] or "Audio Analysis" in res.json()["detail"]

    def test_no_song_is_created_for_a_rejected_url(self, client):
        client.post("/api/songs/from-url", json={"url": "https://open.spotify.com/track/abc"})
        assert client.get("/api/songs").json() == []

    def test_missing_ytdlp_reports_501_not_500(self, client, monkeypatch):
        """A missing optional dependency is a configuration problem, and the
        response should say which command fixes it."""
        monkeypatch.setattr(fetch, "is_available", lambda: False)
        res = client.post("/api/songs/from-url", json={
            "url": "https://youtu.be/dQw4w9WgXcQ",
        })
        assert res.status_code == 501
        assert "pip install" in res.json()["detail"]

    def test_health_reports_ytdlp(self, client):
        assert "ytdlp" in client.get("/api/health").json()


class TestExplains403:
    """yt-dlp words a 403 as "unable to download video data", which read as a
    network failure and sent people to check their connection. A 403 from a video
    host nearly always means yt-dlp is behind a site change."""

    def test_a_403_points_at_yt_dlp_not_the_connection(self):
        message = fetch._explain(
            RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        )
        assert "yt-dlp" in message
        assert "network" not in message.lower()

    def test_a_genuine_network_failure_still_says_so(self):
        message = fetch._explain(RuntimeError("unable to download: <urlopen error timed out>"))
        assert "failed" in message.lower()

    def test_rate_limiting_is_not_confused_with_a_stale_version(self):
        message = fetch._explain(RuntimeError("HTTP Error 429: Too Many Requests"))
        assert "wait" in message.lower()
        assert "yt-dlp" not in message

    def test_missing_ffmpeg_is_still_recognised(self):
        assert "ffmpeg" in fetch._explain(RuntimeError("ffprobe/ffmpeg not found")).lower()


class FakeYdl:
    """Stands in for yt_dlp.YoutubeDL, recording which clients were asked."""

    attempts = []

    def __init__(self, options):
        self.options = options
        args = (options.get("extractor_args") or {}).get("youtube", {})
        self.client = (args.get("player_client") or [None])[0]
        FakeYdl.attempts.append(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def module_with(behaviour):
    """A fake yt_dlp module whose extract_info follows `behaviour(client)`."""
    FakeYdl.attempts = []

    class Ydl(FakeYdl):
        def extract_info(self, url, download=True):
            return behaviour(self.client)

    return types.SimpleNamespace(YoutubeDL=Ydl)


class TestPlayerClientFallback:
    """YouTube gates different stream sets behind different clients, so a refusal
    is worth retrying as someone else before giving up."""

    def test_the_default_client_is_tried_first_and_alone(self):
        ydl = module_with(lambda client: {"title": "ok", "client": client})
        info, error = fetch._download_with_fallbacks(ydl, "https://y/1", {})

        assert error is None and info["title"] == "ok"
        assert FakeYdl.attempts == [None], "a working link must not pay for retries"

    def test_a_403_falls_through_to_another_client(self):
        def behaviour(client):
            if client is None:
                raise RuntimeError("HTTP Error 403: Forbidden")
            return {"title": "ok", "client": client}

        info, error = fetch._download_with_fallbacks(module_with(behaviour), "https://y/1", {})
        assert error is None
        assert info["client"] == fetch.PLAYER_CLIENTS[1]
        assert len(FakeYdl.attempts) == 2

    def test_a_private_video_is_not_retried(self):
        """It fails identically for every client, so retrying only lengthens the
        wait before the same answer."""
        def behaviour(client):
            raise RuntimeError("ERROR: Private video. Sign in if you've been granted access")

        info, error = fetch._download_with_fallbacks(module_with(behaviour), "https://y/1", {})
        assert info is None and error is not None
        assert FakeYdl.attempts == [None]

    def test_every_client_refusing_reports_the_last_failure(self):
        def behaviour(client):
            raise RuntimeError("HTTP Error 403: Forbidden")

        info, error = fetch._download_with_fallbacks(module_with(behaviour), "https://y/1", {})
        assert info is None
        assert "403" in str(error)
        assert len(FakeYdl.attempts) == len(fetch.PLAYER_CLIENTS)

    def test_the_403_message_does_not_blame_the_connection(self):
        message = fetch._explain(RuntimeError("HTTP Error 403: Forbidden"))
        assert "not your connection" in message
        assert "--pre" in message
