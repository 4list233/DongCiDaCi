"""Tests for on-disk song storage.

Covers the behaviour that decides whether the library stays usable over time:
repeated attempts at the same recording should not accumulate as separate,
indistinguishable songs.
"""

import pytest

from dcdc.store import Store


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "songs")


class TestReuseByUrl:
    """Pasting the same link twice means 'do that again', not 'make a copy'.

    Without this, every retry left another untitled-N behind -- and because the
    slug comes from the title before the downloader has resolved it, those copies
    are not even distinguishable by name.
    """

    def test_finds_an_existing_song_for_a_link(self, store):
        song = store.create("Untitled", "", "https://example.com/watch?v=abc")
        assert store.find_by_url("https://example.com/watch?v=abc").slug == song.slug

    def test_ignores_surrounding_whitespace(self, store):
        song = store.create("Untitled", "", "https://example.com/a")
        assert store.find_by_url("  https://example.com/a  ").slug == song.slug

    def test_a_different_link_is_not_a_match(self, store):
        store.create("Untitled", "", "https://example.com/a")
        assert store.find_by_url("https://example.com/b") is None

    def test_songs_without_a_link_never_match(self, store):
        store.create("Uploaded file", "", "")
        assert store.find_by_url("") is None
        assert store.find_by_url("https://example.com/a") is None
