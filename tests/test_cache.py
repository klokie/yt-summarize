"""Tests for caching utilities."""

from yt_summarize.cache import get_cache_key_youtube


class TestCacheKeys:
    """Tests for cache key generation."""

    def test_youtube_cache_key(self) -> None:
        key = get_cache_key_youtube("dQw4w9WgXcQ", "en", "captions")
        assert key == "dQw4w9WgXcQ_en_captions"

    def test_youtube_cache_key_auto_lang(self) -> None:
        key = get_cache_key_youtube("abc123xyz99", "auto", "stt")
        assert key == "abc123xyz99_auto_stt"
