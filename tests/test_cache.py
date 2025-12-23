"""Tests for caching utilities."""

from pathlib import Path

import pytest

from yt_summarize.cache import (
    clear_cache,
    create_summary_cache,
    create_transcript_cache,
    get_cache_key_file,
    get_cache_key_youtube,
    get_cache_stats,
    list_cached,
    load_summary,
    load_transcript,
)


class TestCacheKeys:
    """Tests for cache key generation."""

    def test_youtube_cache_key(self) -> None:
        key = get_cache_key_youtube("dQw4w9WgXcQ", "en", "captions")
        assert key == "dQw4w9WgXcQ_en_captions"

    def test_youtube_cache_key_auto_lang(self) -> None:
        key = get_cache_key_youtube("abc123xyz99", "auto", "stt")
        assert key == "abc123xyz99_auto_stt"

    def test_file_cache_key(self, tmp_path: Path) -> None:
        """Test file-based cache key generation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        key = get_cache_key_file(test_file)
        assert len(key) == 16  # SHA256 truncated to 16 chars
        assert key.isalnum()

    def test_file_cache_key_changes_with_content(self, tmp_path: Path) -> None:
        """Test that file cache key changes when content changes."""
        test_file = tmp_path / "test.txt"

        test_file.write_text("Content A")
        key_a = get_cache_key_file(test_file)

        test_file.write_text("Content B")
        key_b = get_cache_key_file(test_file)

        assert key_a != key_b


class TestTranscriptCache:
    """Tests for transcript caching."""

    def test_create_and_load_transcript(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating and loading cached transcript."""
        # Use temp dir as cache
        monkeypatch.setenv("HOME", str(tmp_path))

        cache_key, transcript = create_transcript_cache(
            video_id="test123",
            text="This is a test transcript.",
            title="Test Video",
            channel="Test Channel",
            lang="en",
            method="captions",
        )

        assert cache_key == "test123_en_captions"

        # Load it back
        loaded = load_transcript(cache_key)
        assert loaded is not None
        assert loaded.text == "This is a test transcript."
        assert loaded.title == "Test Video"
        assert loaded.method == "captions"

    def test_load_missing_transcript(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading non-existent transcript returns None."""
        monkeypatch.setenv("HOME", str(tmp_path))

        loaded = load_transcript("nonexistent_key")
        assert loaded is None


class TestSummaryCache:
    """Tests for summary caching."""

    def test_create_and_load_summary_md(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test caching markdown summary."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache_key = "test_key"
        summary = create_summary_cache(
            cache_key=cache_key,
            markdown="# Summary\n\nContent here.",
            json_data=None,
            model="gpt-4o-mini",
        )

        assert summary.markdown == "# Summary\n\nContent here."

        # Load it back requesting md format
        loaded = load_summary(cache_key, "md")
        assert loaded is not None
        assert loaded.markdown == "# Summary\n\nContent here."
        assert loaded.json_data is None

    def test_create_and_load_summary_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test caching JSON summary."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache_key = "test_key_json"
        json_data = {"title": "Test", "tldr": ["Point 1"]}
        create_summary_cache(
            cache_key=cache_key,
            markdown=None,
            json_data=json_data,
            model="gpt-4o-mini",
        )

        loaded = load_summary(cache_key, "json")
        assert loaded is not None
        assert loaded.json_data == json_data

    def test_load_summary_missing_format(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that loading returns None if requested format isn't cached."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache_key = "test_missing_format"
        create_summary_cache(
            cache_key=cache_key,
            markdown="# Markdown only",
            json_data=None,
            model="gpt-4o-mini",
        )

        # Request JSON format, but only MD is cached
        loaded = load_summary(cache_key, "json")
        assert loaded is None


class TestClearCache:
    """Tests for cache clearing."""

    def test_clear_specific_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clearing cache for specific key."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create two entries
        create_transcript_cache("vid1", "Text 1", "Title 1", "Ch1", "en", "captions")
        create_transcript_cache("vid2", "Text 2", "Title 2", "Ch2", "en", "captions")

        # Clear only first
        count = clear_cache("vid1_en_captions")
        assert count == 1

        # First should be gone, second should remain
        assert load_transcript("vid1_en_captions") is None
        assert load_transcript("vid2_en_captions") is not None

    def test_clear_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clearing all cache."""
        monkeypatch.setenv("HOME", str(tmp_path))

        create_transcript_cache("vid1", "Text 1", "Title 1", "Ch1", "en", "captions")
        create_transcript_cache("vid2", "Text 2", "Title 2", "Ch2", "en", "subs")

        count = clear_cache()
        assert count == 2

        assert load_transcript("vid1_en_captions") is None
        assert load_transcript("vid2_en_subs") is None


class TestCacheStats:
    """Tests for cache statistics."""

    def test_get_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting cache statistics."""
        monkeypatch.setenv("HOME", str(tmp_path))

        create_transcript_cache("vid1", "Text", "Title", "Ch", "en", "captions")
        create_summary_cache("vid1_en_captions", "# MD", None, "gpt-4o-mini")

        stats = get_cache_stats()
        assert stats["transcript_count"] == 1
        assert stats["summary_count"] == 1
        assert stats["total_size_kb"] > 0


class TestListCached:
    """Tests for listing cached entries."""

    def test_list_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test listing cached entries."""
        monkeypatch.setenv("HOME", str(tmp_path))

        create_transcript_cache("vid1", "Text 1", "Video One", "Channel", "en", "captions")
        create_transcript_cache("vid2", "Text 2", "Video Two", "Channel", "sv", "subs")
        create_summary_cache("vid1_en_captions", "# Summary", None, "gpt-4o-mini")

        entries = list_cached()
        assert len(entries) == 2

        # Find vid1 entry
        vid1_entry = next(e for e in entries if e["video_id"] == "vid1")
        assert vid1_entry["title"] == "Video One"
        assert vid1_entry["has_summary"] is True

        # Find vid2 entry
        vid2_entry = next(e for e in entries if e["video_id"] == "vid2")
        assert vid2_entry["title"] == "Video Two"
        assert vid2_entry["has_summary"] is False
