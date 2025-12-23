"""Tests for source modules."""

from yt_summarize.sources.youtube import extract_video_id


class TestExtractVideoId:
    """Tests for YouTube video ID extraction."""

    def test_standard_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_params(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_bare_id(self) -> None:
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url(self) -> None:
        assert extract_video_id("https://example.com") is None

    def test_invalid_id_length(self) -> None:
        assert extract_video_id("tooshort") is None
