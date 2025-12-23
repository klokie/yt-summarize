"""Tests for source modules."""

from unittest.mock import MagicMock, patch

import pytest

from yt_summarize.sources.youtube import (
    TranscriptNotAvailable,
    _snippets_to_text,
    extract_video_id,
    fetch_transcript_api,
)


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


class TestSnippetsToText:
    """Tests for snippet conversion."""

    def test_basic_snippets(self) -> None:
        snippets = [
            MagicMock(text="Hello world"),
            MagicMock(text="How are you"),
        ]
        result = _snippets_to_text(snippets)
        assert result == "Hello world How are you"

    def test_strips_whitespace(self) -> None:
        snippets = [
            MagicMock(text="  Hello  "),
            MagicMock(text="\nWorld\n"),
        ]
        result = _snippets_to_text(snippets)
        assert result == "Hello World"

    def test_empty_snippets(self) -> None:
        snippets = [
            MagicMock(text=""),
            MagicMock(text="Content"),
            MagicMock(text=None),
        ]
        result = _snippets_to_text(snippets)
        assert result == "Content"


class TestFetchTranscriptApi:
    """Tests for youtube-transcript-api integration."""

    @patch("yt_summarize.sources.youtube.YouTubeTranscriptApi")
    def test_fetch_auto_generated(self, mock_api_class: MagicMock) -> None:
        """Test fetching auto-generated transcript."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        mock_transcript = MagicMock()
        mock_transcript.is_generated = True
        mock_transcript.language_code = "en"

        mock_fetched = MagicMock()
        mock_fetched.snippets = [
            MagicMock(text="Hello"),
            MagicMock(text="World"),
        ]
        mock_transcript.fetch.return_value = mock_fetched

        mock_api.list.return_value = [mock_transcript]

        text, lang = fetch_transcript_api("dQw4w9WgXcQ", "auto")

        assert text == "Hello World"
        assert lang == "en"

    @patch("yt_summarize.sources.youtube.YouTubeTranscriptApi")
    def test_prefers_manual_transcript(self, mock_api_class: MagicMock) -> None:
        """Test that manual transcripts are preferred over auto-generated."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        mock_manual = MagicMock()
        mock_manual.is_generated = False
        mock_manual.language_code = "en"
        mock_fetched = MagicMock()
        mock_fetched.snippets = [MagicMock(text="Manual")]
        mock_manual.fetch.return_value = mock_fetched

        mock_auto = MagicMock()
        mock_auto.is_generated = True
        mock_auto.language_code = "en"

        mock_api.list.return_value = [mock_auto, mock_manual]

        text, lang = fetch_transcript_api("video123", "auto")

        assert text == "Manual"
        mock_manual.fetch.assert_called_once()
        mock_auto.fetch.assert_not_called()

    @patch("yt_summarize.sources.youtube.YouTubeTranscriptApi")
    def test_transcripts_disabled(self, mock_api_class: MagicMock) -> None:
        """Test handling of disabled transcripts."""
        from youtube_transcript_api._errors import TranscriptsDisabled

        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list.side_effect = TranscriptsDisabled("video123")

        with pytest.raises(TranscriptNotAvailable, match="Transcripts disabled"):
            fetch_transcript_api("video123", "auto")
