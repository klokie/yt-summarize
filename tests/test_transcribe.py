"""Tests for transcription module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yt_summarize.transcribe.openai_stt import (
    SUPPORTED_FORMATS,
    TranscriptionError,
    transcribe_audio,
)


class TestTranscribeAudio:
    """Tests for audio transcription."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        fake_path = tmp_path / "nonexistent.mp3"
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe_audio(fake_path)

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Test error for unsupported audio format."""
        bad_file = tmp_path / "audio.txt"
        bad_file.write_text("not audio")

        with pytest.raises(TranscriptionError, match="Unsupported audio format"):
            transcribe_audio(bad_file)

    def test_file_too_large(self, tmp_path: Path) -> None:
        """Test error when file exceeds size limit."""
        large_file = tmp_path / "large.mp3"
        # Create a file header that looks like mp3 but is too large
        large_file.write_bytes(b"fake" * (26 * 1024 * 1024 // 4))  # ~26MB

        with pytest.raises(TranscriptionError, match="Audio file too large"):
            transcribe_audio(large_file)

    def test_missing_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when API key not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\xff\xfb\x90\x00" * 100)  # Minimal mp3 header

        with pytest.raises(TranscriptionError, match="OPENAI_API_KEY"):
            transcribe_audio(audio_file)

    @patch("yt_summarize.transcribe.openai_stt._get_client")
    def test_successful_transcription(self, mock_get_client: MagicMock, tmp_path: Path) -> None:
        """Test successful transcription."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Hello world, this is a test."
        mock_client.audio.transcriptions.create.return_value = mock_response

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\xff\xfb\x90\x00" * 100)

        result = transcribe_audio(audio_file, model="whisper-1", lang="en")

        assert result == "Hello world, this is a test."
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch("yt_summarize.transcribe.openai_stt._get_client")
    @patch("yt_summarize.transcribe.openai_stt.time.sleep")
    def test_retry_on_failure(
        self,
        mock_sleep: MagicMock,
        mock_get_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test retry logic on transient failures."""
        from openai import OpenAIError

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Success after retry"

        # Fail twice, succeed on third attempt
        mock_client.audio.transcriptions.create.side_effect = [
            OpenAIError("Rate limit"),
            OpenAIError("Rate limit"),
            mock_response,
        ]

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\xff\xfb\x90\x00" * 100)

        result = transcribe_audio(audio_file)

        assert result == "Success after retry"
        assert mock_client.audio.transcriptions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("yt_summarize.transcribe.openai_stt._get_client")
    @patch("yt_summarize.transcribe.openai_stt.time.sleep")
    def test_max_retries_exceeded(
        self,
        _mock_sleep: MagicMock,
        mock_get_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test error after max retries exceeded."""
        from openai import OpenAIError

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = OpenAIError("Always fails")

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\xff\xfb\x90\x00" * 100)

        with pytest.raises(TranscriptionError, match="failed after 3 attempts"):
            transcribe_audio(audio_file)


class TestSupportedFormats:
    """Tests for supported format constants."""

    def test_common_formats_supported(self) -> None:
        """Test that common audio formats are supported."""
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".m4a" in SUPPORTED_FORMATS
        assert ".wav" in SUPPORTED_FORMATS
        assert ".webm" in SUPPORTED_FORMATS
