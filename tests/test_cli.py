"""Tests for CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from yt_summarize.cli import app

runner = CliRunner()


class TestSummarizeCommand:
    """Tests for summarize command."""

    def test_help(self) -> None:
        """Test help output."""
        result = runner.invoke(app, ["summarize", "--help"])
        assert result.exit_code == 0
        assert "YouTube URL or local .txt file" in result.output

    def test_missing_source(self) -> None:
        """Test error when source is missing."""
        result = runner.invoke(app, ["summarize"])
        assert result.exit_code != 0

    def test_local_file_not_found(self) -> None:
        """Test error when local file doesn't exist."""
        result = runner.invoke(app, ["summarize", "/nonexistent/file.txt"])
        assert result.exit_code == 1
        assert "File not found" in result.output

    @patch("yt_summarize.cli.fetch_youtube_transcript")
    @patch("yt_summarize.cli.summarize_transcript")
    def test_local_only_mode(
        self,
        mock_summarize: MagicMock,
        _mock_fetch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --local-only mode skips summarization."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for summarization.")

        out_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "summarize",
                str(test_file),
                "--local-only",
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Done!" in result.output

        # Summarize should NOT be called
        mock_summarize.assert_not_called()

        # Check outputs
        assert (out_dir / "transcript.txt").exists()
        assert (out_dir / "meta.json").exists()
        assert not (out_dir / "summary.md").exists()

    @patch("yt_summarize.cli.load_transcript")
    @patch("yt_summarize.cli.load_summary")
    def test_force_bypasses_cache(
        self,
        _mock_load_summary: MagicMock,
        mock_load_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --force bypasses cache."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        # Run with --force
        result = runner.invoke(
            app,
            [
                "summarize",
                str(test_file),
                "--force",
                "--local-only",
                "--out",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0
        # Cache should NOT be checked when --force is used
        mock_load_transcript.assert_not_called()


class TestCacheCommands:
    """Tests for cache commands."""

    def test_cache_list_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test cache list when empty."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = runner.invoke(app, ["cache", "list"])
        assert result.exit_code == 0
        assert "empty" in result.output

    def test_cache_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test cache stats."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = runner.invoke(app, ["cache", "stats"])
        assert result.exit_code == 0
        assert "Transcripts:" in result.output
        assert "Summaries:" in result.output

    def test_cache_clear_requires_flag(self) -> None:
        """Test cache clear requires --key or --all."""
        result = runner.invoke(app, ["cache", "clear"])
        assert result.exit_code == 1
        assert "--key or --all" in result.output

    def test_cache_clear_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test cache clear --all."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create cache dir
        cache_dir = tmp_path / ".cache" / "yt-summarize"
        cache_dir.mkdir(parents=True)
        (cache_dir / "test_transcript.json").write_text("{}")

        result = runner.invoke(app, ["cache", "clear", "--all"])
        assert result.exit_code == 0
        assert "Cleared" in result.output


class TestUrlParsing:
    """Tests for URL detection in CLI."""

    @patch("yt_summarize.cli.fetch_youtube_transcript")
    @patch("yt_summarize.cli.load_transcript")
    def test_detects_youtube_url(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that YouTube URLs are detected."""
        from yt_summarize.sources.youtube import TranscriptResult

        mock_load.return_value = None
        mock_fetch.return_value = TranscriptResult(
            text="Test transcript",
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            channel="Test Channel",
            lang="en",
            method="captions",
        )

        result = runner.invoke(
            app,
            [
                "summarize",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "--local-only",
                "--out",
                str(tmp_path / "out"),
            ],
        )

        # Should attempt to fetch from YouTube
        mock_fetch.assert_called_once()
        assert result.exit_code == 0

    def test_detects_local_file(self, tmp_path: Path) -> None:
        """Test that local files are detected."""
        test_file = tmp_path / "transcript.txt"
        test_file.write_text("Local content")

        result = runner.invoke(
            app,
            [
                "summarize",
                str(test_file),
                "--local-only",
                "--out",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0
        assert (tmp_path / "out" / "transcript.txt").exists()
