"""Tests for cost estimation module."""

from yt_summarize.costs import (
    WARN_AUDIO_MINUTES,
    WARN_TRANSCRIPT_TOKENS,
    estimate_summarization_cost,
    estimate_transcription_cost,
    format_cost_warning,
)


class TestEstimateSummarizationCost:
    """Tests for summarization cost estimation."""

    def test_small_transcript(self) -> None:
        """Test cost estimation for small transcript."""
        result = estimate_summarization_cost(1000, chunk_tokens=3000, model="gpt-4o-mini")

        assert result["num_chunks"] == 1
        assert result["estimated_cost"] > 0
        assert result["should_warn"] is False

    def test_large_transcript_warns(self) -> None:
        """Test that large transcripts trigger warning."""
        result = estimate_summarization_cost(
            WARN_TRANSCRIPT_TOKENS + 10000, chunk_tokens=3000, model="gpt-4o-mini"
        )

        assert result["num_chunks"] > 10
        assert result["should_warn"] is True

    def test_expensive_model_costs_more(self) -> None:
        """Test that expensive models have higher costs."""
        cheap = estimate_summarization_cost(10000, model="gpt-4o-mini")
        expensive = estimate_summarization_cost(10000, model="gpt-4o")

        assert expensive["estimated_cost"] > cheap["estimated_cost"]


class TestEstimateTranscriptionCost:
    """Tests for transcription cost estimation."""

    def test_short_audio(self) -> None:
        """Test cost estimation for short audio."""
        result = estimate_transcription_cost(300, model="whisper-1")  # 5 min

        assert result["duration_minutes"] == 5
        assert result["estimated_cost"] > 0
        assert result["should_warn"] is False

    def test_long_audio_warns(self) -> None:
        """Test that long audio triggers warning."""
        result = estimate_transcription_cost(
            (WARN_AUDIO_MINUTES + 10) * 60, model="whisper-1"
        )

        assert result["duration_minutes"] > WARN_AUDIO_MINUTES
        assert result["should_warn"] is True


class TestFormatCostWarning:
    """Tests for cost warning formatting."""

    def test_basic_warning(self) -> None:
        """Test basic warning format."""
        msg = format_cost_warning("Test operation", 0.50)
        assert "Test operation" in msg
        assert "$0.500" in msg

    def test_warning_with_details(self) -> None:
        """Test warning with additional details."""
        msg = format_cost_warning("Test", 1.25, "100 tokens")
        assert "100 tokens" in msg
        assert "$1.250" in msg

