"""Tests for summarization module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from yt_summarize.summarize.map_reduce import (
    SummarizationError,
    SummarizeOptions,
    chunk_transcript,
    count_tokens,
    summarize_transcript,
)
from yt_summarize.summarize.schema import SummarySchema


class TestCountTokens:
    """Tests for token counting."""

    def test_counts_tokens(self) -> None:
        """Test basic token counting."""
        text = "Hello, world!"
        count = count_tokens(text)
        assert count > 0
        assert count < 10  # Should be just a few tokens

    def test_longer_text_more_tokens(self) -> None:
        """Test that longer text has more tokens."""
        short = "Hello"
        long = "Hello world, this is a much longer piece of text."
        assert count_tokens(long) > count_tokens(short)


class TestChunkTranscript:
    """Tests for transcript chunking."""

    def test_short_text_single_chunk(self) -> None:
        """Test that short text stays in one chunk."""
        text = "This is a short text."
        chunks = chunk_transcript(text, chunk_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short text.."

    def test_long_text_multiple_chunks(self) -> None:
        """Test that long text is split into multiple chunks."""
        # Create a long text
        text = ". ".join(["This is sentence number " + str(i) for i in range(100)])
        chunks = chunk_transcript(text, chunk_tokens=50)
        assert len(chunks) > 1

    def test_respects_token_limit(self) -> None:
        """Test that chunks respect approximate token limit."""
        text = ". ".join(["Word " * 20 for _ in range(50)])
        chunks = chunk_transcript(text, chunk_tokens=100)

        for chunk in chunks:
            # Allow some flexibility since we split on sentences
            tokens = count_tokens(chunk)
            assert tokens < 200  # Should be roughly within 2x the limit


class TestSummarizeOptions:
    """Tests for SummarizeOptions."""

    def test_defaults(self) -> None:
        """Test default values."""
        opts = SummarizeOptions(title="Test", source_url="https://example.com")
        assert opts.model == "gpt-4o-mini"
        assert opts.chunk_tokens == 3000
        assert opts.output_format == "md"

    def test_custom_values(self) -> None:
        """Test custom values."""
        opts = SummarizeOptions(
            title="Test",
            source_url="https://example.com",
            model="gpt-4o",
            chunk_tokens=2000,
            output_format="json",
        )
        assert opts.model == "gpt-4o"
        assert opts.chunk_tokens == 2000
        assert opts.output_format == "json"


class TestSummarizeTranscript:
    """Tests for transcript summarization."""

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when API key not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        opts = SummarizeOptions(title="Test", source_url="https://example.com")
        with pytest.raises(SummarizationError, match="OPENAI_API_KEY"):
            summarize_transcript("Some text to summarize", opts)

    @patch("yt_summarize.summarize.map_reduce._get_client")
    def test_summarize_md(self, mock_get_client: MagicMock) -> None:
        """Test markdown summarization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock map phase response
        map_response = MagicMock()
        map_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "key_points": ["Point 1", "Point 2"],
                            "quotes": ["Quote 1"],
                            "topics": ["Topic 1"],
                            "terms": {},
                        }
                    )
                )
            )
        ]

        # Mock reduce phase response
        reduce_response = MagicMock()
        reduce_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""# Test Video
[Source](https://example.com)

## TL;DR
- Point 1
- Point 2
- Point 3

## Key Points
- Key point 1
- Key point 2
"""
                )
            )
        ]

        mock_client.chat.completions.create.side_effect = [map_response, reduce_response]

        opts = SummarizeOptions(
            title="Test Video",
            source_url="https://example.com",
            output_format="md",
        )
        md_result, json_result = summarize_transcript("Short transcript text.", opts)

        assert md_result is not None
        assert "Test Video" in md_result
        assert "TL;DR" in md_result
        assert json_result is None

    @patch("yt_summarize.summarize.map_reduce._get_client")
    def test_summarize_json(self, mock_get_client: MagicMock) -> None:
        """Test JSON summarization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock map phase response
        map_response = MagicMock()
        map_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "key_points": ["Point 1"],
                            "quotes": [],
                            "topics": ["Topic"],
                            "terms": {},
                        }
                    )
                )
            )
        ]

        # Mock reduce phase response
        json_output = {
            "title": "Test Video",
            "source_url": "https://example.com",
            "tldr": ["Point 1", "Point 2", "Point 3"],
            "key_points": ["Key 1", "Key 2"],
            "chapters": [{"start": "0:00", "heading": "Intro", "bullets": ["bullet"]}],
            "quotes": ["Quote 1"],
            "action_items": ["Do this"],
            "tags": ["tag1", "tag2"],
        }
        reduce_response = MagicMock()
        reduce_response.choices = [MagicMock(message=MagicMock(content=json.dumps(json_output)))]

        mock_client.chat.completions.create.side_effect = [map_response, reduce_response]

        opts = SummarizeOptions(
            title="Test Video",
            source_url="https://example.com",
            output_format="json",
        )
        # Test with structured output disabled (legacy mode)
        md_result, json_result = summarize_transcript(
            "Short transcript.", opts, use_structured_output=False
        )

        assert md_result is None
        assert json_result is not None
        assert isinstance(json_result, SummarySchema)
        assert json_result.title == "Test Video"
        assert len(json_result.tldr) == 3

    @patch("yt_summarize.summarize.map_reduce._get_client")
    def test_summarize_json_structured(self, mock_get_client: MagicMock) -> None:
        """Test JSON summarization with structured output (guaranteed schema)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock map phase response (structured output returns clean JSON)
        map_response = MagicMock()
        map_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "key_points": ["Point 1"],
                            "quotes": ["A quote"],
                            "topics": ["Topic"],
                            "terms": {},
                        }
                    )
                )
            )
        ]

        # Mock reduce phase response (structured output)
        json_output = {
            "title": "Structured Test",
            "source_url": "https://example.com",
            "tldr": ["TL1", "TL2", "TL3"],
            "key_points": ["Key 1", "Key 2", "Key 3"],
            "chapters": [{"start": "0:00", "heading": "Intro", "bullets": ["point"]}],
            "quotes": ["Quote here"],
            "action_items": ["Action 1"],
            "tags": ["tag1"],
        }
        reduce_response = MagicMock()
        reduce_response.choices = [MagicMock(message=MagicMock(content=json.dumps(json_output)))]

        mock_client.chat.completions.create.side_effect = [map_response, reduce_response]

        opts = SummarizeOptions(
            title="Structured Test",
            source_url="https://example.com",
            output_format="json",
        )
        # Test with structured output enabled (default)
        md_result, json_result = summarize_transcript("Short text.", opts)

        assert md_result is None
        assert json_result is not None
        assert json_result.title == "Structured Test"

        # Verify structured output was requested
        call_kwargs = mock_client.chat.completions.create.call_args_list[-1].kwargs
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    @patch("yt_summarize.summarize.map_reduce._get_client")
    def test_summarize_both_formats(self, mock_get_client: MagicMock) -> None:
        """Test generating both MD and JSON output."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        map_response = MagicMock()
        map_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {"key_points": ["P1"], "quotes": [], "topics": [], "terms": {}}
                    )
                )
            )
        ]

        md_response = MagicMock()
        md_response.choices = [MagicMock(message=MagicMock(content="# Title\n\n## TL;DR\n- Point"))]

        json_output = {
            "title": "Both",
            "source_url": "https://example.com",
            "tldr": ["1", "2", "3"],
            "key_points": ["K1"],
            "chapters": [{"start": "0:00", "heading": "Ch", "bullets": ["b"]}],
            "quotes": [],
            "action_items": ["A1"],
            "tags": ["t1"],
        }
        json_response = MagicMock()
        json_response.choices = [MagicMock(message=MagicMock(content=json.dumps(json_output)))]

        mock_client.chat.completions.create.side_effect = [
            map_response,
            md_response,
            json_response,
        ]

        opts = SummarizeOptions(
            title="Both",
            source_url="https://example.com",
            output_format="md,json",
        )
        md_result, json_result = summarize_transcript("Text.", opts)

        assert md_result is not None
        assert json_result is not None
        assert "TL;DR" in md_result
        assert json_result.title == "Both"


class TestSummarySchema:
    """Tests for SummarySchema validation."""

    def test_valid_schema(self) -> None:
        """Test valid schema creation."""
        data = {
            "title": "Test Video",
            "source_url": "https://youtube.com/watch?v=test",
            "tldr": ["Point 1", "Point 2", "Point 3"],
            "key_points": ["Key 1", "Key 2"],
            "chapters": [{"start": "0:00", "heading": "Intro", "bullets": ["Welcome"]}],
            "quotes": ["Great quote"],
            "action_items": ["Do this"],
            "tags": ["python", "tutorial"],
        }
        schema = SummarySchema(**data)
        assert schema.title == "Test Video"
        assert len(schema.tldr) == 3

    def test_missing_required_field(self) -> None:
        """Test error on missing required field."""
        from pydantic import ValidationError

        data = {"title": "Test Video"}
        with pytest.raises(ValidationError):
            SummarySchema(**data)
