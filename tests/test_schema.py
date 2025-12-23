"""Tests for summary schema."""

import pytest
from pydantic import ValidationError

from yt_summarize.summarize.schema import ChapterSchema, SummarySchema


class TestSummarySchema:
    """Tests for SummarySchema validation."""

    def test_valid_schema(self) -> None:
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
        data = {
            "title": "Test Video",
            # missing source_url and other required fields
        }
        with pytest.raises(ValidationError):
            SummarySchema(**data)

    def test_chapter_schema(self) -> None:
        chapter = ChapterSchema(
            start="5:30",
            heading="Main Content",
            bullets=["Point A", "Point B"],
        )
        assert chapter.start == "5:30"
        assert len(chapter.bullets) == 2
