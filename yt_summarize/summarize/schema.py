"""Pydantic schemas for structured summary output."""

from pydantic import BaseModel


class ChapterSchema(BaseModel):
    """Schema for a single chapter."""

    start: str
    heading: str
    bullets: list[str]


class SummarySchema(BaseModel):
    """Schema for structured summary JSON output."""

    title: str
    source_url: str
    tldr: list[str]
    key_points: list[str]
    chapters: list[ChapterSchema]
    quotes: list[str]
    action_items: list[str]
    tags: list[str]
