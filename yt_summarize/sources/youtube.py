"""YouTube transcript fetching utilities."""

import re
from dataclasses import dataclass


@dataclass
class TranscriptResult:
    """Result of transcript fetch."""

    text: str
    video_id: str
    title: str
    channel: str
    lang: str
    method: str  # "captions" | "subs" | "stt"


def extract_video_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # bare video ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_transcript(
    url: str,
    lang: str = "auto",
    allow_audio_fallback: bool = True,
) -> TranscriptResult:
    """
    Fetch transcript from YouTube video.

    Priority:
    1. youtube-transcript-api (auto-captions)
    2. yt-dlp subtitles
    3. Audio fallback (if allowed)
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    # TODO: Implement fetch logic
    # 1. Try youtube-transcript-api
    # 2. Fallback to yt-dlp subs
    # 3. Fallback to audio transcription

    raise NotImplementedError("YouTube transcript fetching not yet implemented")
