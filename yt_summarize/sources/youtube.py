"""YouTube transcript fetching utilities."""

import re
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


@dataclass
class TranscriptResult:
    """Result of transcript fetch."""

    text: str
    video_id: str
    title: str
    channel: str
    lang: str
    method: str  # "captions" | "subs" | "stt"


class TranscriptNotAvailable(Exception):
    """Raised when no transcript can be obtained."""


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


def _fetch_video_metadata(video_id: str) -> tuple[str, str]:
    """
    Fetch video title and channel name.

    Uses yt-dlp for metadata extraction (no download).
    Falls back to video_id if unavailable.
    """
    try:
        import json
        import subprocess

        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--no-playlist",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("title", video_id), data.get("channel", "Unknown")
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return video_id, "Unknown"


def _snippets_to_text(snippets: list) -> str:
    """Convert transcript snippets to plain text."""
    return " ".join(s.text.strip() for s in snippets if s.text)


def fetch_transcript_api(
    video_id: str,
    lang: str = "auto",
) -> tuple[str, str]:
    """
    Fetch transcript using youtube-transcript-api v1.x.

    Returns:
        Tuple of (transcript_text, language_code)

    Raises:
        TranscriptNotAvailable: If no transcript found
    """
    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled as e:
        raise TranscriptNotAvailable(f"Transcripts disabled for video {video_id}") from e
    except Exception as e:
        raise TranscriptNotAvailable(f"Failed to list transcripts: {e}") from e

    transcript = None
    found_lang = None

    if lang != "auto":
        # Try exact language match first
        try:
            for t in transcript_list:
                if t.language_code == lang:
                    transcript = t
                    found_lang = lang
                    break
        except NoTranscriptFound:
            pass

        # Try translated version if not found
        if transcript is None:
            try:
                for t in transcript_list:
                    if t.is_translatable and lang in [
                        tl["language_code"] for tl in t.translation_languages
                    ]:
                        transcript = t.translate(lang)
                        found_lang = lang
                        break
            except Exception:
                pass

    if transcript is None:
        # Auto mode: prefer manual transcripts, then auto-generated
        transcripts = list(transcript_list)

        # Try manual transcripts first (higher quality)
        for t in transcripts:
            if not t.is_generated:
                transcript = t
                found_lang = t.language_code
                break

        # Fall back to auto-generated
        if transcript is None:
            for t in transcripts:
                if t.is_generated:
                    transcript = t
                    found_lang = t.language_code
                    break

    if transcript is None:
        raise TranscriptNotAvailable(f"No suitable transcript found for video {video_id}")

    fetched = transcript.fetch()
    text = _snippets_to_text(fetched.snippets)

    return text, found_lang or "unknown"


def fetch_youtube_transcript(
    url: str,
    lang: str = "auto",
    allow_audio_fallback: bool = True,
) -> TranscriptResult:
    """
    Fetch transcript from YouTube video.

    Priority:
    1. youtube-transcript-api (captions/auto-captions)
    2. Audio fallback via STT (if allowed) - not implemented yet

    Args:
        url: YouTube URL or video ID
        lang: Language code or "auto"
        allow_audio_fallback: Whether to transcribe audio if no captions

    Returns:
        TranscriptResult with text and metadata

    Raises:
        ValueError: Invalid URL
        TranscriptNotAvailable: No transcript available
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    # Try youtube-transcript-api first
    try:
        text, found_lang = fetch_transcript_api(video_id, lang)
        title, channel = _fetch_video_metadata(video_id)

        return TranscriptResult(
            text=text,
            video_id=video_id,
            title=title,
            channel=channel,
            lang=found_lang,
            method="captions",
        )
    except TranscriptNotAvailable:
        if not allow_audio_fallback:
            raise

        # TODO: Audio fallback will be implemented in milestone 3/4
        raise TranscriptNotAvailable(
            f"No captions available for {video_id}. Audio fallback not yet implemented."
        ) from None
