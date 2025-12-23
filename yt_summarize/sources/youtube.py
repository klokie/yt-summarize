"""YouTube transcript fetching utilities."""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class AudioDownloadResult:
    """Result of audio download."""

    audio_path: Path
    video_id: str
    title: str
    channel: str
    duration_seconds: float


class TranscriptNotAvailable(Exception):
    """Raised when no transcript can be obtained."""


class YtDlpNotFound(Exception):
    """Raised when yt-dlp is not installed."""


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


def _check_ytdlp() -> None:
    """Check if yt-dlp is available."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True, timeout=10)
    except FileNotFoundError as e:
        raise YtDlpNotFound(
            "yt-dlp not found. Install with: brew install yt-dlp (or pip install yt-dlp)"
        ) from e
    except subprocess.CalledProcessError as e:
        raise YtDlpNotFound(f"yt-dlp check failed: {e}") from e


def _fetch_video_metadata(video_id: str) -> tuple[str, str, float]:
    """
    Fetch video title, channel name, and duration.

    Uses yt-dlp for metadata extraction (no download).
    Falls back to defaults if unavailable.
    """
    try:
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
            return (
                data.get("title", video_id),
                data.get("channel", "Unknown"),
                data.get("duration", 0.0),
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return video_id, "Unknown", 0.0


def _snippets_to_text(snippets: list) -> str:
    """Convert transcript snippets to plain text."""
    return " ".join(s.text.strip() for s in snippets if s.text)


def _parse_vtt(vtt_content: str) -> str:
    """Parse VTT subtitle file to plain text."""
    lines = []
    in_cue = False

    for line in vtt_content.split("\n"):
        line = line.strip()

        # Skip header and empty lines
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            in_cue = False
            continue

        # Skip timestamp lines
        if "-->" in line:
            in_cue = True
            continue

        # Skip cue identifiers (numeric or named)
        if in_cue is False and (line.isdigit() or re.match(r"^[\w-]+$", line)):
            continue

        if in_cue:
            # Remove VTT tags like <c> </c> <00:00:00.000>
            clean = re.sub(r"<[^>]+>", "", line)
            # Remove speaker labels like [Music] or (applause)
            clean = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", clean)
            clean = clean.strip()
            if clean:
                lines.append(clean)

    # Dedupe consecutive identical lines (common in auto-subs)
    deduped = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return " ".join(deduped)


def fetch_subtitles_ytdlp(
    video_id: str,
    lang: str = "auto",
    output_dir: Path | None = None,
) -> tuple[str, str]:
    """
    Fetch subtitles using yt-dlp.

    Args:
        video_id: YouTube video ID
        lang: Language code or "auto"
        output_dir: Directory to save subtitle files (temp if None)

    Returns:
        Tuple of (transcript_text, language_code)

    Raises:
        TranscriptNotAvailable: If no subtitles found
        YtDlpNotFound: If yt-dlp not installed
    """
    _check_ytdlp()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(output_dir) if output_dir else Path(tmpdir)
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Build language args
        lang_args = []
        if lang != "auto":
            lang_args = ["--sub-langs", f"{lang},en", "--write-subs"]
        else:
            lang_args = ["--write-auto-subs", "--write-subs", "--sub-langs", "en,en-US,en-GB"]

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--no-playlist",
            *lang_args,
            "--sub-format", "vtt",
            "--convert-subs", "vtt",
            "-o", str(work_dir / "%(id)s.%(ext)s"),
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise TranscriptNotAvailable(f"yt-dlp subtitle fetch failed: {result.stderr}")

        # Find downloaded subtitle file
        vtt_files = list(work_dir.glob(f"{video_id}*.vtt"))
        if not vtt_files:
            raise TranscriptNotAvailable(f"No subtitles found for video {video_id}")

        # Prefer non-auto-generated if available
        chosen = vtt_files[0]
        for f in vtt_files:
            if ".en." in f.name and "auto" not in f.name.lower():
                chosen = f
                break

        # Extract language from filename (e.g., "VIDEO_ID.en.vtt")
        found_lang = "en"
        match = re.search(r"\.([a-z]{2}(?:-[A-Z]{2})?)\.vtt$", chosen.name)
        if match:
            found_lang = match.group(1)

        vtt_content = chosen.read_text(encoding="utf-8")
        text = _parse_vtt(vtt_content)

        if not text.strip():
            raise TranscriptNotAvailable("Subtitle file was empty")

        return text, found_lang


def download_audio(
    video_id: str,
    output_dir: Path,
    max_minutes: int = 180,
) -> AudioDownloadResult:
    """
    Download audio from YouTube video using yt-dlp.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save audio file
        max_minutes: Maximum video length in minutes

    Returns:
        AudioDownloadResult with path and metadata

    Raises:
        TranscriptNotAvailable: If download fails or video too long
        YtDlpNotFound: If yt-dlp not installed
    """
    _check_ytdlp()

    url = f"https://www.youtube.com/watch?v={video_id}"
    title, channel, duration = _fetch_video_metadata(video_id)

    # Check duration
    if duration > max_minutes * 60:
        raise TranscriptNotAvailable(
            f"Video too long ({duration / 60:.0f} min > {max_minutes} min limit)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",  # extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # best quality
        "--no-playlist",
        "-o", output_template,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise TranscriptNotAvailable(f"Audio download failed: {result.stderr}")

    # Find the downloaded file
    audio_files = list(output_dir.glob(f"{video_id}.*"))
    audio_files = [f for f in audio_files if f.suffix in (".mp3", ".m4a", ".webm", ".opus")]

    if not audio_files:
        raise TranscriptNotAvailable("Audio file not found after download")

    return AudioDownloadResult(
        audio_path=audio_files[0],
        video_id=video_id,
        title=title,
        channel=channel,
        duration_seconds=duration,
    )


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
    audio_output_dir: Path | None = None,
    max_minutes: int = 180,
) -> TranscriptResult | AudioDownloadResult:
    """
    Fetch transcript from YouTube video.

    Priority:
    1. youtube-transcript-api (captions/auto-captions)
    2. yt-dlp subtitles
    3. Audio download for STT (if allowed)

    Args:
        url: YouTube URL or video ID
        lang: Language code or "auto"
        allow_audio_fallback: Whether to download audio for STT if no captions
        audio_output_dir: Directory for audio files (required if audio fallback used)
        max_minutes: Maximum video length for audio download

    Returns:
        TranscriptResult with text and metadata, or AudioDownloadResult for STT

    Raises:
        ValueError: Invalid URL
        TranscriptNotAvailable: No transcript available
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    # Try youtube-transcript-api first (fastest)
    try:
        text, found_lang = fetch_transcript_api(video_id, lang)
        title, channel, _ = _fetch_video_metadata(video_id)

        return TranscriptResult(
            text=text,
            video_id=video_id,
            title=title,
            channel=channel,
            lang=found_lang,
            method="captions",
        )
    except TranscriptNotAvailable:
        pass  # Try next method

    # Try yt-dlp subtitles
    try:
        text, found_lang = fetch_subtitles_ytdlp(video_id, lang)
        title, channel, _ = _fetch_video_metadata(video_id)

        return TranscriptResult(
            text=text,
            video_id=video_id,
            title=title,
            channel=channel,
            lang=found_lang,
            method="subs",
        )
    except (TranscriptNotAvailable, YtDlpNotFound):
        pass  # Try audio fallback

    # Audio fallback for STT
    if not allow_audio_fallback:
        raise TranscriptNotAvailable(
            f"No captions/subtitles available for {video_id} and audio fallback disabled"
        )

    if audio_output_dir is None:
        audio_output_dir = Path(tempfile.gettempdir()) / "yt-summarize" / video_id

    return download_audio(video_id, audio_output_dir, max_minutes)
