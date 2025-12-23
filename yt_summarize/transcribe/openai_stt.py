"""OpenAI speech-to-text transcription."""

import os
import time
from pathlib import Path

from openai import OpenAI, OpenAIError

# Maximum file size for OpenAI API (25MB)
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def _get_client() -> OpenAI:
    """Get OpenAI client, checking for API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise TranscriptionError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def _transcribe_with_retry(
    client: OpenAI,
    audio_path: Path,
    model: str,
    lang: str | None,
    max_retries: int = 3,
) -> str:
    """Transcribe with exponential backoff retry."""
    last_error = None

    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                # Build kwargs
                kwargs: dict = {"model": model, "file": f}
                if lang and lang != "auto":
                    kwargs["language"] = lang

                response = client.audio.transcriptions.create(**kwargs)
                return response.text

        except OpenAIError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                time.sleep(wait_time)

    raise TranscriptionError(f"Transcription failed after {max_retries} attempts: {last_error}")


def transcribe_audio(
    audio_path: Path,
    model: str = "whisper-1",
    lang: str | None = None,
) -> str:
    """
    Transcribe audio file using OpenAI speech-to-text.

    Args:
        audio_path: Path to audio file (mp3, m4a, wav, etc.)
        model: OpenAI STT model (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)
        lang: Optional language hint (ISO-639-1 code like 'en', 'sv')

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: Audio file doesn't exist
        TranscriptionError: Transcription failed
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check file format
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise TranscriptionError(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check file size
    file_size = audio_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise TranscriptionError(
            f"Audio file too large: {file_size / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_MB}MB limit. "
            "Consider splitting the audio or using a shorter video."
        )

    client = _get_client()
    return _transcribe_with_retry(client, audio_path, model, lang)


def transcribe_audio_chunked(
    audio_path: Path,
    model: str = "whisper-1",
    lang: str | None = None,
    chunk_duration_minutes: int = 10,
) -> str:
    """
    Transcribe large audio file by splitting into chunks.

    Uses ffmpeg to split audio, transcribes each chunk, then concatenates.

    Args:
        audio_path: Path to audio file
        model: OpenAI STT model
        lang: Optional language hint
        chunk_duration_minutes: Duration of each chunk in minutes

    Returns:
        Full transcribed text

    Raises:
        FileNotFoundError: Audio file doesn't exist
        TranscriptionError: Transcription failed
    """
    import subprocess
    import tempfile

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size

    # If file is small enough, use regular transcription
    if file_size <= MAX_FILE_SIZE_BYTES:
        return transcribe_audio(audio_path, model, lang)

    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError as e:
        raise TranscriptionError(
            "ffmpeg required for large file transcription. Install with: brew install ffmpeg"
        ) from e

    client = _get_client()
    transcripts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        chunk_pattern = tmpdir_path / "chunk_%03d.mp3"

        # Split audio into chunks
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(chunk_duration_minutes * 60),
            "-c", "copy",
            "-y",
            str(chunk_pattern),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise TranscriptionError(f"Failed to split audio: {result.stderr}")

        # Transcribe each chunk
        chunk_files = sorted(tmpdir_path.glob("chunk_*.mp3"))

        for i, chunk_file in enumerate(chunk_files):
            try:
                text = _transcribe_with_retry(client, chunk_file, model, lang)
                transcripts.append(text)
            except TranscriptionError as e:
                raise TranscriptionError(f"Failed on chunk {i + 1}/{len(chunk_files)}: {e}") from e

    return " ".join(transcripts)
