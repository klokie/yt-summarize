"""OpenAI speech-to-text transcription."""

from pathlib import Path


def transcribe_audio(
    audio_path: Path,
    model: str = "gpt-4o-mini-transcribe",
    lang: str | None = None,
) -> str:
    """
    Transcribe audio file using OpenAI speech-to-text.

    Args:
        audio_path: Path to audio file (mp3, m4a, etc.)
        model: OpenAI STT model to use
        lang: Optional language hint

    Returns:
        Transcribed text
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # TODO: Implement OpenAI STT call
    # from openai import OpenAI
    # client = OpenAI()
    # with open(audio_path, "rb") as f:
    #     result = client.audio.transcriptions.create(model=model, file=f, language=lang)
    # return result.text

    raise NotImplementedError("OpenAI STT transcription not yet implemented")
