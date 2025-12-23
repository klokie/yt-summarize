"""Transcription modules for audio-to-text."""

from .openai_stt import TranscriptionError, transcribe_audio, transcribe_audio_chunked

__all__ = ["TranscriptionError", "transcribe_audio", "transcribe_audio_chunked"]
