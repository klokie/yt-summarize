"""Local file transcript loading."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalTranscriptResult:
    """Result of local file load."""

    text: str
    file_path: Path
    title: str


def load_local_transcript(file_path: Path, title: str | None = None) -> LocalTranscriptResult:
    """Load transcript from local text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix == ".txt":
        raise ValueError(f"Expected .txt file, got: {file_path.suffix}")

    text = file_path.read_text(encoding="utf-8")
    resolved_title = title or file_path.stem

    return LocalTranscriptResult(
        text=text,
        file_path=file_path,
        title=resolved_title,
    )
