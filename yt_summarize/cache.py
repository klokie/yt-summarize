"""Caching utilities for transcripts and summaries."""

import hashlib
import json
from pathlib import Path
from typing import Any


def get_cache_key_youtube(video_id: str, lang: str, method: str) -> str:
    """Generate cache key for YouTube video."""
    return f"{video_id}_{lang}_{method}"


def get_cache_key_file(file_path: Path) -> str:
    """Generate cache key for local file based on content hash."""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = Path.home() / ".cache" / "yt-summarize"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_cached(cache_key: str, cache_type: str) -> dict[str, Any] | None:
    """Load cached data if it exists."""
    cache_file = get_cache_dir() / f"{cache_key}_{cache_type}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def save_to_cache(cache_key: str, cache_type: str, data: dict[str, Any]) -> None:
    """Save data to cache."""
    cache_file = get_cache_dir() / f"{cache_key}_{cache_type}.json"
    cache_file.write_text(json.dumps(data, indent=2))
