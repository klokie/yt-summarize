"""Caching utilities for transcripts and summaries."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CachedTranscript:
    """Cached transcript data."""

    text: str
    video_id: str
    title: str
    channel: str
    lang: str
    method: str  # "captions" | "subs" | "stt"
    cached_at: str


@dataclass
class CachedSummary:
    """Cached summary data."""

    markdown: str | None
    json_data: dict[str, Any] | None
    model: str
    cached_at: str


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


def _get_cache_path(cache_key: str, cache_type: str) -> Path:
    """Get path to cache file."""
    return get_cache_dir() / f"{cache_key}_{cache_type}.json"


def load_cached(cache_key: str, cache_type: str) -> dict[str, Any] | None:
    """Load cached data if it exists."""
    cache_file = _get_cache_path(cache_key, cache_type)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            return None
    return None


def save_to_cache(cache_key: str, cache_type: str, data: dict[str, Any]) -> None:
    """Save data to cache."""
    cache_file = _get_cache_path(cache_key, cache_type)
    cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_transcript(cache_key: str) -> CachedTranscript | None:
    """Load cached transcript."""
    data = load_cached(cache_key, "transcript")
    if data:
        return CachedTranscript(**data)
    return None


def save_transcript(cache_key: str, transcript: CachedTranscript) -> None:
    """Save transcript to cache."""
    save_to_cache(cache_key, "transcript", asdict(transcript))


def load_summary(cache_key: str, output_format: str) -> CachedSummary | None:
    """
    Load cached summary.

    Args:
        cache_key: Cache key
        output_format: "md" | "json" | "md,json"

    Returns:
        CachedSummary if found and contains requested format(s), else None
    """
    data = load_cached(cache_key, "summary")
    if not data:
        return None

    summary = CachedSummary(**data)

    # Check if cached summary has the requested format(s)
    formats = output_format.split(",")
    if "md" in formats and not summary.markdown:
        return None
    if "json" in formats and not summary.json_data:
        return None

    return summary


def save_summary(cache_key: str, summary: CachedSummary) -> None:
    """Save summary to cache."""
    save_to_cache(cache_key, "summary", asdict(summary))


def clear_cache(cache_key: str | None = None) -> int:
    """
    Clear cache entries.

    Args:
        cache_key: If provided, clear only entries for this key.
                   If None, clear all cache.

    Returns:
        Number of files deleted
    """
    cache_dir = get_cache_dir()
    count = 0

    if cache_key:
        # Clear specific key
        for suffix in ["transcript", "summary"]:
            cache_file = _get_cache_path(cache_key, suffix)
            if cache_file.exists():
                cache_file.unlink()
                count += 1
    else:
        # Clear all
        for cache_file in cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

    return count


def list_cached() -> list[dict[str, Any]]:
    """List all cached entries."""
    cache_dir = get_cache_dir()
    entries = []

    for cache_file in sorted(cache_dir.glob("*_transcript.json")):
        try:
            data = json.loads(cache_file.read_text())
            entries.append(
                {
                    "cache_key": cache_file.stem.replace("_transcript", ""),
                    "title": data.get("title", "Unknown"),
                    "video_id": data.get("video_id", ""),
                    "method": data.get("method", ""),
                    "cached_at": data.get("cached_at", ""),
                    "has_summary": _get_cache_path(
                        cache_file.stem.replace("_transcript", ""), "summary"
                    ).exists(),
                }
            )
        except (json.JSONDecodeError, KeyError):
            pass

    return entries


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    cache_dir = get_cache_dir()

    transcript_files = list(cache_dir.glob("*_transcript.json"))
    summary_files = list(cache_dir.glob("*_summary.json"))

    total_size = sum(f.stat().st_size for f in cache_dir.glob("*.json"))

    return {
        "cache_dir": str(cache_dir),
        "transcript_count": len(transcript_files),
        "summary_count": len(summary_files),
        "total_size_kb": total_size / 1024,
    }


def create_transcript_cache(
    video_id: str,
    text: str,
    title: str,
    channel: str,
    lang: str,
    method: str,
) -> tuple[str, CachedTranscript]:
    """
    Create and save transcript cache entry.

    Returns:
        Tuple of (cache_key, cached_transcript)
    """
    cache_key = get_cache_key_youtube(video_id, lang, method)
    transcript = CachedTranscript(
        text=text,
        video_id=video_id,
        title=title,
        channel=channel,
        lang=lang,
        method=method,
        cached_at=datetime.now().isoformat(),
    )
    save_transcript(cache_key, transcript)
    return cache_key, transcript


def create_summary_cache(
    cache_key: str,
    markdown: str | None,
    json_data: dict[str, Any] | None,
    model: str,
) -> CachedSummary:
    """
    Create and save summary cache entry.

    Returns:
        CachedSummary
    """
    summary = CachedSummary(
        markdown=markdown,
        json_data=json_data,
        model=model,
        cached_at=datetime.now().isoformat(),
    )
    save_summary(cache_key, summary)
    return summary
