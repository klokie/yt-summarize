"""CLI entry point for yt-summarize."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import typer.core
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .cache import (
    clear_cache,
    create_summary_cache,
    create_transcript_cache,
    get_cache_key_file,
    get_cache_key_youtube,
    get_cache_stats,
    list_cached,
    load_summary,
    load_transcript,
)
from .costs import (
    estimate_summarization_cost,
    estimate_transcription_cost,
    format_cost_warning,
)
from .sources.local_file import load_local_transcript
from .sources.youtube import (
    AudioDownloadResult,
    TranscriptNotAvailable,
    YtDlpNotFound,
    extract_video_id,
    fetch_youtube_transcript,
)
from .summarize import SummarizationError, SummarizeOptions, count_tokens, summarize_transcript
from .transcribe import TranscriptionError, transcribe_audio


# Main app - use TyperGroup subclass for default command support
class DefaultGroup(typer.core.TyperGroup):
    """TyperGroup that routes unknown commands to a default command."""

    def __init__(self, *args, default_cmd_name: str = "summarize", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def resolve_command(self, ctx, args):
        # If first arg looks like a file/URL rather than a command, route to default
        if args and args[0] not in self.commands:
            return self.default_cmd_name, self.get_command(ctx, self.default_cmd_name), args
        return super().resolve_command(ctx, args)


app = typer.Typer(
    name="yt-summarize",
    help="Fetch YouTube transcripts and generate AI summaries.",
    cls=DefaultGroup,
    no_args_is_help=True,
)

# Cache subcommand group
cache_app = typer.Typer(help="Manage cache.")
app.add_typer(cache_app, name="cache")

console = Console()


def _is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube URL."""
    return extract_video_id(source) is not None


def _sanitize_dirname(name: str, max_length: int = 100) -> str:
    """Sanitize a string for use as a directory name."""
    import re

    # Replace problematic characters with safe alternatives
    name = name.replace("/", "-").replace("\\", "-")
    name = name.replace(":", " -").replace("|", "-")
    name = name.replace("?", "").replace("*", "")
    name = name.replace('"', "'").replace("<", "").replace(">", "")

    # Collapse multiple spaces/dashes
    name = re.sub(r"[\s]+", " ", name)
    name = re.sub(r"-+", "-", name)

    # Strip and truncate
    name = name.strip(" .-")
    if len(name) > max_length:
        name = name[:max_length].strip(" .-")

    return name or "untitled"


def _build_frontmatter(meta: dict) -> str:
    """Build YAML front matter for the summary."""
    lines = ["---"]

    # Title
    title = meta.get("title", "Untitled")
    # Escape quotes in title for YAML
    title = title.replace('"', '\\"')
    lines.append(f'title: "{title}"')

    # Source URL
    if source_url := meta.get("source_url"):
        lines.append(f"source: {source_url}")

    # Upload date (original video date)
    if upload_date := meta.get("upload_date"):
        lines.append(f"date: {upload_date}")

    # Author section
    if channel := meta.get("channel"):
        lines.append("author:")
        lines.append(f'  name: "{channel}"')
        if channel_url := meta.get("channel_url"):
            lines.append(f"  url: {channel_url}")
        if handle := meta.get("uploader_handle"):
            lines.append(f"  youtube: {handle}")

    # Processing metadata
    if fetched_at := meta.get("fetched_at"):
        lines.append(f"fetched: {fetched_at}")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(
    out_dir: Path,
    transcript_text: str,
    md_summary: str | None,
    json_summary: dict | None,
    meta: dict,
) -> None:
    """Write output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write meta.json
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Write transcript
    (out_dir / "transcript.txt").write_text(transcript_text)

    # Write summary with front matter
    if md_summary:
        frontmatter = _build_frontmatter(meta)
        (out_dir / "summary.md").write_text(frontmatter + md_summary)

    if json_summary:
        (out_dir / "summary.json").write_text(
            json.dumps(json_summary, indent=2, ensure_ascii=False)
        )


@app.command()
def summarize(
    source: Annotated[str, typer.Argument(help="YouTube URL or local .txt file path")],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output directory"),
    ] = None,
    lang: Annotated[
        str,
        typer.Option("--lang", "-l", help="Language code (auto, en, sv, etc.)"),
    ] = "auto",
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: md, json, or md,json"),
    ] = "md",
    force: Annotated[
        bool,
        typer.Option("--force", help="Ignore cache and re-fetch/re-summarize"),
    ] = False,
    no_audio_fallback: Annotated[
        bool,
        typer.Option("--no-audio-fallback", help="Fail if no captions available"),
    ] = False,
    max_minutes: Annotated[
        int,
        typer.Option("--max-minutes", help="Maximum video length in minutes"),
    ] = 180,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="OpenAI model for summarization"),
    ] = "gpt-4o-mini",
    transcribe_model: Annotated[
        str,
        typer.Option("--transcribe-model", help="OpenAI STT model"),
    ] = "whisper-1",
    chunk_tokens: Annotated[
        int,
        typer.Option("--chunk-tokens", help="Token chunk size for map-reduce"),
    ] = 3000,
    title: Annotated[
        str | None,
        typer.Option("--title", "-t", help="Title for local file input"),
    ] = None,
    local_only: Annotated[
        bool,
        typer.Option("--local-only", help="Export transcript only, no AI calls"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip cost confirmation prompts"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Fetch transcript and generate summary from YouTube URL or local file."""
    is_youtube = _is_youtube_url(source)

    if verbose:
        console.print(f"[dim]Source type: {'YouTube' if is_youtube else 'Local file'}[/dim]")

    # Track if user provided explicit output dir
    user_provided_out = out is not None

    # For YouTube, we'll determine output dir after getting the title
    # For local files, use the filename stem
    if out is None and not is_youtube:
        out = Path("./yt-summary") / Path(source).stem

    transcript_text: str | None = None
    video_id: str | None = None
    meta: dict = {}
    cache_key: str | None = None

    # Step 1: Get transcript
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        if is_youtube:
            video_id = extract_video_id(source)

            # Check cache first (unless --force)
            if not force:
                # Try to find cached transcript with any method
                for method in ["captions", "subs", "stt"]:
                    cache_key = get_cache_key_youtube(video_id, lang, method)
                    cached = load_transcript(cache_key)
                    if cached:
                        if verbose:
                            console.print(f"[dim]Using cached transcript ({method})[/dim]")
                        transcript_text = cached.text
                        meta = {
                            "source_url": source,
                            "video_id": video_id,
                            "title": cached.title,
                            "channel": cached.channel,
                            "fetched_at": cached.cached_at,
                            "method": cached.method,
                            "lang": cached.lang,
                        }
                        break

            if transcript_text is None:
                progress.add_task("Fetching transcript...", total=None)
                # Use video_id for audio temp dir (title not known yet)
                audio_dir = (
                    Path("./yt-summary") / video_id / ".audio" if not no_audio_fallback else None
                )
                try:
                    result = fetch_youtube_transcript(
                        source,
                        lang=lang,
                        allow_audio_fallback=not no_audio_fallback,
                        audio_output_dir=audio_dir,
                        max_minutes=max_minutes,
                    )

                    if isinstance(result, AudioDownloadResult):
                        # Cost warning for STT
                        stt_estimate = estimate_transcription_cost(
                            result.duration_seconds, transcribe_model
                        )
                        if stt_estimate["should_warn"] and not yes:
                            console.print(
                                format_cost_warning(
                                    "Audio transcription",
                                    stt_estimate["estimated_cost"],
                                    f"{stt_estimate['duration_minutes']:.1f} min of audio",
                                )
                            )
                            if not typer.confirm("Continue?"):
                                raise typer.Exit(0)

                        # Need to transcribe audio
                        progress.add_task("Transcribing audio...", total=None)
                        try:
                            transcript_text = transcribe_audio(
                                result.audio_path,
                                model=transcribe_model,
                                lang=lang if lang != "auto" else None,
                            )
                            method = "stt"
                            meta = {
                                "source_url": source,
                                "video_id": result.video_id,
                                "title": result.title,
                                "channel": result.channel,
                                "channel_url": result.channel_url,
                                "uploader_handle": result.uploader_handle,
                                "upload_date": result.upload_date,
                                "fetched_at": datetime.now().isoformat(),
                                "method": method,
                                "lang": lang,
                            }
                        except TranscriptionError as e:
                            console.print(f"[red]Transcription failed:[/red] {e}")
                            raise typer.Exit(1) from e
                    else:
                        # Got transcript directly
                        transcript_text = result.text
                        method = result.method
                        meta = {
                            "source_url": source,
                            "video_id": result.video_id,
                            "title": result.title,
                            "channel": result.channel,
                            "channel_url": result.channel_url,
                            "uploader_handle": result.uploader_handle,
                            "upload_date": result.upload_date,
                            "fetched_at": datetime.now().isoformat(),
                            "method": method,
                            "lang": result.lang,
                        }

                    # Cache the transcript
                    cache_key, _ = create_transcript_cache(
                        video_id=video_id,
                        text=transcript_text,
                        title=meta["title"],
                        channel=meta["channel"],
                        lang=meta["lang"],
                        method=meta["method"],
                    )
                    if verbose:
                        console.print(f"[dim]Cached transcript with key: {cache_key}[/dim]")

                except TranscriptNotAvailable as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise typer.Exit(1) from e
                except YtDlpNotFound as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise typer.Exit(1) from e
        else:
            # Local file
            file_path = Path(source)
            if not file_path.exists():
                console.print(f"[red]File not found:[/red] {source}")
                raise typer.Exit(1)

            cache_key = get_cache_key_file(file_path)

            # Check cache
            if not force:
                cached = load_transcript(cache_key)
                if cached:
                    if verbose:
                        console.print("[dim]Using cached transcript[/dim]")
                    transcript_text = cached.text
                    meta = {
                        "source_file": str(file_path.absolute()),
                        "title": cached.title,
                        "fetched_at": cached.cached_at,
                        "method": "file",
                    }

            if transcript_text is None:
                progress.add_task("Loading file...", total=None)
                result = load_local_transcript(file_path, title=title)
                transcript_text = result.text
                meta = {
                    "source_file": str(file_path.absolute()),
                    "title": result.title,
                    "fetched_at": datetime.now().isoformat(),
                    "method": "file",
                }

                # Cache it
                create_transcript_cache(
                    video_id=cache_key,
                    text=transcript_text,
                    title=meta["title"],
                    channel="",
                    lang="",
                    method="file",
                )

    # Determine output directory from title (if not user-provided)
    if not user_provided_out and is_youtube:
        video_title = meta.get("title", video_id)
        out = Path("./yt-summary") / _sanitize_dirname(video_title)

    console.print(f"[green]✓[/green] Transcript: {len(transcript_text)} chars")

    # Step 2: Summarize (unless --local-only)
    md_summary: str | None = None
    json_summary: dict | None = None

    if not local_only:
        # Check summary cache
        cached_summary = None
        if not force and cache_key:
            cached_summary = load_summary(cache_key, format)
            if cached_summary:
                if verbose:
                    console.print("[dim]Using cached summary[/dim]")
                md_summary = cached_summary.markdown
                json_summary = cached_summary.json_data

        if cached_summary is None:
            # Cost warning for summarization
            token_count = count_tokens(transcript_text, model)
            summary_estimate = estimate_summarization_cost(token_count, chunk_tokens, model)

            if summary_estimate["should_warn"] and not yes:
                console.print(
                    format_cost_warning(
                        "Summarization",
                        summary_estimate["estimated_cost"],
                        f"{token_count:,} tokens → {summary_estimate['num_chunks']} chunks",
                    )
                )
                if not typer.confirm("Continue?"):
                    raise typer.Exit(0)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Generating summary...", total=None)

                try:
                    opts = SummarizeOptions(
                        title=meta.get("title", "Untitled"),
                        source_url=meta.get("source_url", meta.get("source_file", "")),
                        model=model,
                        chunk_tokens=chunk_tokens,
                        output_format=format,
                    )
                    md_summary, json_result = summarize_transcript(transcript_text, opts)
                    if json_result:
                        json_summary = json_result.model_dump()

                    # Cache the summary
                    if cache_key:
                        create_summary_cache(
                            cache_key=cache_key,
                            markdown=md_summary,
                            json_data=json_summary,
                            model=model,
                        )
                        if verbose:
                            console.print("[dim]Cached summary[/dim]")

                except SummarizationError as e:
                    console.print(f"[red]Summarization failed:[/red] {e}")
                    raise typer.Exit(1) from e

        console.print("[green]✓[/green] Summary generated")

    # Step 3: Write outputs
    _write_outputs(out, transcript_text, md_summary, json_summary, meta)

    console.print(Panel(f"[bold green]Done![/bold green]\n\nOutput: {out}"))


# Cache subcommands
@cache_app.command("list")
def cache_list() -> None:
    """List cached entries."""
    entries = list_cached()
    if not entries:
        console.print("[dim]Cache is empty[/dim]")
        return

    for entry in entries:
        summary_indicator = "📝" if entry["has_summary"] else "  "
        console.print(
            f"{summary_indicator} [bold]{entry['title'][:50]}[/bold] "
            f"[dim]({entry['video_id']}, {entry['method']})[/dim]"
        )


@cache_app.command("stats")
def cache_stats() -> None:
    """Show cache statistics."""
    stats = get_cache_stats()
    console.print(f"Cache directory: {stats['cache_dir']}")
    console.print(f"Transcripts: {stats['transcript_count']}")
    console.print(f"Summaries: {stats['summary_count']}")
    console.print(f"Total size: {stats['total_size_kb']:.1f} KB")


@cache_app.command("clear")
def cache_clear(
    key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="Specific cache key to clear"),
    ] = None,
    all_entries: Annotated[
        bool,
        typer.Option("--all", "-a", help="Clear all cache entries"),
    ] = False,
) -> None:
    """Clear cache entries."""
    if not key and not all_entries:
        console.print("[yellow]Specify --key or --all to clear cache[/yellow]")
        raise typer.Exit(1)

    count = clear_cache(key)
    console.print(f"[green]Cleared {count} cache files[/green]")


if __name__ == "__main__":
    app()
