"""CLI entry point for yt-summarize."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
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

# Main app
app = typer.Typer(
    name="yt-summarize",
    help="Fetch YouTube transcripts and generate AI summaries.",
    no_args_is_help=True,
)

# Cache subcommand group
cache_app = typer.Typer(help="Manage cache.")
app.add_typer(cache_app, name="cache")

console = Console()


def _is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube URL."""
    return extract_video_id(source) is not None


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

    # Write summary
    if md_summary:
        (out_dir / "summary.md").write_text(md_summary)

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

    # Determine output directory
    if out is None:
        if is_youtube:
            video_id = extract_video_id(source)
            out = Path("./yt-summary") / video_id
        else:
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
                try:
                    result = fetch_youtube_transcript(
                        source,
                        lang=lang,
                        allow_audio_fallback=not no_audio_fallback,
                        audio_output_dir=out / ".audio" if not no_audio_fallback else None,
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
