"""CLI entry point for yt-summarize."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="yt-summarize",
    help="Fetch YouTube transcripts and generate AI summaries.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def main(
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
    ] = "gpt-5-mini",
    transcribe_model: Annotated[
        str,
        typer.Option("--transcribe-model", help="OpenAI STT model"),
    ] = "gpt-4o-mini-transcribe",
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Fetch transcript and generate summary from YouTube URL or local file."""
    console.print(f"[bold blue]yt-summarize[/bold blue] processing: {source}")

    # TODO: Implement pipeline
    # 1. Parse source (YouTube URL vs local file)
    # 2. Fetch/load transcript
    # 3. Summarize (unless --local-only)
    # 4. Write outputs

    console.print("[yellow]Not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
