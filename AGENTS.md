# Agent Guidelines for yt-summarize

## Project Overview

CLI tool to fetch YouTube transcripts and generate AI summaries using OpenAI.

## Before Committing

**Always run these before committing:**

```bash
# Fix lint errors
ruff check --fix yt_summarize/ tests/
ruff format yt_summarize/ tests/

# Run all tests
python -m pytest tests/ -v
```

CI will fail if linting or tests fail. Fix all errors before committing.

## Code Style & Conventions

### Python

- Use `ruff` for linting and formatting
- Run `ruff check --fix` and `ruff format` before committing
- Import sorting is enforced (isort-style via ruff)
- Type hints required (uses `Annotated` for typer options)

### CLI (Typer)

- Default command pattern: Uses custom `DefaultGroup(typer.core.TyperGroup)` to route unknown commands to `summarize`
- This allows `yt-summarize <url>` instead of requiring `yt-summarize summarize <url>`
- Subcommands (like `cache`) still work normally

### File/Directory Naming

- Output directories use sanitized video titles, not video IDs
- `_sanitize_dirname()` handles filesystem-unsafe characters: `/ \ : | ? * " < >`
- Max length 100 chars, collapses whitespace/dashes

## Architecture Decisions

### Deferred Output Directory

The output directory for YouTube videos is determined **after** fetching the transcript (not before), because we need the video title. For audio fallback, a temporary path using video_id is used.

```python
# Audio temp dir uses video_id (title unknown yet)
audio_dir = Path("./yt-summary") / video_id / ".audio"

# Final output dir set after transcript fetch
out = Path("./yt-summary") / _sanitize_dirname(meta["title"])
```

### Caching

- Cache keys for YouTube: `{video_id}_{lang}_{method}`
- Cache keys for files: hash of file path + mtime
- Transcripts and summaries cached separately

## Testing

- All tests in `tests/` directory
- Use `pytest` with coverage
- CLI tests use `typer.testing.CliRunner`
- Mock external API calls (OpenAI, YouTube)

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=yt_summarize
```

## Common Tasks

### Adding a new CLI option

1. Add to `summarize()` function signature with `Annotated[type, typer.Option(...)]`
2. Update tests in `tests/test_cli.py`

### Renaming output folders from old runs

```python
# Read title from meta.json and rename
import json, os
with open(f"{folder}/meta.json") as f:
    title = json.load(f)["title"]
os.rename(folder, sanitize(title))
```

## Dependencies

- `typer` + `rich` for CLI
- `openai` for summarization and transcription
- `youtube-transcript-api` for captions
- `yt-dlp` for audio fallback (optional)
- `tiktoken` for token counting
