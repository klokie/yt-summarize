# yt-summarize

CLI tool to fetch YouTube transcripts and generate AI-powered summaries.

## Features

- Extract transcripts from YouTube videos (captions or audio fallback)
- Generate structured summaries using OpenAI
- Output as Markdown, JSON, or both
- Smart caching for fast re-runs
- Map-reduce chunking for long videos

## Installation

```bash
# Clone and install in dev mode
git clone https://github.com/klokie/yt-summarize.git
cd yt-summarize
pip install -e ".[dev]"

# For audio fallback, install yt-dlp
brew install yt-dlp  # or: pip install yt-dlp
```

## Usage

```bash
# Basic usage
yt-summarize "https://www.youtube.com/watch?v=VIDEO_ID"

# With options
yt-summarize "https://youtu.be/VIDEO" --lang en --format md,json --out ./summaries

# From local transcript file
yt-summarize ./transcript.txt --title "Cool Talk"

# Force refresh (ignore cache)
yt-summarize "https://www.youtube.com/watch?v=VIDEO" --force
```

## Options

| Flag                  | Default                  | Description                         |
| --------------------- | ------------------------ | ----------------------------------- |
| `--out`               | `./yt-summary/<id>/`     | Output directory                    |
| `--lang`              | `auto`                   | Language code (en, sv, etc.)        |
| `--format`            | `md`                     | Output format: md, json, or md,json |
| `--force`             | false                    | Ignore cache                        |
| `--no-audio-fallback` | false                    | Fail if no captions available       |
| `--max-minutes`       | 180                      | Max video length                    |
| `--model`             | `gpt-5-mini`             | OpenAI model for summarization      |
| `--transcribe-model`  | `gpt-4o-mini-transcribe` | OpenAI STT model                    |
| `--chunk-tokens`      | 3000                     | Token chunk size for map-reduce     |
| `--local-only`        | false                    | Export transcript only, no AI       |
| `--verbose`           | false                    | Verbose output                      |

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
```

## Output Files

```
yt-summary/<video-id>/
├── meta.json        # Video metadata
├── transcript.txt   # Raw transcript
├── summary.md       # Markdown summary
└── summary.json     # Structured JSON (optional)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
ruff format .

# Type check
mypy yt_summarize
```

## License

MIT
