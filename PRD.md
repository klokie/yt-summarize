# PRD — yt-summarize (CLI + library)

## 1) Decision: Python vs Node

**Recommendation: Python for v1.**

- Most reliable tooling for captions/subtitles + audio fallback (`youtube-transcript-api`, `yt-dlp`)
- Less dependency hell, easier chunking + caching, great for CLI workflows
Node is fine if you’re shipping a Next.js UI immediately, but you’ll still end up shelling out to `yt-dlp` and fighting transcript edge cases.

---

## 2) Problem

You want a dead-simple way to go from:

- YouTube URL (or local `.txt`)
to:
- `transcript.txt`
- `summary.md`
- optional `summary.json` (strict schema)

Fast path should complete in <60s when captions exist.

---

## 3) Users

Primary: you (dev) doing research/notes → paste URL → get clean summary for Notion / Cursor / docs.

---

## 4) User stories

1. Paste YouTube URL → get summary + transcript saved locally.
2. If captions exist → use them (fast, cheap).
3. If captions don’t exist → download audio-only → transcribe → summarize.
4. Re-run is instant (cache) unless `--force`.
5. Choose output format: `md`, `json`, or both.
6. Choose language: `auto`, `en`, `sv`, etc.

---

## 5) Non-goals

- Perfect speaker diarization
- Full UI + auth flows
- Playlist management, “watch later” syncing
- Video download beyond audio-only fallback

---

## 6) Pipeline (priority order)

### A) Captions (fastest)

1. Try `youtube-transcript-api` (auto-captions often work, no key).
2. If that fails: `yt-dlp --write-subs --write-auto-subs --skip-download`.

### B) Audio fallback

1. `yt-dlp -x --audio-format mp3` (or m4a)
2. OpenAI speech-to-text (`gpt-4o-mini-transcribe` default; `gpt-4o-transcribe` optional)

### C) Summarization

Use OpenAI Responses API:

- `gpt-5-mini` default (fast/good)
- Chunk map-reduce for long transcripts
- Output Markdown + optional structured JSON

---

## 7) CLI spec

Binary: `yt-summarize`

### Examples

```bash
yt-summarize "https://www.youtube.com/watch?v=VIDEO" --out ./out
yt-summarize "https://youtu.be/VIDEO" --lang en --format md,json
yt-summarize ./transcript.txt --title "Cool Talk" --format md
yt-summarize "https://www.youtube.com/watch?v=VIDEO" --force --model gpt-5-mini
```

### Flags

- `--out <dir>` default `./yt-summary/<id-or-file>/`
- `--lang <auto|en|sv|...>` default `auto`
- `--format <md|json|md,json>` default `md`
- `--force` ignore cache
- `--no-audio-fallback` fail if no captions
- `--max-minutes <n>` default `180` (safety)
- `--model <text-model>` default `gpt-5-mini`
- `--transcribe-model <stt-model>` default `gpt-4o-mini-transcribe`
- `--chunk-tokens <n>` default `3000`
- `--verbose`

---

## 8) Outputs

Inside output dir:

- `meta.json`
  - `{ source_url, video_id, title, channel, fetched_at, method: "captions"|"subs"|"stt", lang }`
- `transcript.txt`
- `summary.md`
- optional `summary.json` (schema-valid)

---

## 9) Summary format requirements

### Markdown (`summary.md`)

- Title + link
- TL;DR (3 bullets)
- Key points (8–12 bullets)
- Chapters (timestamp-ish if available)
- Quotes (max 5)
- Action items (3–7)
- Glossary / terms

### JSON (`summary.json`)

Strict schema:

```json
{
  "title": "string",
  "source_url": "string",
  "tldr": ["string"],
  "key_points": ["string"],
  "chapters": [
    { "start": "string", "heading": "string", "bullets": ["string"] }
  ],
  "quotes": ["string"],
  "action_items": ["string"],
  "tags": ["string"]
}
```

---

## 10) Chunking strategy (map-reduce)

- Split transcript into chunks by approx token count (`--chunk-tokens`)
- Map prompt: extract bullets, candidate chapters, strong quotes, terms
- Reduce prompt: merge + dedupe + produce final MD and/or JSON
- Deterministic-ish: set low temp for summarization

---

## 11) Caching

Cache key:

- YouTube: `videoId + lang + method`
- Local file: `sha256(file_contents)`

Cache stores:

- raw transcript
- (optional) per-chunk map results
- final summary

---

## 12) Error handling

- Captions disabled → fallback to audio unless `--no-audio-fallback`
- Age/region locked → clear error + exit non-zero
- Audio too long (> `--max-minutes`) → abort with message
- OpenAI errors → retry x3 with exponential backoff
- `yt-dlp` missing → install guidance + exit non-zero

---

## 13) Security & privacy

- Read API key from `OPENAI_API_KEY`
- Don’t print transcript in logs by default
- Optional `--local-only` to just export transcript (no OpenAI calls)

---

## 14) Tech plan: Python v1

### Dependencies

- `typer`
- `youtube-transcript-api`
- `openai`
- `tiktoken` (optional, for chunking)
- `rich` (optional, nice CLI output)
- `ruff`, `pytest`

### Repo layout

```
yt-summarize/
  pyproject.toml
  setup.py                 # legacy shim
  yt_summarize/
    cli.py
    cache.py
    sources/
      youtube.py
      local_file.py
    transcribe/
      openai_stt.py
    summarize/
      map_reduce.py
      prompts.py
      schema.py
  tests/
  README.md
  LICENSE                  # MIT
```

---

## 15) Acceptance criteria (DoD)

- 5 captioned YouTube URLs → summary + transcript produced successfully
- 2 non-captioned URLs → audio fallback transcribes + summarizes successfully
- Cache works; `--force` bypasses it
- `--format json` output validates against schema
- Unit tests cover: URL parsing, cache keys, chunking boundaries, markdown rendering

---

## 16) Milestones

1. CLI skeleton + folder output conventions
2. Caption fetch (youtube-transcript-api)
3. Subtitle/audio fallback via yt-dlp
4. STT transcription module
5. Summarization map-reduce (MD)
6. JSON schema + structured output mode
7. Cache + `--force`
8. Tests + CI
