"""Prompt templates for summarization."""

MAP_PROMPT = """You are summarizing a chunk of a video transcript.

Extract the following from this chunk:
- Key points (bullet points)
- Notable quotes (exact quotes worth highlighting)
- Potential chapter headings with approximate timestamps if mentioned
- Technical terms or concepts that need definition

Be concise. Focus on substance over filler.

TRANSCRIPT CHUNK:
{chunk}

OUTPUT (JSON):
"""

REDUCE_PROMPT = """You are creating a final summary from multiple chunk summaries of a video transcript.

Video Title: {title}
Source URL: {source_url}

Merge and deduplicate the following chunk summaries into a cohesive final summary.

CHUNK SUMMARIES:
{chunk_summaries}

Create a summary with:
1. TL;DR: 3 bullet points capturing the essence
2. Key Points: 8-12 most important takeaways
3. Chapters: Logical sections with headings and bullets
4. Quotes: Up to 5 best quotes
5. Action Items: 3-7 practical next steps for the viewer
6. Tags: 5-10 topic tags

OUTPUT FORMAT: {output_format}
"""

MARKDOWN_FORMAT_INSTRUCTION = """Output as clean Markdown with sections:
# {title}
[Source]({url})

## TL;DR
- ...

## Key Points
- ...

## Chapters
### Chapter 1: ...
- ...

## Notable Quotes
> "..."

## Action Items
- [ ] ...

## Glossary
- **Term**: Definition
"""

JSON_FORMAT_INSTRUCTION = """Output as valid JSON matching this schema:
{
  "title": "string",
  "source_url": "string",
  "tldr": ["string"],
  "key_points": ["string"],
  "chapters": [{"start": "string", "heading": "string", "bullets": ["string"]}],
  "quotes": ["string"],
  "action_items": ["string"],
  "tags": ["string"]
}
"""
