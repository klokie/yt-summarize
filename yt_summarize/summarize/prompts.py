"""Prompt templates for summarization."""

MAP_SYSTEM = """You are an expert at extracting key information from video transcripts.
Your task is to identify the most important content from transcript chunks.
Be concise and focus on substance. Ignore filler words and repetition."""

MAP_PROMPT = """Extract the following from this transcript chunk:

1. KEY_POINTS: 3-5 most important points (as bullet points)
2. QUOTES: 1-3 notable/quotable statements (exact wording if possible)
3. TOPICS: Main topics or concepts discussed
4. TERMS: Technical terms or jargon that might need definition

TRANSCRIPT CHUNK:
{chunk}

Respond in this exact JSON format:
{{
  "key_points": ["point 1", "point 2", ...],
  "quotes": ["quote 1", ...],
  "topics": ["topic 1", ...],
  "terms": [{{"term": "example", "definition": "explanation"}}, ...]
}}"""

REDUCE_SYSTEM = """You are an expert at creating comprehensive video summaries.
Your task is to synthesize multiple chunk extractions into a cohesive summary.
Deduplicate similar points and organize information logically."""

REDUCE_PROMPT_MD = """Create a comprehensive summary from these extracted notes.

Video Title: {title}
Source URL: {source_url}

EXTRACTED NOTES:
{chunk_summaries}

Create a Markdown summary with these sections:

# {title}
[Source]({source_url})

## TL;DR
- (3 bullet points capturing the essence)

## Key Points
- (8-12 most important takeaways, deduplicated and organized)

## Chapters
### [Topic 1]
- key points for this section

### [Topic 2]
- key points for this section

(organize into 3-6 logical chapters based on topic flow)

## Notable Quotes
> "exact quote here"

(up to 5 best quotes)

## Action Items
- [ ] (3-7 practical next steps for the viewer)

## Glossary
- **Term**: Definition

(only include if technical terms were found)

Output ONLY the Markdown, no additional commentary."""

REDUCE_PROMPT_JSON = """Create a comprehensive summary from these extracted notes.

Video Title: {title}
Source URL: {source_url}

EXTRACTED NOTES:
{chunk_summaries}

Output a JSON object with this exact structure:
{{
  "title": "{title}",
  "source_url": "{source_url}",
  "tldr": ["point 1", "point 2", "point 3"],
  "key_points": ["8-12 most important takeaways"],
  "chapters": [
    {{"start": "0:00", "heading": "Introduction", "bullets": ["point 1", "point 2"]}},
    {{"start": "~5:00", "heading": "Topic Name", "bullets": ["point 1"]}}
  ],
  "quotes": ["quote 1", "quote 2"],
  "action_items": ["action 1", "action 2"],
  "tags": ["tag1", "tag2", "tag3"]
}}

Rules:
- tldr: exactly 3 bullet points
- key_points: 8-12 items
- chapters: 3-6 logical sections
- quotes: up to 5 best quotes
- action_items: 3-7 practical next steps
- tags: 5-10 topic tags

Output ONLY valid JSON, no markdown or commentary."""
