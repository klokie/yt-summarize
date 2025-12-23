"""Map-reduce summarization for long transcripts."""

import json
import os
import time
from dataclasses import dataclass

import tiktoken
from openai import OpenAI, OpenAIError

from .prompts import MAP_PROMPT, MAP_SYSTEM, REDUCE_PROMPT_JSON, REDUCE_PROMPT_MD, REDUCE_SYSTEM
from .schema import SummarySchema


class SummarizationError(Exception):
    """Raised when summarization fails."""


@dataclass
class SummarizeOptions:
    """Options for summarization."""

    title: str
    source_url: str
    model: str = "gpt-4o-mini"
    chunk_tokens: int = 3000
    output_format: str = "md"  # "md" | "json" | "md,json"


def _get_client() -> OpenAI:
    """Get OpenAI client, checking for API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SummarizationError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def _call_with_retry(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_retries: int = 3,
    temperature: float = 0.3,
) -> str:
    """Call OpenAI API with exponential backoff retry."""
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        except OpenAIError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)

    raise SummarizationError(f"API call failed after {max_retries} attempts: {last_error}")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_transcript(text: str, chunk_tokens: int = 3000, model: str = "gpt-4o-mini") -> list[str]:
    """
    Split transcript into chunks of approximately chunk_tokens tokens.

    Uses tiktoken for accurate token counting.
    Tries to split on sentence boundaries.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    # Split into sentences (rough approximation)
    sentences = text.replace("。", ". ").replace("？", "? ").replace("！", "! ").split(". ")
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    chunks = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))

        if current_tokens + sentence_tokens > chunk_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _map_chunk(client: OpenAI, chunk: str, model: str) -> dict:
    """Extract key info from a single chunk."""
    prompt = MAP_PROMPT.format(chunk=chunk)
    response = _call_with_retry(client, model, MAP_SYSTEM, prompt)

    # Parse JSON response
    try:
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response)
    except json.JSONDecodeError:
        # Return empty structure if parsing fails
        return {"key_points": [], "quotes": [], "topics": [], "terms": {}}


def _reduce_chunks(
    client: OpenAI,
    chunk_summaries: list[dict],
    options: SummarizeOptions,
    output_format: str,
) -> str:
    """Merge chunk summaries into final summary."""
    # Format chunk summaries for the prompt
    formatted = json.dumps(chunk_summaries, indent=2)

    if output_format == "json":
        prompt = REDUCE_PROMPT_JSON.format(
            title=options.title,
            source_url=options.source_url,
            chunk_summaries=formatted,
        )
    else:
        prompt = REDUCE_PROMPT_MD.format(
            title=options.title,
            source_url=options.source_url,
            chunk_summaries=formatted,
        )

    return _call_with_retry(client, options.model, REDUCE_SYSTEM, prompt)


def summarize_transcript(
    text: str,
    options: SummarizeOptions,
) -> tuple[str | None, SummarySchema | None]:
    """
    Summarize transcript using map-reduce pattern.

    Args:
        text: Full transcript text
        options: Summarization options

    Returns:
        Tuple of (markdown_summary, json_schema) - either may be None based on format option

    Raises:
        SummarizationError: If summarization fails
    """
    client = _get_client()

    # Chunk the transcript
    chunks = chunk_transcript(text, options.chunk_tokens, options.model)

    # Map phase: extract info from each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = _map_chunk(client, chunk, options.model)
            chunk_summaries.append(summary)
        except Exception as e:
            raise SummarizationError(f"Failed on chunk {i + 1}/{len(chunks)}: {e}") from e

    # Reduce phase: merge into final summary
    md_result = None
    json_result = None

    formats = options.output_format.split(",")

    if "md" in formats:
        md_response = _reduce_chunks(client, chunk_summaries, options, "md")
        md_result = md_response.strip()
        # Remove markdown code fence if present
        if md_result.startswith("```markdown"):
            md_result = md_result[11:]
        if md_result.startswith("```"):
            md_result = md_result[3:]
        if md_result.endswith("```"):
            md_result = md_result[:-3]
        md_result = md_result.strip()

    if "json" in formats:
        json_response = _reduce_chunks(client, chunk_summaries, options, "json")
        json_response = json_response.strip()
        # Remove code fence if present
        if json_response.startswith("```json"):
            json_response = json_response[7:]
        if json_response.startswith("```"):
            json_response = json_response[3:]
        if json_response.endswith("```"):
            json_response = json_response[:-3]

        try:
            json_data = json.loads(json_response.strip())
            json_result = SummarySchema(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise SummarizationError(f"Failed to parse JSON response: {e}") from e

    return md_result, json_result


def summarize_short(
    text: str,
    options: SummarizeOptions,
) -> tuple[str | None, SummarySchema | None]:
    """
    Summarize short transcript without chunking.

    Use this for transcripts under ~3000 tokens.
    """
    token_count = count_tokens(text, options.model)

    if token_count > options.chunk_tokens * 2:
        # Too long, use map-reduce
        return summarize_transcript(text, options)

    # Short enough to summarize directly
    client = _get_client()

    md_result = None
    json_result = None
    formats = options.output_format.split(",")

    # Create a single "chunk summary" from the full text
    chunk_summary = _map_chunk(client, text, options.model)

    if "md" in formats:
        md_response = _reduce_chunks(client, [chunk_summary], options, "md")
        md_result = md_response.strip()
        if md_result.startswith("```"):
            lines = md_result.split("\n")
            md_result = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    if "json" in formats:
        json_response = _reduce_chunks(client, [chunk_summary], options, "json")
        json_response = json_response.strip()
        if json_response.startswith("```"):
            lines = json_response.split("\n")
            json_response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            json_data = json.loads(json_response)
            json_result = SummarySchema(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise SummarizationError(f"Failed to parse JSON response: {e}") from e

    return md_result, json_result
