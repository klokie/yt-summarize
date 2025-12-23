"""Map-reduce summarization for long transcripts."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import tiktoken
from openai import OpenAI, OpenAIError

from .prompts import MAP_PROMPT, MAP_SYSTEM, REDUCE_PROMPT_JSON, REDUCE_PROMPT_MD, REDUCE_SYSTEM
from .schema import SummarySchema

# JSON Schema for structured output
MAP_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points": {"type": "array", "items": {"type": "string"}},
        "quotes": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "terms": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["key_points", "quotes", "topics", "terms"],
    "additionalProperties": False,
}

SUMMARY_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "source_url": {"type": "string"},
        "tldr": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "heading": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "heading", "bullets"],
                "additionalProperties": False,
            },
        },
        "quotes": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "source_url",
        "tldr",
        "key_points",
        "chapters",
        "quotes",
        "action_items",
        "tags",
    ],
    "additionalProperties": False,
}


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
    json_schema: dict[str, Any] | None = None,
) -> str:
    """
    Call OpenAI API with exponential backoff retry.

    Args:
        client: OpenAI client
        model: Model name
        system: System prompt
        user: User prompt
        max_retries: Number of retries
        temperature: Sampling temperature
        json_schema: Optional JSON schema for structured output
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }

            # Use structured output if schema provided
            if json_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": json_schema,
                    },
                }

            response = client.chat.completions.create(**kwargs)
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


def _map_chunk(client: OpenAI, chunk: str, model: str, use_structured: bool = True) -> dict:
    """
    Extract key info from a single chunk.

    Args:
        client: OpenAI client
        chunk: Transcript chunk
        model: Model name
        use_structured: Whether to use structured output (guaranteed valid JSON)
    """
    prompt = MAP_PROMPT.format(chunk=chunk)

    if use_structured:
        response = _call_with_retry(
            client, model, MAP_SYSTEM, prompt, json_schema=MAP_OUTPUT_SCHEMA
        )
        return json.loads(response)

    # Fallback: parse JSON from response
    response = _call_with_retry(client, model, MAP_SYSTEM, prompt)
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response)
    except json.JSONDecodeError:
        return {"key_points": [], "quotes": [], "topics": [], "terms": {}}


def _reduce_chunks_structured(
    client: OpenAI,
    chunk_summaries: list[dict],
    options: SummarizeOptions,
) -> SummarySchema:
    """
    Merge chunk summaries into final summary using structured output.

    Returns guaranteed valid SummarySchema.
    """
    formatted = json.dumps(chunk_summaries, indent=2)
    prompt = REDUCE_PROMPT_JSON.format(
        title=options.title,
        source_url=options.source_url,
        chunk_summaries=formatted,
    )

    response = _call_with_retry(
        client,
        options.model,
        REDUCE_SYSTEM,
        prompt,
        json_schema=SUMMARY_OUTPUT_SCHEMA,
    )

    json_data = json.loads(response)
    return SummarySchema(**json_data)


def _reduce_chunks(
    client: OpenAI,
    chunk_summaries: list[dict],
    options: SummarizeOptions,
    output_format: str,
) -> str:
    """Merge chunk summaries into final summary (text response)."""
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


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fence from text."""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[11:]
    elif text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def summarize_transcript(
    text: str,
    options: SummarizeOptions,
    use_structured_output: bool = True,
) -> tuple[str | None, SummarySchema | None]:
    """
    Summarize transcript using map-reduce pattern.

    Args:
        text: Full transcript text
        options: Summarization options
        use_structured_output: Use OpenAI structured output for guaranteed valid JSON

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
            summary = _map_chunk(client, chunk, options.model, use_structured=use_structured_output)
            chunk_summaries.append(summary)
        except Exception as e:
            raise SummarizationError(f"Failed on chunk {i + 1}/{len(chunks)}: {e}") from e

    # Reduce phase: merge into final summary
    md_result = None
    json_result = None
    formats = options.output_format.split(",")

    if "md" in formats:
        md_response = _reduce_chunks(client, chunk_summaries, options, "md")
        md_result = _strip_code_fence(md_response)

    if "json" in formats:
        if use_structured_output:
            # Use structured output for guaranteed valid schema
            json_result = _reduce_chunks_structured(client, chunk_summaries, options)
        else:
            # Fallback: parse JSON from response
            json_response = _reduce_chunks(client, chunk_summaries, options, "json")
            json_response = _strip_code_fence(json_response)
            try:
                json_data = json.loads(json_response)
                json_result = SummarySchema(**json_data)
            except (json.JSONDecodeError, ValueError) as e:
                raise SummarizationError(f"Failed to parse JSON response: {e}") from e

    return md_result, json_result


def summarize_short(
    text: str,
    options: SummarizeOptions,
    use_structured_output: bool = True,
) -> tuple[str | None, SummarySchema | None]:
    """
    Summarize short transcript without chunking.

    Use this for transcripts under ~3000 tokens.
    """
    token_count = count_tokens(text, options.model)

    if token_count > options.chunk_tokens * 2:
        return summarize_transcript(text, options, use_structured_output)

    client = _get_client()
    md_result = None
    json_result = None
    formats = options.output_format.split(",")

    # Create a single "chunk summary" from the full text
    chunk_summary = _map_chunk(client, text, options.model, use_structured=use_structured_output)

    if "md" in formats:
        md_response = _reduce_chunks(client, [chunk_summary], options, "md")
        md_result = _strip_code_fence(md_response)

    if "json" in formats:
        if use_structured_output:
            json_result = _reduce_chunks_structured(client, [chunk_summary], options)
        else:
            json_response = _reduce_chunks(client, [chunk_summary], options, "json")
            json_response = _strip_code_fence(json_response)
            try:
                json_data = json.loads(json_response)
                json_result = SummarySchema(**json_data)
            except (json.JSONDecodeError, ValueError) as e:
                raise SummarizationError(f"Failed to parse JSON response: {e}") from e

    return md_result, json_result
