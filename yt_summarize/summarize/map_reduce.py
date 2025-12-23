"""Map-reduce summarization for long transcripts."""

from dataclasses import dataclass

from .schema import SummarySchema


@dataclass
class SummarizeOptions:
    """Options for summarization."""

    title: str
    source_url: str
    model: str = "gpt-5-mini"
    chunk_tokens: int = 3000
    output_format: str = "md"  # "md" | "json" | "md,json"


def chunk_transcript(text: str, chunk_tokens: int = 3000) -> list[str]:
    """
    Split transcript into chunks of approximately chunk_tokens tokens.

    Uses tiktoken for accurate token counting.
    """
    # TODO: Implement with tiktoken
    # import tiktoken
    # enc = tiktoken.encoding_for_model("gpt-4")
    # tokens = enc.encode(text)
    # ...

    # Fallback: rough word-based chunking (1 token ≈ 0.75 words)
    words = text.split()
    words_per_chunk = int(chunk_tokens * 0.75)
    chunks = []

    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        chunks.append(chunk)

    return chunks


def summarize_transcript(
    text: str,
    options: SummarizeOptions,
) -> tuple[str | None, SummarySchema | None]:
    """
    Summarize transcript using map-reduce pattern.

    Returns:
        Tuple of (markdown_summary, json_schema) - either may be None based on format option
    """
    # TODO: Implement map-reduce summarization
    # 1. Chunk transcript
    # 2. Map: extract key info from each chunk
    # 3. Reduce: merge into final summary
    # 4. Format as MD and/or JSON

    raise NotImplementedError("Summarization not yet implemented")
