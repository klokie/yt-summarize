"""Cost estimation utilities for API calls."""

# Approximate costs per 1M tokens (as of Dec 2024)
# These are estimates - actual costs may vary
MODEL_COSTS = {
    # Summarization models (input/output per 1M tokens)
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # STT models (per minute of audio)
    "whisper-1": {"per_minute": 0.006},
    "gpt-4o-transcribe": {"per_minute": 0.006},
    "gpt-4o-mini-transcribe": {"per_minute": 0.003},
}

# Thresholds for warnings
WARN_TRANSCRIPT_TOKENS = 50_000  # ~20+ chunks, many API calls
WARN_AUDIO_MINUTES = 30  # 30+ minutes of audio
WARN_ESTIMATED_COST = 0.50  # $0.50 threshold for warning


def estimate_summarization_cost(
    token_count: int,
    chunk_tokens: int = 3000,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Estimate cost for summarization.

    Returns dict with:
        - num_chunks: number of transcript chunks
        - estimated_input_tokens: total input tokens (chunks + prompts)
        - estimated_output_tokens: approximate output tokens
        - estimated_cost: cost in USD
        - should_warn: whether to show warning
    """
    costs = MODEL_COSTS.get(model, MODEL_COSTS["gpt-4o-mini"])

    # Estimate chunks
    num_chunks = max(1, token_count // chunk_tokens)

    # Map phase: each chunk + prompt (~500 tokens) -> ~200 token response
    map_input = num_chunks * (chunk_tokens + 500)
    map_output = num_chunks * 200

    # Reduce phase: all chunk summaries + prompt -> final summary
    reduce_input = num_chunks * 200 + 1000  # chunk summaries + prompt
    reduce_output = 2000  # final summary

    total_input = map_input + reduce_input
    total_output = map_output + reduce_output

    # Cost calculation
    input_cost = (total_input / 1_000_000) * costs["input"]
    output_cost = (total_output / 1_000_000) * costs["output"]
    total_cost = input_cost + output_cost

    return {
        "num_chunks": num_chunks,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost": total_cost,
        "should_warn": token_count > WARN_TRANSCRIPT_TOKENS or total_cost > WARN_ESTIMATED_COST,
    }


def estimate_transcription_cost(
    duration_seconds: float,
    model: str = "whisper-1",
) -> dict:
    """
    Estimate cost for audio transcription.

    Returns dict with:
        - duration_minutes: audio length in minutes
        - estimated_cost: cost in USD
        - should_warn: whether to show warning
    """
    costs = MODEL_COSTS.get(model, MODEL_COSTS["whisper-1"])
    duration_minutes = duration_seconds / 60

    cost = duration_minutes * costs.get("per_minute", 0.006)

    return {
        "duration_minutes": duration_minutes,
        "estimated_cost": cost,
        "should_warn": duration_minutes > WARN_AUDIO_MINUTES or cost > WARN_ESTIMATED_COST,
    }


def format_cost_warning(
    operation: str,
    estimated_cost: float,
    details: str = "",
) -> str:
    """Format a cost warning message."""
    msg = f"⚠️  {operation} may cost approximately ${estimated_cost:.3f}"
    if details:
        msg += f"\n   {details}"
    return msg

