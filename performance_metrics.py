from __future__ import annotations

import time
from typing import Any


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    if end_ns is None:
        end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1_000_000.0


def ns_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _rate(count: Any, duration_ns: Any) -> float | None:
    try:
        count_value = float(count)
        duration_value = float(duration_ns)
    except (TypeError, ValueError):
        return None

    if duration_value <= 0:
        return None
    return count_value / (duration_value / 1_000_000_000.0)


def extract_ollama_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract Ollama timing/token counters from a completed response."""
    prompt_count = payload.get('prompt_eval_count')
    eval_count = payload.get('eval_count')
    prompt_duration = payload.get('prompt_eval_duration')
    eval_duration = payload.get('eval_duration')

    return {
        'total_ms': ns_to_ms(payload.get('total_duration')),
        'load_ms': ns_to_ms(payload.get('load_duration')),
        'prompt_eval_ms': ns_to_ms(prompt_duration),
        'eval_ms': ns_to_ms(eval_duration),
        'prompt_tokens': prompt_count,
        'output_tokens': eval_count,
        'prompt_tokens_per_s': _rate(prompt_count, prompt_duration),
        'output_tokens_per_s': _rate(eval_count, eval_duration),
    }
