from performance_metrics import extract_ollama_metrics


def test_extract_ollama_metrics_converts_ns_and_rates():
    metrics = extract_ollama_metrics(
        {
            'total_duration': 2_000_000_000,
            'load_duration': 100_000_000,
            'prompt_eval_count': 1000,
            'prompt_eval_duration': 500_000_000,
            'eval_count': 200,
            'eval_duration': 1_000_000_000,
        }
    )

    assert metrics['total_ms'] == 2000.0
    assert metrics['load_ms'] == 100.0
    assert metrics['prompt_eval_ms'] == 500.0
    assert metrics['eval_ms'] == 1000.0
    assert metrics['prompt_tokens_per_s'] == 2000.0
    assert metrics['output_tokens_per_s'] == 200.0


def test_extract_ollama_metrics_handles_missing_fields():
    metrics = extract_ollama_metrics({})

    assert metrics['total_ms'] is None
    assert metrics['prompt_tokens_per_s'] is None
    assert metrics['output_tokens_per_s'] is None
