from diagnostics.analyze_timing import summarize


def row(name, total, asr, p1, p2, words, audio, true_count):
    return {
        'audio_filename': name,
        'request_core_ms': total,
        'base64_decode_ms': 1.0,
        'asr_total_ms': asr,
        'asr_whisper_ms': asr - 1.0,
        'audio_duration_s': audio,
        'asr_realtime_factor': (asr / 1000.0) / audio,
        'segment_count': 10.0,
        'word_count': words,
        'pass1_total_ms': p1,
        'pass1_prompt_build_ms': 1.0,
        'pass1_http_ms': p1 - 2.0,
        'pass1_parse_ms': 1.0,
        'pass1_ollama_total_ms': p1 - 3.0,
        'pass1_load_ms': 0.0,
        'pass1_prompt_eval_ms': p1 * 0.4,
        'pass1_eval_ms': p1 * 0.5,
        'pass1_prompt_tokens': words * 10.0,
        'pass1_output_tokens': 100.0,
        'pass1_prompt_tokens_per_s': 1000.0,
        'pass1_output_tokens_per_s': 50.0,
        'pass1_retry_count': 0.0,
        'pass2_enabled': True,
        'true_question_count': true_count,
        'pass2_total_ms': p2,
        'pass2_prompt_build_ms': 1.0,
        'pass2_http_ms': p2 - 2.0,
        'pass2_parse_ms': 1.0,
        'pass2_ollama_total_ms': p2 - 3.0,
        'pass2_load_ms': 0.0,
        'pass2_prompt_eval_ms': p2 * 0.6,
        'pass2_eval_ms': p2 * 0.3,
        'pass2_prompt_tokens': words * 11.0,
        'pass2_output_tokens': true_count * 20.0,
        'pass2_prompt_tokens_per_s': 900.0,
        'pass2_output_tokens_per_s': 55.0,
        'pass2_retry_count': 0.0,
        'evidence_resolution_ms': 2.0,
        'unaccounted_ms': total - asr - p1 - p2 - 3.0,
        'diagnostic_payload_build_ms': 2.0,
        'diagnostic_json_serialize_ms': 1.0,
    }


def test_timing_summary_reports_stage_stats_and_slowest():
    rows = [
        row('a.mp3', 100.0, 20.0, 40.0, 30.0, 100.0, 60.0, 3.0),
        row('b.mp3', 200.0, 40.0, 80.0, 60.0, 200.0, 120.0, 6.0),
    ]

    summary = summarize(rows)

    assert summary['conversations'] == 2
    assert summary['metrics']['request_core_ms']['mean'] == 150.0
    assert summary['stages']['asr_total_ms']['mean'] == 30.0
    assert summary['slowest_conversations'][0]['audio_filename'] == 'b.mp3'
    assert summary['correlations']['audio_duration_vs_asr_ms'] == 1.0
