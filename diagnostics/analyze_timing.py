#!/usr/bin/env python3
"""Aggregate performance timing from diagnostics schema v3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            'n': 0,
            'mean': None,
            'p50': None,
            'p90': None,
            'max': None,
        }
    return {
        'n': len(clean),
        'mean': statistics.mean(clean),
        'p50': statistics.median(clean),
        'p90': _percentile(clean, 0.90),
        'max': max(clean),
    }


def _corr(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None

    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in pairs
    )
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in x_values)
        * sum((y - mean_y) ** 2 for y in y_values)
    )
    if denominator == 0:
        return None
    return numerator / denominator


def _successful_attempt(stage: dict[str, Any]) -> dict[str, Any]:
    attempts = stage.get('attempts') or []
    for attempt in reversed(attempts):
        if attempt.get('success'):
            return attempt
    return {}


def load_rows(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 3:
            continue

        timing = payload.get('timing') or {}
        asr = timing.get('asr') or {}
        pass1 = timing.get('pass1') or {}
        pass2 = timing.get('pass2') or {}
        pass1_attempt = _successful_attempt(pass1)
        pass2_attempt = _successful_attempt(pass2)
        pass1_ollama = pass1.get('ollama') or pass1_attempt.get('ollama') or {}
        pass2_ollama = pass2.get('ollama') or pass2_attempt.get('ollama') or {}

        row = {
            'audio_filename': payload.get('audio_filename', path.stem),
            'request_core_ms': _number(timing.get('request_core_ms')),
            'base64_decode_ms': _number(timing.get('base64_decode_ms')),
            'asr_total_ms': _number(asr.get('total_ms')),
            'asr_whisper_ms': _number(asr.get('whisper_materialize_ms')),
            'audio_duration_s': _number(asr.get('audio_duration_s')),
            'asr_realtime_factor': _number(asr.get('realtime_factor')),
            'segment_count': _number(asr.get('segment_count')),
            'word_count': _number(asr.get('word_count')),
            'pass1_total_ms': _number(pass1.get('total_wall_ms')),
            'pass1_prompt_build_ms': _number(pass1.get('prompt_build_ms')),
            'pass1_http_ms': _number(pass1.get('http_wall_ms')),
            'pass1_parse_ms': _number(pass1.get('parse_validate_ms')),
            'pass1_ollama_total_ms': _number(pass1_ollama.get('total_ms')),
            'pass1_load_ms': _number(pass1_ollama.get('load_ms')),
            'pass1_prompt_eval_ms': _number(pass1_ollama.get('prompt_eval_ms')),
            'pass1_eval_ms': _number(pass1_ollama.get('eval_ms')),
            'pass1_prompt_tokens': _number(pass1_ollama.get('prompt_tokens')),
            'pass1_output_tokens': _number(pass1_ollama.get('output_tokens')),
            'pass1_prompt_tokens_per_s': _number(
                pass1_ollama.get('prompt_tokens_per_s')
            ),
            'pass1_output_tokens_per_s': _number(
                pass1_ollama.get('output_tokens_per_s')
            ),
            'pass1_retry_count': _number(pass1.get('retry_count')) or 0.0,
            'pass2_enabled': bool(pass2.get('enabled', False)),
            'true_question_count': _number(pass2.get('true_question_count')),
            'pass2_total_ms': _number(pass2.get('total_wall_ms')),
            'pass2_prompt_build_ms': _number(pass2.get('prompt_build_ms')),
            'pass2_http_ms': _number(pass2.get('http_wall_ms')),
            'pass2_parse_ms': _number(pass2.get('parse_validate_ms')),
            'pass2_ollama_total_ms': _number(pass2_ollama.get('total_ms')),
            'pass2_load_ms': _number(pass2_ollama.get('load_ms')),
            'pass2_prompt_eval_ms': _number(pass2_ollama.get('prompt_eval_ms')),
            'pass2_eval_ms': _number(pass2_ollama.get('eval_ms')),
            'pass2_prompt_tokens': _number(pass2_ollama.get('prompt_tokens')),
            'pass2_output_tokens': _number(pass2_ollama.get('output_tokens')),
            'pass2_prompt_tokens_per_s': _number(
                pass2_ollama.get('prompt_tokens_per_s')
            ),
            'pass2_output_tokens_per_s': _number(
                pass2_ollama.get('output_tokens_per_s')
            ),
            'pass2_retry_count': _number(pass2.get('retry_count')) or 0.0,
            'evidence_resolution_ms': _number(
                timing.get('evidence_resolution_ms')
            ),
            'unaccounted_ms': _number(timing.get('unaccounted_ms')),
            'diagnostic_payload_build_ms': _number(
                _path(timing, 'diagnostics', 'payload_build_ms')
            ),
            'diagnostic_json_serialize_ms': _number(
                _path(timing, 'diagnostics', 'json_serialize_ms')
            ),
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f'No diagnostics schema v3 JSON files found in {directory}'
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if row.get(key) is not None
        ]

    request_mean = statistics.mean(values('request_core_ms'))

    stage_keys = [
        'asr_total_ms',
        'pass1_total_ms',
        'pass2_total_ms',
        'evidence_resolution_ms',
        'unaccounted_ms',
    ]
    stages = {}
    for key in stage_keys:
        stage_stats = _stats(values(key))
        stage_mean = stage_stats['mean']
        stage_stats['percent_of_request_mean'] = (
            100.0 * float(stage_mean) / request_mean
            if stage_mean is not None and request_mean > 0
            else None
        )
        stages[key] = stage_stats

    metrics = {}
    for key in [
        'request_core_ms',
        'base64_decode_ms',
        'asr_whisper_ms',
        'audio_duration_s',
        'asr_realtime_factor',
        'word_count',
        'pass1_prompt_eval_ms',
        'pass1_eval_ms',
        'pass1_prompt_tokens',
        'pass1_output_tokens',
        'pass1_prompt_tokens_per_s',
        'pass1_output_tokens_per_s',
        'pass2_prompt_eval_ms',
        'pass2_eval_ms',
        'pass2_prompt_tokens',
        'pass2_output_tokens',
        'pass2_prompt_tokens_per_s',
        'pass2_output_tokens_per_s',
        'diagnostic_payload_build_ms',
        'diagnostic_json_serialize_ms',
    ]:
        metrics[key] = _stats(values(key))

    correlations = {
        'audio_duration_vs_asr_ms': _corr(
            values('audio_duration_s'),
            values('asr_total_ms'),
        ),
        'word_count_vs_pass1_prompt_eval_ms': _corr(
            values('word_count'),
            values('pass1_prompt_eval_ms'),
        ),
        'word_count_vs_pass2_prompt_eval_ms': _corr(
            values('word_count'),
            values('pass2_prompt_eval_ms'),
        ),
        'true_questions_vs_pass2_eval_ms': _corr(
            values('true_question_count'),
            values('pass2_eval_ms'),
        ),
        'pass1_output_tokens_vs_eval_ms': _corr(
            values('pass1_output_tokens'),
            values('pass1_eval_ms'),
        ),
        'pass2_output_tokens_vs_eval_ms': _corr(
            values('pass2_output_tokens'),
            values('pass2_eval_ms'),
        ),
    }

    retry_summary = {
        'pass1_total_retries': int(sum(values('pass1_retry_count'))),
        'pass2_total_retries': int(sum(values('pass2_retry_count'))),
        'conversations_with_pass1_retry': sum(
            float(row.get('pass1_retry_count') or 0) > 0
            for row in rows
        ),
        'conversations_with_pass2_retry': sum(
            float(row.get('pass2_retry_count') or 0) > 0
            for row in rows
        ),
    }

    slowest = sorted(
        rows,
        key=lambda row: float(row.get('request_core_ms') or 0.0),
        reverse=True,
    )[:10]

    return {
        'conversations': len(rows),
        'stages': stages,
        'metrics': metrics,
        'retries': retry_summary,
        'correlations': correlations,
        'slowest_conversations': slowest,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return '-'
    return f'{float(value):.{digits}f}'


def print_summary(summary: dict[str, Any]) -> None:
    print('Performance timing diagnostics')
    print(f"  conversations: {summary['conversations']}")
    print()
    print('Stage timing (ms)')
    print(
        f"{'stage':28s} {'mean':>10s} {'p50':>10s} "
        f"{'p90':>10s} {'max':>10s} {'% total':>9s}"
    )
    print('-' * 82)

    request_stats = summary['metrics']['request_core_ms']
    print(
        f"{'request_core':28s} "
        f"{_fmt(request_stats['mean']):>10s} "
        f"{_fmt(request_stats['p50']):>10s} "
        f"{_fmt(request_stats['p90']):>10s} "
        f"{_fmt(request_stats['max']):>10s} "
        f"{'100.0':>9s}"
    )

    labels = {
        'asr_total_ms': 'ASR',
        'pass1_total_ms': 'Pass 1',
        'pass2_total_ms': 'Pass 2',
        'evidence_resolution_ms': 'Evidence resolution',
        'unaccounted_ms': 'Unaccounted',
    }
    for key, label in labels.items():
        stats = summary['stages'][key]
        print(
            f"{label:28s} "
            f"{_fmt(stats['mean']):>10s} "
            f"{_fmt(stats['p50']):>10s} "
            f"{_fmt(stats['p90']):>10s} "
            f"{_fmt(stats['max']):>10s} "
            f"{_fmt(stats['percent_of_request_mean']):>9s}"
        )

    print()
    print('Ollama throughput')
    for pass_name in ('pass1', 'pass2'):
        prompt_stats = summary['metrics'][
            f'{pass_name}_prompt_tokens_per_s'
        ]
        output_stats = summary['metrics'][
            f'{pass_name}_output_tokens_per_s'
        ]
        print(
            f"  {pass_name} prompt tok/s: "
            f"mean {_fmt(prompt_stats['mean'])}, "
            f"p50 {_fmt(prompt_stats['p50'])}, "
            f"p90 {_fmt(prompt_stats['p90'])}"
        )
        print(
            f"  {pass_name} output tok/s: "
            f"mean {_fmt(output_stats['mean'])}, "
            f"p50 {_fmt(output_stats['p50'])}, "
            f"p90 {_fmt(output_stats['p90'])}"
        )

    retries = summary['retries']
    print()
    print('Retries')
    print(
        f"  Pass 1: {retries['pass1_total_retries']} total retries "
        f"across {retries['conversations_with_pass1_retry']} conversations"
    )
    print(
        f"  Pass 2: {retries['pass2_total_retries']} total retries "
        f"across {retries['conversations_with_pass2_retry']} conversations"
    )

    print()
    print('Correlations')
    for key, value in summary['correlations'].items():
        print(f"  {key:42s} {_fmt(value, 3)}")

    print()
    print('10 slowest conversations')
    print(
        f"{'conversation':28s} {'total':>8s} {'ASR':>8s} "
        f"{'P1 prm':>8s} {'P1 gen':>8s} {'P2 prm':>8s} "
        f"{'P2 gen':>8s} {'TRUE':>5s}"
    )
    for row in summary['slowest_conversations']:
        print(
            f"{str(row['audio_filename'])[:28]:28s} "
            f"{_fmt(row['request_core_ms']):>8s} "
            f"{_fmt(row['asr_total_ms']):>8s} "
            f"{_fmt(row['pass1_prompt_eval_ms']):>8s} "
            f"{_fmt(row['pass1_eval_ms']):>8s} "
            f"{_fmt(row['pass2_prompt_eval_ms']):>8s} "
            f"{_fmt(row['pass2_eval_ms']):>8s} "
            f"{_fmt(row['true_question_count'], 0):>5s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diagnostics-dir',
        type=Path,
        default=Path('.diagnostics/evidence-refinement-v2'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.diagnostics_dir / 'analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.diagnostics_dir)
    summary = summarize(rows)

    summary_path = output_dir / 'timing_summary.json'
    csv_path = output_dir / 'timing_by_conversation.csv'

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    write_csv(rows, csv_path)

    print_summary(summary)
    print()
    print(f'Wrote {summary_path}')
    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    main()
