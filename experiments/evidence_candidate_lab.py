#!/usr/bin/env python3
"""Offline multi-candidate evidence discovery from cached diagnostics.

This script never runs Whisper and never calls /predict. It reconstructs the
word-numbered transcript from diagnostics JSON, asks Ollama to find multiple
independent supporting occurrences, and scores candidate recall against gold.

Important experimental invariant: the candidate-finder prompt never receives
Pass-1 evidence or gold timestamps. Gold is used only after inference to score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx


PROMPT_VERSION = 'candidate-discovery-v1'
DEFAULT_THRESHOLDS = (0.50, 0.75, 0.90)

SYSTEM_PROMPT = '''You are an evidence retrieval system for doctor-patient transcripts.

An upstream classifier has already decided that each supplied proposition is supported. Your ONLY task is to independently find the distinct transcript passages that could serve as evidence. Do not classify the proposition and do not assume any previous evidence choice.

The transcript contains segment labels S0, S1, ... and every timestamped word is prefixed by a global word ID such as 142:word.

For each question:
- Search the ENTIRE transcript from beginning to end.
- Return up to the requested number of DISTINCT supporting occurrences.
- If the same fact is stated, repeated, confirmed, summarized, or acted on at different times, return those as separate candidates when each can independently support the proposition.
- Do not waste candidate slots on boundary variants of the same occurrence.
- Each candidate must be one contiguous word range.
- Use the shortest COMPLETE passage that establishes the whole proposition, not merely a keyword.
- Prefer self-contained ranges. Include an antecedent or nearby words when a pronoun or shorthand would otherwise make the support ambiguous.
- A candidate may cross at most one adjacent Whisper segment boundary.
- A candidate may contain at most the configured maximum number of words.
- Rank candidates by how directly, explicitly, and completely they establish the proposition, but still include plausible alternative occurrences lower in the list.

Return word IDs only. Output JSON matching the supplied schema exactly.'''


def normalize_question(text: str) -> str:
    return ' '.join(text.split())


def transcript_id_from_audio(audio_filename: str) -> str:
    stem = Path(audio_filename).stem
    return stem[len('conversation_'):] if stem.startswith('conversation_') else stem


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    return float(text) if text else None


def bool_from_gold(row: dict[str, str]) -> bool:
    for key in ('label', 'answer'):
        value = str(row.get(key, '')).strip().lower()
        if value in {'1', 'true', 'yes'}:
            return True
        if value in {'0', 'false', 'no'}:
            return False
    question_type = str(row.get('question_type', '')).strip().lower()
    if question_type == 'positive':
        return True
    if question_type in {'hard_negative', 'off_topic'}:
        return False
    raise ValueError(f'Cannot determine gold answer from row: {row}')


def tiou(
    pred_start: float | None,
    pred_end: float | None,
    gold_start: float | None,
    gold_end: float | None,
) -> float:
    if None in {pred_start, pred_end, gold_start, gold_end}:
        return 0.0
    assert pred_start is not None and pred_end is not None
    assert gold_start is not None and gold_end is not None
    if pred_end < pred_start or gold_end < gold_start:
        return 0.0
    overlap = max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    return overlap / union if union > 0 else 0.0


def load_gold_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {
            'transcript_id', 'question', 'question_type',
            'evidence_start', 'evidence_end',
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path} is missing required columns: {sorted(missing)}')
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            parsed['_gold_answer'] = bool_from_gold(row)
            parsed['_gold_start'] = float_or_none(row.get('evidence_start'))
            parsed['_gold_end'] = float_or_none(row.get('evidence_end'))
            grouped[str(row['transcript_id']).strip()].append(parsed)
    return grouped


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob('conversation_*.json'))
    if not files:
        raise FileNotFoundError(f'No conversation_*.json files found in {path}')
    payloads = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding='utf-8'))
        schema = int(payload.get('schema_version', 0))
        if schema not in {1, 2, 3}:
            raise ValueError(f'Unsupported diagnostics schema {schema} in {file_path}')
        payloads.append(payload)
    return payloads


def match_gold_questions(
    payload: dict[str, Any],
    gold_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_text: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for gold in gold_rows:
        by_text[normalize_question(str(gold['question']))].append(gold)
    matched = []
    for prediction in payload['questions']:
        key = normalize_question(str(prediction['question']))
        if not by_text[key]:
            raise ValueError(
                f"Could not match {prediction['question']!r} in "
                f"{payload.get('audio_filename')}"
            )
        matched.append((prediction, by_text[key].popleft()))
    return matched


def first_pass_span(prediction: dict[str, Any]) -> tuple[float | None, float | None]:
    if 'first_pass_resolved_evidence_start' in prediction:
        return (
            float_or_none(prediction.get('first_pass_resolved_evidence_start')),
            float_or_none(prediction.get('first_pass_resolved_evidence_end')),
        )
    return (
        float_or_none(prediction.get('resolved_evidence_start')),
        float_or_none(prediction.get('resolved_evidence_end')),
    )


def build_examples(
    payloads: list[dict[str, Any]],
    gold: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    examples = []
    payload_by_transcript = {}
    for payload in payloads:
        transcript_id = transcript_id_from_audio(str(payload['audio_filename']))
        if transcript_id not in gold:
            raise KeyError(f'No gold rows for {transcript_id}')
        payload_by_transcript[transcript_id] = payload
        for prediction, gold_row in match_gold_questions(payload, gold[transcript_id]):
            if not bool(gold_row['_gold_answer']):
                continue
            gold_start = gold_row['_gold_start']
            gold_end = gold_row['_gold_end']
            if gold_start is None or gold_end is None:
                continue
            baseline_start, baseline_end = first_pass_span(prediction)
            examples.append({
                'transcript_id': transcript_id,
                'audio_filename': payload['audio_filename'],
                'question_index': int(prediction['question_index']),
                'question_id': str(gold_row.get('question_id', '')),
                'question': str(prediction['question']),
                'predicted_answer': bool(prediction.get('answer')),
                'gold_start': float(gold_start),
                'gold_end': float(gold_end),
                'baseline_start': baseline_start,
                'baseline_end': baseline_end,
                'baseline_tiou': tiou(
                    baseline_start, baseline_end, gold_start, gold_end
                ),
                'baseline_start_word_id': prediction.get('evidence_start_word_id'),
                'baseline_end_word_id': prediction.get('evidence_end_word_id'),
                'baseline_text': prediction.get('selected_word_text', ''),
            })
    return examples, payload_by_transcript


def choose_cohort(
    examples: list[dict[str, Any]],
    cohort: str,
    max_current_tiou: float,
    control_min_tiou: float,
    include_classification_misses: bool,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in examples
        if include_classification_misses or row['predicted_answer']
    ]
    if cohort == 'zero':
        return [row for row in eligible if row['baseline_tiou'] <= 1e-12]
    if cohort == 'low':
        return [row for row in eligible if row['baseline_tiou'] <= max_current_tiou]
    if cohort == 'control':
        return [row for row in eligible if row['baseline_tiou'] >= control_min_tiou]
    if cohort == 'all':
        return eligible
    raise ValueError(f'Unknown cohort: {cohort}')


def render_cached_transcript(payload: dict[str, Any]) -> str:
    words_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for word in sorted(payload['words'], key=lambda item: int(item['id'])):
        words_by_segment[int(word['segment_id'])].append(word)
    lines = []
    for segment in sorted(payload['segments'], key=lambda item: int(item['id'])):
        sid = int(segment['id'])
        lines.append(
            f"[S{sid} {float(segment['start']):.2f}-{float(segment['end']):.2f}]"
        )
        addressed = ' '.join(
            f"{int(word['id'])}:{word['text']}"
            for word in words_by_segment.get(sid, [])
        )
        lines.append(addressed if addressed else str(segment.get('text', '')))
    return '\n'.join(lines)


def candidate_schema(question_count: int, candidate_count: int) -> dict[str, Any]:
    candidate = {
        'type': 'object',
        'properties': {
            'start_word_id': {'type': 'integer', 'minimum': 0},
            'end_word_id': {'type': 'integer', 'minimum': 0},
        },
        'required': ['start_word_id', 'end_word_id'],
        'additionalProperties': False,
    }
    per_question = {
        'type': 'object',
        'properties': {
            'candidates': {
                'type': 'array',
                'items': candidate,
                'minItems': 0,
                'maxItems': candidate_count,
            }
        },
        'required': ['candidates'],
        'additionalProperties': False,
    }
    properties = {str(i): per_question for i in range(question_count)}
    return {
        'type': 'object',
        'properties': properties,
        'required': list(properties),
        'additionalProperties': False,
    }


def validate_candidate(
    raw: dict[str, Any],
    payload: dict[str, Any],
    max_words: int,
) -> dict[str, Any] | None:
    try:
        start_id = int(raw['start_word_id'])
        end_id = int(raw['end_word_id'])
    except (KeyError, TypeError, ValueError):
        return None
    if start_id < 0 or end_id < start_id or end_id - start_id + 1 > max_words:
        return None

    words = sorted(payload['words'], key=lambda item: int(item['id']))
    by_id = {int(word['id']): word for word in words}
    ids = list(range(start_id, end_id + 1))
    if any(word_id not in by_id for word_id in ids):
        return None
    selected = [by_id[word_id] for word_id in ids]

    segment_ids = []
    for word in selected:
        sid = int(word['segment_id'])
        if not segment_ids or segment_ids[-1] != sid:
            segment_ids.append(sid)
    if len(segment_ids) > 2:
        return None
    if len(segment_ids) == 2 and abs(segment_ids[1] - segment_ids[0]) != 1:
        return None

    return {
        'start_word_id': start_id,
        'end_word_id': end_id,
        'start': float(selected[0]['start']),
        'end': float(selected[-1]['end']),
        'segment_ids': segment_ids,
        'text': ' '.join(str(word['text']).strip() for word in selected).strip(),
    }


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for candidate in candidates:
        key = (candidate['start_word_id'], candidate['end_word_id'])
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def build_user_prompt(
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
    candidate_count: int,
    max_words: int,
) -> str:
    transcript = render_cached_transcript(payload)
    numbered = '\n'.join(
        f"{i}. {row['question']}" for i, row in enumerate(questions)
    )
    keys = ', '.join(str(i) for i in range(len(questions)))
    return f'''TRANSCRIPT:
{transcript}

QUESTIONS:
{numbered}

For EACH question, independently search the full transcript and return up to {candidate_count} distinct supporting occurrences.

Each occurrence must be a contiguous range of at most {max_words} words and may cross at most one adjacent segment boundary. Do not return multiple boundary variants of the same occurrence.

Return a JSON object with exactly these keys: {keys}'''


def ollama_discover(
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
    candidate_count: int,
    max_words: int,
) -> list[list[dict[str, Any]]]:
    request_payload = {
        'model': model,
        'stream': False,
        'think': False,
        'keep_alive': keep_alive,
        'format': candidate_schema(len(questions), candidate_count),
        'options': {
            'temperature': 0,
            'num_predict': num_predict,
            'num_ctx': num_ctx,
        },
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': build_user_prompt(
                    payload, questions, candidate_count, max_words
                ),
            },
        ],
    }

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout, connect=min(5.0, timeout))
            ) as client:
                response = client.post(
                    f"{ollama_url.rstrip('/')}/api/chat",
                    json=request_payload,
                )
                response.raise_for_status()
                response_json = response.json()
            raw = json.loads(response_json['message']['content'])
            expected = {str(i) for i in range(len(questions))}
            if set(raw) != expected:
                raise ValueError(
                    f'LLM keys {sorted(raw)} != expected {sorted(expected)}'
                )

            result = []
            for i in range(len(questions)):
                validated = [
                    candidate
                    for item in raw[str(i)].get('candidates', [])
                    if (
                        candidate := validate_candidate(
                            item, payload, max_words
                        )
                    ) is not None
                ]
                result.append(
                    dedupe_candidates(validated)[:candidate_count]
                )
            return result
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.2)

    raise RuntimeError(
        f'Candidate discovery failed twice: {last_error}'
    ) from last_error


def cache_key(
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
    model: str,
    candidate_count: int,
    max_words: int,
) -> str:
    material = {
        'prompt_version': PROMPT_VERSION,
        'model': model,
        'candidate_count': candidate_count,
        'max_words': max_words,
        'audio_filename': payload['audio_filename'],
        'words': [
            [
                int(word['id']), str(word['text']),
                float(word['start']), float(word['end']),
                int(word['segment_id']),
            ]
            for word in payload['words']
        ],
        'questions': [
            [row['question_id'], row['question']] for row in questions
        ],
    }
    blob = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    )
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def discover_with_cache(
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
    cache_dir: Path,
    refresh: bool,
    **kwargs: Any,
) -> tuple[list[list[dict[str, Any]]], bool, Path]:
    key = cache_key(
        payload,
        questions,
        kwargs['model'],
        kwargs['candidate_count'],
        kwargs['max_words'],
    )
    transcript_id = transcript_id_from_audio(str(payload['audio_filename']))
    cache_path = cache_dir / f'{transcript_id}__{key}.json'
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        return cached['candidates'], True, cache_path

    candidates = ollama_discover(payload, questions, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                'prompt_version': PROMPT_VERSION,
                'model': kwargs['model'],
                'audio_filename': payload['audio_filename'],
                'question_ids': [row['question_id'] for row in questions],
                'questions': [row['question'] for row in questions],
                'candidates': candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return candidates, False, cache_path


def score_candidates(
    example: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item['rank'] = rank
        item['tiou'] = tiou(
            candidate['start'],
            candidate['end'],
            example['gold_start'],
            example['gold_end'],
        )
        scored.append(item)
    best = max(scored, key=lambda item: item['tiou'], default=None)
    return {
        **example,
        'candidates': scored,
        'candidate_count': len(scored),
        'top1_tiou': float(scored[0]['tiou']) if scored else 0.0,
        'best_candidate_tiou': float(best['tiou']) if best else 0.0,
        'best_candidate_rank': int(best['rank']) if best else None,
    }


def recall_at_k(
    rows: list[dict[str, Any]],
    k: int,
    threshold: float,
) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        best = max(
            (float(c['tiou']) for c in row['candidates'][:k]),
            default=0.0,
        )
        hits += best >= threshold
    return hits / len(rows)


def summarize_results(
    selected_rows: list[dict[str, Any]],
    all_positive_examples: list[dict[str, Any]],
    candidate_count: int,
    recall_threshold: float,
    model_calls: int,
    cache_hits: int,
) -> dict[str, Any]:
    selected_by_key = {
        (row['transcript_id'], row['question_id'], row['question']): row
        for row in selected_rows
    }
    baseline_all = [
        float(row['baseline_tiou']) for row in all_positive_examples
    ]
    top1_all = []
    oracle_all = []
    for row in all_positive_examples:
        key = (row['transcript_id'], row['question_id'], row['question'])
        selected = selected_by_key.get(key)
        if selected is None:
            top1_all.append(float(row['baseline_tiou']))
            oracle_all.append(float(row['baseline_tiou']))
        else:
            top1_all.append(float(selected['top1_tiou']))
            oracle_all.append(float(selected['best_candidate_tiou']))

    thresholds = sorted(set(DEFAULT_THRESHOLDS + (recall_threshold,)))
    recall = {
        f'{threshold:.2f}': {
            f'@{k}': recall_at_k(selected_rows, k, threshold)
            for k in range(1, candidate_count + 1)
        }
        for threshold in thresholds
    }

    def mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    baseline_mean = mean(baseline_all)
    top1_mean = mean(top1_all)
    oracle_mean = mean(oracle_all)
    return {
        'prompt_version': PROMPT_VERSION,
        'selected_questions': len(selected_rows),
        'selected_conversations': len({
            row['transcript_id'] for row in selected_rows
        }),
        'model_calls': model_calls,
        'cache_hits': cache_hits,
        'candidate_count_limit': candidate_count,
        'recall_threshold': recall_threshold,
        'baseline_mean_tiou_selected': mean([
            float(row['baseline_tiou']) for row in selected_rows
        ]),
        'top1_mean_tiou_selected': mean([
            float(row['top1_tiou']) for row in selected_rows
        ]),
        'candidate_oracle_mean_tiou_selected': mean([
            float(row['best_candidate_tiou']) for row in selected_rows
        ]),
        'no_valid_candidate_questions': sum(
            row['candidate_count'] == 0 for row in selected_rows
        ),
        'top1_improved': sum(
            row['top1_tiou'] > row['baseline_tiou'] + 1e-12
            for row in selected_rows
        ),
        'top1_degraded': sum(
            row['top1_tiou'] + 1e-12 < row['baseline_tiou']
            for row in selected_rows
        ),
        'top1_unchanged': sum(
            abs(row['top1_tiou'] - row['baseline_tiou']) <= 1e-12
            for row in selected_rows
        ),
        'candidate_recall': recall,
        'all_positive_count': len(all_positive_examples),
        'all_positive_baseline_mean_tiou': baseline_mean,
        'all_positive_mean_tiou_if_top1_replaces_selected': top1_mean,
        'all_positive_mean_tiou_if_best_returned_candidate_replaces_selected':
            oracle_mean,
        'all_positive_top1_delta': top1_mean - baseline_mean,
        'all_positive_returned_candidate_oracle_delta':
            oracle_mean - baseline_mean,
    }


def write_rows_csv(
    rows: list[dict[str, Any]],
    path: Path,
    candidate_count: int,
) -> None:
    fields = [
        'transcript_id', 'question_id', 'question_index', 'question',
        'baseline_tiou', 'baseline_start', 'baseline_end',
        'baseline_start_word_id', 'baseline_end_word_id', 'baseline_text',
        'gold_start', 'gold_end', 'candidate_count', 'top1_tiou',
        'best_candidate_tiou', 'best_candidate_rank',
    ]
    for i in range(1, candidate_count + 1):
        fields.extend([
            f'candidate_{i}_start_word_id',
            f'candidate_{i}_end_word_id',
            f'candidate_{i}_start',
            f'candidate_{i}_end',
            f'candidate_{i}_tiou',
            f'candidate_{i}_text',
        ])

    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fields}
            for i, candidate in enumerate(
                row['candidates'][:candidate_count], start=1
            ):
                for field in (
                    'start_word_id', 'end_word_id', 'start',
                    'end', 'tiou', 'text',
                ):
                    output[f'candidate_{i}_{field}'] = candidate[field]
            writer.writerow(output)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (-row['best_candidate_tiou'], row['question_id']),
    )
    lines = ['# Evidence candidate discovery', '']
    for row in ordered:
        lines.extend([
            f"## {row['question_id'] or row['transcript_id']}",
            '',
            f"**Question:** {row['question']}",
            '',
            f"**Baseline:** tIoU={row['baseline_tiou']:.3f} | "
            f"{row['baseline_text'] or '(no span)'}",
            '',
            f"**Gold:** {row['gold_start']:.3f}–{row['gold_end']:.3f}s",
            '',
        ])
        if not row['candidates']:
            lines.extend(['No valid candidates returned.', ''])
            continue
        for candidate in row['candidates']:
            lines.extend([
                f"**Candidate {candidate['rank']}** — "
                f"tIoU={candidate['tiou']:.3f}, "
                f"W{candidate['start_word_id']}–W{candidate['end_word_id']} "
                f"({candidate['start']:.3f}–{candidate['end']:.3f}s)",
                '',
                candidate['text'] or '(empty)',
                '',
            ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def print_summary(summary: dict[str, Any]) -> None:
    print('Offline evidence candidate discovery')
    print(f"  selected questions                 {summary['selected_questions']}")
    print(f"  selected conversations             {summary['selected_conversations']}")
    print(f"  model calls                         {summary['model_calls']}")
    print(f"  cache hits                          {summary['cache_hits']}")
    print()
    print('Selected cohort')
    print(
        f"  baseline mean tIoU                  "
        f"{summary['baseline_mean_tiou_selected']:.3f}"
    )
    print(
        f"  top-1 candidate mean tIoU           "
        f"{summary['top1_mean_tiou_selected']:.3f}"
    )
    print(
        f"  returned-candidate oracle mean      "
        f"{summary['candidate_oracle_mean_tiou_selected']:.3f}"
    )
    print(
        f"  no valid candidates                 "
        f"{summary['no_valid_candidate_questions']}"
    )
    print(
        f"  top-1 improved/degraded/unchanged   "
        f"{summary['top1_improved']}/{summary['top1_degraded']}/"
        f"{summary['top1_unchanged']}"
    )
    print()
    threshold_key = f"{summary['recall_threshold']:.2f}"
    print(
        f"Candidate recall "
        f"(best tIoU >= {summary['recall_threshold']:.2f})"
    )
    for key, value in summary['candidate_recall'][threshold_key].items():
        print(f"  recall{key:>3s}                          {value:.3f}")
    print()
    print('Whole positive set, replacing only selected cohort')
    print(
        f"  baseline mean tIoU                  "
        f"{summary['all_positive_baseline_mean_tiou']:.3f}"
    )
    print(
        f"  top-1 replacement mean              "
        f"{summary['all_positive_mean_tiou_if_top1_replaces_selected']:.3f} "
        f"({summary['all_positive_top1_delta']:+.3f})"
    )
    print(
        f"  returned-candidate oracle mean      "
        f"{summary['all_positive_mean_tiou_if_best_returned_candidate_replaces_selected']:.3f} "
        f"({summary['all_positive_returned_candidate_oracle_delta']:+.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Offline multi-candidate evidence discovery from cached diagnostics.'
        )
    )
    parser.add_argument('--diagnostics-dir', type=Path, required=True)
    parser.add_argument('--question-csv', type=Path, required=True)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.diagnostics/evidence-candidate-lab'),
    )
    parser.add_argument(
        '--cohort',
        choices=['zero', 'low', 'control', 'all'],
        default='zero',
    )
    parser.add_argument('--max-current-tiou', type=float, default=0.25)
    parser.add_argument('--control-min-tiou', type=float, default=0.75)
    parser.add_argument('--include-classification-misses', action='store_true')
    parser.add_argument('--candidate-count', type=int, default=4)
    parser.add_argument('--max-words', type=int, default=80)
    parser.add_argument('--recall-threshold', type=float, default=0.75)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--ollama-url',
        default=os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434'),
    )
    parser.add_argument(
        '--model',
        default=os.getenv('OLLAMA_MODEL', 'qwen3.5:27b'),
    )
    parser.add_argument('--timeout', type=float, default=90.0)
    parser.add_argument(
        '--keep-alive',
        default=os.getenv('OLLAMA_KEEP_ALIVE', '30m'),
    )
    parser.add_argument(
        '--num-ctx',
        type=int,
        default=int(os.getenv('OLLAMA_NUM_CTX', '8192')),
    )
    parser.add_argument('--num-predict', type=int, default=1800)
    args = parser.parse_args()

    if args.candidate_count < 1:
        raise ValueError('--candidate-count must be >= 1')
    if args.max_words < 1:
        raise ValueError('--max-words must be >= 1')
    if not 0 <= args.recall_threshold <= 1:
        raise ValueError('--recall-threshold must be between 0 and 1')

    gold = load_gold_rows(args.question_csv)
    payloads = load_diagnostics(args.diagnostics_dir)
    examples, payload_by_transcript = build_examples(payloads, gold)
    selected = choose_cohort(
        examples,
        args.cohort,
        args.max_current_tiou,
        args.control_min_tiou,
        args.include_classification_misses,
    )
    selected.sort(
        key=lambda row: (
            row['baseline_tiou'], row['transcript_id'], row['question_id']
        )
    )
    if args.limit is not None:
        selected = selected[:max(0, args.limit)]

    print(
        f"Selected {len(selected)} positive questions from "
        f"{len({row['transcript_id'] for row in selected})} conversations "
        f"(cohort={args.cohort})."
    )
    if args.dry_run:
        for row in selected:
            print(
                f"  {row['question_id']:<28s} "
                f"tIoU={row['baseline_tiou']:.3f}  {row['question']}"
            )
        return
    if not selected:
        raise ValueError('Selected cohort is empty')

    run_dir = args.output_dir / args.cohort
    cache_dir = args.output_dir / 'cache'
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[row['transcript_id']].append(row)

    scored_rows = []
    model_calls = 0
    cache_hits = 0
    for index, transcript_id in enumerate(sorted(grouped), start=1):
        questions = grouped[transcript_id]
        payload = payload_by_transcript[transcript_id]
        candidates, from_cache, cache_path = discover_with_cache(
            payload,
            questions,
            cache_dir,
            args.refresh,
            ollama_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
            keep_alive=args.keep_alive,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            candidate_count=args.candidate_count,
            max_words=args.max_words,
        )
        cache_hits += int(from_cache)
        model_calls += int(not from_cache)
        print(
            f"[{index}/{len(grouped)}] {transcript_id}: "
            f"{len(questions)} questions "
            f"({'cache' if from_cache else 'model'}) -> {cache_path.name}"
        )
        for example, question_candidates in zip(
            questions, candidates, strict=True
        ):
            scored_rows.append(
                score_candidates(example, question_candidates)
            )

    scored_rows.sort(
        key=lambda row: (
            row['baseline_tiou'], row['transcript_id'], row['question_id']
        )
    )
    summary = summarize_results(
        scored_rows,
        examples,
        args.candidate_count,
        args.recall_threshold,
        model_calls,
        cache_hits,
    )
    summary.update({
        'cohort': args.cohort,
        'max_current_tiou': args.max_current_tiou,
        'control_min_tiou': args.control_min_tiou,
        'include_classification_misses': args.include_classification_misses,
        'model': args.model,
        'diagnostics_dir': str(args.diagnostics_dir),
        'question_csv': str(args.question_csv),
    })

    summary_path = run_dir / 'summary.json'
    rows_path = run_dir / 'candidate_results.csv'
    markdown_path = run_dir / 'candidate_results.md'
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    write_rows_csv(scored_rows, rows_path, args.candidate_count)
    write_markdown(scored_rows, markdown_path)

    print()
    print_summary(summary)
    print()
    print(f'Wrote {summary_path}')
    print(f'Wrote {rows_path}')
    print(f'Wrote {markdown_path}')


if __name__ == '__main__':
    main()
