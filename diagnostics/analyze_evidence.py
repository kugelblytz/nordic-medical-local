#!/usr/bin/env python3
"""Analyze word-boundary evidence predictions against the supplied training CSV.

This script consumes the per-conversation JSON files written by
DIAGNOSTICS_ENABLED=true and produces:

- summary.json
- evidence_diagnostics.csv
- worst_cases.md

It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _bool_from_gold(row: dict[str, str]) -> bool:
    label = str(row.get('label', '')).strip().lower()
    if label in {'1', 'true', 'yes'}:
        return True
    if label in {'0', 'false', 'no'}:
        return False

    answer = str(row.get('answer', '')).strip().lower()
    if answer in {'yes', 'true', '1'}:
        return True
    if answer in {'no', 'false', '0'}:
        return False

    raise ValueError(f'Cannot determine gold label from row: {row}')


def _normalize_question(text: str) -> str:
    return ' '.join(text.split())


def _transcript_id(audio_filename: str) -> str:
    stem = Path(audio_filename).stem
    if stem.startswith('conversation_'):
        return stem[len('conversation_'):]
    return stem


def tiou(
    pred_start: float | None,
    pred_end: float | None,
    gold_start: float | None,
    gold_end: float | None,
) -> float:
    if None in {pred_start, pred_end, gold_start, gold_end}:
        return 0.0

    assert pred_start is not None
    assert pred_end is not None
    assert gold_start is not None
    assert gold_end is not None

    if pred_end < pred_start or gold_end < gold_start:
        return 0.0

    overlap = max(
        0.0,
        min(pred_end, gold_end) - max(pred_start, gold_start),
    )
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    if union <= 0.0:
        return 0.0
    return overlap / union


def _best_word_oracle(
    words: list[dict[str, Any]],
    gold_start: float,
    gold_end: float,
    max_words: int,
) -> dict[str, Any]:
    best = {
        'tiou': 0.0,
        'start': None,
        'end': None,
        'start_word_id': None,
        'end_word_id': None,
    }

    n = len(words)
    for i in range(n):
        start = float(words[i]['start'])
        upper = min(n, i + max_words)
        for j in range(i, upper):
            end = float(words[j]['end'])
            score = tiou(start, end, gold_start, gold_end)
            if score > best['tiou']:
                best = {
                    'tiou': score,
                    'start': start,
                    'end': end,
                    'start_word_id': int(words[i]['id']),
                    'end_word_id': int(words[j]['id']),
                }
                if score >= 1.0 - 1e-12:
                    return best

    return best


def _best_segment_oracle(
    segments: list[dict[str, Any]],
    gold_start: float,
    gold_end: float,
) -> dict[str, Any]:
    best = {
        'tiou': 0.0,
        'start': None,
        'end': None,
        'segment_ids': [],
    }

    for i, segment in enumerate(segments):
        candidates = [
            (
                float(segment['start']),
                float(segment['end']),
                [int(segment['id'])],
            )
        ]

        if i + 1 < len(segments):
            nxt = segments[i + 1]
            candidates.append(
                (
                    float(segment['start']),
                    float(nxt['end']),
                    [int(segment['id']), int(nxt['id'])],
                )
            )

        for start, end, ids in candidates:
            score = tiou(start, end, gold_start, gold_end)
            if score > best['tiou']:
                best = {
                    'tiou': score,
                    'start': start,
                    'end': end,
                    'segment_ids': ids,
                }

    return best


def _failure_category(
    predicted_answer: bool,
    pred_start: float | None,
    pred_end: float | None,
    gold_start: float,
    gold_end: float,
    score: float,
    tolerance: float = 0.15,
) -> str:
    if not predicted_answer:
        return 'missed_positive'
    if pred_start is None or pred_end is None:
        return 'no_span'
    if score <= 1e-12:
        return 'wrong_location_no_overlap'
    if score >= 0.90:
        return 'excellent'

    start_delta = pred_start - gold_start
    end_delta = pred_end - gold_end

    if start_delta > tolerance and end_delta < -tolerance:
        return 'too_narrow_both_sides'
    if start_delta < -tolerance and end_delta > tolerance:
        return 'too_wide_both_sides'
    if start_delta > tolerance:
        return 'start_too_late'
    if start_delta < -tolerance:
        return 'start_too_early'
    if end_delta < -tolerance:
        return 'end_too_early'
    if end_delta > tolerance:
        return 'end_too_late'
    return 'near_gold_boundaries'


def _tiou_bucket(score: float) -> str:
    if score <= 1e-12:
        return '0'
    if score < 0.25:
        return '(0,0.25)'
    if score < 0.50:
        return '[0.25,0.50)'
    if score < 0.75:
        return '[0.50,0.75)'
    if score < 0.90:
        return '[0.75,0.90)'
    return '[0.90,1.00]'


def _interval_context(
    segments: list[dict[str, Any]],
    start: float | None,
    end: float | None,
    radius: int = 1,
) -> str:
    if start is None or end is None or not segments:
        return ''

    hit_indices = [
        i
        for i, segment in enumerate(segments)
        if float(segment['end']) >= start and float(segment['start']) <= end
    ]

    if not hit_indices:
        midpoint = (start + end) / 2.0
        nearest = min(
            range(len(segments)),
            key=lambda i: abs(
                (
                    float(segments[i]['start'])
                    + float(segments[i]['end'])
                )
                / 2.0
                - midpoint
            ),
        )
        hit_indices = [nearest]

    lo = max(0, min(hit_indices) - radius)
    hi = min(len(segments), max(hit_indices) + radius + 1)

    return '\n'.join(
        f"[S{segment['id']} {float(segment['start']):.2f}-"
        f"{float(segment['end']):.2f}] {segment['text']}"
        for segment in segments[lo:hi]
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def load_gold_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {
            'transcript_id',
            'question',
            'question_type',
            'evidence_start',
            'evidence_end',
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'{path} is missing required CSV columns: {sorted(missing)}'
            )

        for row in reader:
            parsed = dict(row)
            parsed['_gold_answer'] = _bool_from_gold(row)
            parsed['_gold_start'] = _float_or_none(row.get('evidence_start'))
            parsed['_gold_end'] = _float_or_none(row.get('evidence_end'))
            grouped[str(row['transcript_id']).strip()].append(parsed)

    return grouped


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob('*.json'))
    if not files:
        raise FileNotFoundError(
            f'No conversation JSON files found in {path}. '
            'Run the API with DIAGNOSTICS_ENABLED=true first.'
        )

    payloads = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 1:
            raise ValueError(
                f'Unsupported diagnostics schema in {file_path}: '
                f"{payload.get('schema_version')}"
            )
        payload['_source_file'] = str(file_path)
        payloads.append(payload)

    return payloads


def match_gold_questions(
    payload: dict[str, Any],
    gold_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_text: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in gold_rows:
        by_text[_normalize_question(str(row['question']))].append(row)

    pairs = []
    unmatched = []
    for prediction in payload['questions']:
        key = _normalize_question(str(prediction['question']))
        if not by_text[key]:
            unmatched.append(prediction['question'])
            continue
        pairs.append((prediction, by_text[key].popleft()))

    leftovers = sum(len(queue) for queue in by_text.values())
    if unmatched or leftovers:
        raise ValueError(
            f"Question mismatch for {payload['audio_filename']}: "
            f'{len(unmatched)} predictions unmatched, '
            f'{leftovers} gold rows unused. '
            f'First unmatched: {unmatched[:1]}'
        )

    return pairs


def build_rows(
    payloads: list[dict[str, Any]],
    gold_by_transcript: dict[str, list[dict[str, Any]]],
    oracle_max_words: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        transcript_id = _transcript_id(payload['audio_filename'])
        if transcript_id not in gold_by_transcript:
            raise KeyError(
                f"No gold rows found for {payload['audio_filename']} "
                f'(derived transcript_id={transcript_id!r})'
            )

        for prediction, gold in match_gold_questions(
            payload,
            gold_by_transcript[transcript_id],
        ):
            gold_answer = bool(gold['_gold_answer'])
            predicted_answer = bool(prediction['answer'])
            gold_start = gold['_gold_start']
            gold_end = gold['_gold_end']
            pred_start = prediction.get('resolved_evidence_start')
            pred_end = prediction.get('resolved_evidence_end')

            row: dict[str, Any] = {
                'audio_filename': payload['audio_filename'],
                'transcript_id': transcript_id,
                'question_index': prediction['question_index'],
                'question_id': gold.get('question_id', ''),
                'question': prediction['question'],
                'question_type': gold['question_type'],
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'correct': predicted_answer == gold_answer,
                'reason_code': prediction.get('reason_code'),
                'evidence_strategy': prediction.get('evidence_strategy'),
                'evidence_segment_ids': prediction.get('evidence_segment_ids'),
                'evidence_start_word_id': prediction.get(
                    'evidence_start_word_id'
                ),
                'evidence_end_word_id': prediction.get(
                    'evidence_end_word_id'
                ),
                'evidence_quote': prediction.get('evidence_quote', ''),
                'selected_word_text': prediction.get(
                    'selected_word_text',
                    '',
                ),
                'gold_start': gold_start,
                'gold_end': gold_end,
                'pred_start': pred_start,
                'pred_end': pred_end,
                'gold_duration': (
                    gold_end - gold_start
                    if gold_start is not None and gold_end is not None
                    else None
                ),
                'pred_duration': (
                    pred_end - pred_start
                    if pred_start is not None and pred_end is not None
                    else None
                ),
                'source_file': payload['_source_file'],
            }

            if gold_answer and gold_start is not None and gold_end is not None:
                score = tiou(
                    pred_start,
                    pred_end,
                    gold_start,
                    gold_end,
                )
                word_oracle = _best_word_oracle(
                    payload['words'],
                    gold_start,
                    gold_end,
                    oracle_max_words,
                )
                segment_oracle = _best_segment_oracle(
                    payload['segments'],
                    gold_start,
                    gold_end,
                )

                row.update(
                    {
                        'tiou': score,
                        'tiou_bucket': _tiou_bucket(score),
                        'failure_category': _failure_category(
                            predicted_answer,
                            pred_start,
                            pred_end,
                            gold_start,
                            gold_end,
                            score,
                        ),
                        'start_error_s': (
                            pred_start - gold_start
                            if pred_start is not None
                            else None
                        ),
                        'end_error_s': (
                            pred_end - gold_end
                            if pred_end is not None
                            else None
                        ),
                        'oracle_word_tiou': word_oracle['tiou'],
                        'oracle_word_start': word_oracle['start'],
                        'oracle_word_end': word_oracle['end'],
                        'oracle_word_start_id': word_oracle['start_word_id'],
                        'oracle_word_end_id': word_oracle['end_word_id'],
                        'oracle_segment_tiou': segment_oracle['tiou'],
                        'oracle_segment_start': segment_oracle['start'],
                        'oracle_segment_end': segment_oracle['end'],
                        'oracle_segment_ids': segment_oracle['segment_ids'],
                        'oracle_gap': max(0.0, word_oracle['tiou'] - score),
                        'gold_context': _interval_context(
                            payload['segments'],
                            gold_start,
                            gold_end,
                        ),
                        'pred_context': _interval_context(
                            payload['segments'],
                            pred_start,
                            pred_end,
                        ),
                    }
                )
            else:
                row.update(
                    {
                        'tiou': None,
                        'tiou_bucket': '',
                        'failure_category': '',
                        'start_error_s': None,
                        'end_error_s': None,
                        'oracle_word_tiou': None,
                        'oracle_word_start': None,
                        'oracle_word_end': None,
                        'oracle_word_start_id': None,
                        'oracle_word_end_id': None,
                        'oracle_segment_tiou': None,
                        'oracle_segment_start': None,
                        'oracle_segment_end': None,
                        'oracle_segment_ids': [],
                        'oracle_gap': None,
                        'gold_context': '',
                        'pred_context': '',
                    }
                )

            rows.append(row)

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row['gold_answer']]
    predicted_yes_positives = [
        row for row in positives if row['predicted_answer']
    ]

    accuracy_by_type = {}
    for question_type in sorted({row['question_type'] for row in rows}):
        group = [row for row in rows if row['question_type'] == question_type]
        correct = sum(bool(row['correct']) for row in group)
        accuracy_by_type[question_type] = {
            'correct': correct,
            'total': len(group),
            'accuracy': correct / len(group) if group else None,
        }

    tiou_values = [float(row['tiou']) for row in positives]
    diag_tiou_values = [
        float(row['tiou']) for row in predicted_yes_positives
    ]
    word_oracles = [
        float(row['oracle_word_tiou']) for row in positives
    ]
    segment_oracles = [
        float(row['oracle_segment_tiou']) for row in positives
    ]

    oracle_same_classifier = [
        float(row['oracle_word_tiou'])
        if row['predicted_answer']
        else 0.0
        for row in positives
    ]

    spans = [
        row
        for row in predicted_yes_positives
        if row['pred_start'] is not None and row['pred_end'] is not None
    ]
    overlapping = [row for row in spans if float(row['tiou']) > 0.0]

    pred_durations = [
        float(row['pred_duration'])
        for row in spans
        if row['pred_duration'] is not None
    ]
    gold_durations = [
        float(row['gold_duration'])
        for row in positives
        if row['gold_duration'] is not None
    ]

    start_errors = [
        float(row['start_error_s'])
        for row in overlapping
        if row['start_error_s'] is not None
    ]
    end_errors = [
        float(row['end_error_s'])
        for row in overlapping
        if row['end_error_s'] is not None
    ]

    accuracy = (
        sum(bool(row['correct']) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    mean_tiou = _mean(tiou_values) or 0.0
    current_score = 0.4 * accuracy + 0.6 * mean_tiou

    return {
        'questions': len(rows),
        'correct': sum(bool(row['correct']) for row in rows),
        'accuracy': accuracy,
        'accuracy_by_type': accuracy_by_type,
        'positives': len(positives),
        'mean_tiou': mean_tiou,
        'median_tiou': _median(tiou_values),
        'tiou_when_answered_yes': _mean(diag_tiou_values),
        'zero_tiou_positives': sum(
            float(row['tiou']) <= 1e-12 for row in positives
        ),
        'correct_yes_zero_tiou': sum(
            row['predicted_answer']
            and row['correct']
            and float(row['tiou']) <= 1e-12
            for row in positives
        ),
        'tiou_buckets': dict(
            Counter(str(row['tiou_bucket']) for row in positives)
        ),
        'failure_categories': dict(
            Counter(str(row['failure_category']) for row in positives)
        ),
        'evidence_strategies': dict(
            Counter(
                str(row['evidence_strategy'])
                for row in rows
                if row['predicted_answer']
            )
        ),
        'word_oracle': {
            'mean_tiou_all_positives': _mean(word_oracles),
            'median_tiou_all_positives': _median(word_oracles),
            'mean_tiou_with_current_classifier': _mean(
                oracle_same_classifier
            ),
            'recoverable_mean_tiou_gap_with_current_classifier': (
                (_mean(oracle_same_classifier) or 0.0) - mean_tiou
            ),
        },
        'segment_oracle': {
            'mean_tiou_all_positives': _mean(segment_oracles),
            'median_tiou_all_positives': _median(segment_oracles),
        },
        'span_shape': {
            'mean_gold_duration_s': _mean(gold_durations),
            'median_gold_duration_s': _median(gold_durations),
            'mean_pred_duration_s': _mean(pred_durations),
            'median_pred_duration_s': _median(pred_durations),
            'mean_start_error_s_on_overlaps': _mean(start_errors),
            'median_start_error_s_on_overlaps': _median(start_errors),
            'mean_end_error_s_on_overlaps': _mean(end_errors),
            'median_end_error_s_on_overlaps': _median(end_errors),
            'pred_shorter_than_gold': sum(
                row['pred_duration'] is not None
                and row['gold_duration'] is not None
                and float(row['pred_duration']) < float(row['gold_duration'])
                for row in spans
            ),
            'pred_longer_than_gold': sum(
                row['pred_duration'] is not None
                and row['gold_duration'] is not None
                and float(row['pred_duration']) > float(row['gold_duration'])
                for row in spans
            ),
        },
        'current_score': current_score,
        'score_if_word_localization_oracle_with_current_classifier': (
            0.4 * accuracy
            + 0.6 * (_mean(oracle_same_classifier) or 0.0)
        ),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        'audio_filename',
        'transcript_id',
        'question_index',
        'question_id',
        'question',
        'question_type',
        'gold_answer',
        'predicted_answer',
        'correct',
        'reason_code',
        'evidence_strategy',
        'evidence_segment_ids',
        'evidence_start_word_id',
        'evidence_end_word_id',
        'evidence_quote',
        'selected_word_text',
        'gold_start',
        'gold_end',
        'pred_start',
        'pred_end',
        'gold_duration',
        'pred_duration',
        'tiou',
        'tiou_bucket',
        'failure_category',
        'start_error_s',
        'end_error_s',
        'oracle_word_tiou',
        'oracle_word_start',
        'oracle_word_end',
        'oracle_word_start_id',
        'oracle_word_end_id',
        'oracle_segment_tiou',
        'oracle_segment_start',
        'oracle_segment_end',
        'oracle_segment_ids',
        'oracle_gap',
    ]

    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                output[key] = value
            writer.writerow(output)


def write_worst_cases(
    rows: list[dict[str, Any]],
    path: Path,
    limit: int,
) -> None:
    positives = [row for row in rows if row['gold_answer']]
    positives.sort(
        key=lambda row: (
            float(row['tiou']),
            -float(row['oracle_gap'] or 0.0),
        )
    )

    lines = [
        '# Worst evidence-localization cases',
        '',
        'Sorted by current tIoU ascending, then recoverable word-oracle gap.',
        '',
    ]

    for rank, row in enumerate(positives[:limit], start=1):
        pred_text = (
            f"{row['pred_start']:.3f}–{row['pred_end']:.3f}s"
            if row['pred_start'] is not None and row['pred_end'] is not None
            else 'none'
        )
        oracle_text = (
            f"W{row['oracle_word_start_id']}–W{row['oracle_word_end_id']} "
            f"({row['oracle_word_start']:.3f}–"
            f"{row['oracle_word_end']:.3f}s)"
            if row['oracle_word_start'] is not None
            else 'none'
        )

        lines.extend(
            [
                f"## {rank}. {row['question_id'] or row['audio_filename']}",
                '',
                f"**Question:** {row['question']}",
                '',
                (
                    f"**Result:** predicted={row['predicted_answer']} "
                    f"gold={row['gold_answer']} "
                    f"tIoU={float(row['tiou']):.3f} "
                    f"word-oracle={float(row['oracle_word_tiou']):.3f} "
                    f"category={row['failure_category']}"
                ),
                '',
                (
                    f"**Gold:** {row['gold_start']:.3f}–"
                    f"{row['gold_end']:.3f}s | "
                    f"**Pred:** {pred_text}"
                ),
                '',
                (
                    f"**Strategy:** {row['evidence_strategy']} | "
                    f"segments={row['evidence_segment_ids']} | "
                    f"word_ids={row['evidence_start_word_id']}–"
                    f"{row['evidence_end_word_id']}"
                ),
                '',
                f"**Qwen quote:** {row['evidence_quote']}",
                '',
                f"**Selected words:** {row['selected_word_text']}",
                '',
                '**Gold context:**',
                '',
                '\`\`\`text',
                row['gold_context'] or '(none)',
                '\`\`\`',
                '',
                '**Predicted context:**',
                '',
                '\`\`\`text',
                row['pred_context'] or '(none)',
                '\`\`\`',
                '',
                f'**Best word-aligned oracle:** {oracle_text}',
                '',
            ]
        )

    path.write_text('\n'.join(lines), encoding='utf-8')


def print_summary(summary: dict[str, Any]) -> None:
    print('Evidence diagnostics')
    print(f"  questions                         {summary['questions']}")
    print(
        f"  accuracy                          "
        f"{summary['accuracy']:.3f} "
        f"({summary['correct']}/{summary['questions']})"
    )
    print(
        f"  mean tIoU                         "
        f"{summary['mean_tiou']:.3f}"
    )
    print(
        f"  tIoU when answered yes            "
        f"{(summary['tiou_when_answered_yes'] or 0.0):.3f}"
    )
    print(
        f"  zero-tIoU positives               "
        f"{summary['zero_tiou_positives']}"
    )
    print(
        f"  correct-YES zero-tIoU             "
        f"{summary['correct_yes_zero_tiou']}"
    )
    print()
    print('Oracle ceilings')
    print(
        f"  word oracle, all positives        "
        f"{summary['word_oracle']['mean_tiou_all_positives']:.3f}"
    )
    print(
        f"  word oracle, same classifier      "
        f"{summary['word_oracle']['mean_tiou_with_current_classifier']:.3f}"
    )
    print(
        f"  recoverable mean tIoU gap         "
        f"{summary['word_oracle']['recoverable_mean_tiou_gap_with_current_classifier']:.3f}"
    )
    print(
        f"  segment/adjacent oracle           "
        f"{summary['segment_oracle']['mean_tiou_all_positives']:.3f}"
    )
    print()
    print('Failure categories')
    for key, value in sorted(
        summary['failure_categories'].items(),
        key=lambda pair: (-pair[1], pair[0]),
    ):
        print(f'  {key:32s} {value}')
    print()
    print('Span shape')
    shape = summary['span_shape']
    print(
        f"  median gold duration              "
        f"{(shape['median_gold_duration_s'] or 0.0):.3f}s"
    )
    print(
        f"  median predicted duration         "
        f"{(shape['median_pred_duration_s'] or 0.0):.3f}s"
    )
    print(
        f"  median start error on overlaps    "
        f"{(shape['median_start_error_s_on_overlaps'] or 0.0):+.3f}s"
    )
    print(
        f"  median end error on overlaps      "
        f"{(shape['median_end_error_s_on_overlaps'] or 0.0):+.3f}s"
    )
    print()
    print(
        f"Current score                         "
        f"{summary['current_score']:.3f}"
    )
    print(
        f"Score with word-oracle localization   "
        f"{summary['score_if_word_localization_oracle_with_current_classifier']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diagnostics-dir',
        type=Path,
        default=Path('.diagnostics/word-boundary-v1'),
        help='Directory containing per-conversation diagnostics JSON files.',
    )
    parser.add_argument(
        '--question-csv',
        type=Path,
        required=True,
        help='Path to the official data/question_train.csv.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory. Defaults to <diagnostics-dir>/analysis.',
    )
    parser.add_argument(
        '--worst',
        type=int,
        default=30,
        help='Number of lowest-tIoU positives to include in worst_cases.md.',
    )
    parser.add_argument(
        '--oracle-max-words',
        type=int,
        default=80,
        help='Maximum contiguous words considered by the word oracle.',
    )
    args = parser.parse_args()

    if args.oracle_max_words < 1:
        raise ValueError('--oracle-max-words must be at least 1')

    output_dir = args.output_dir or (args.diagnostics_dir / 'analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold_rows(args.question_csv)
    payloads = load_diagnostics(args.diagnostics_dir)
    rows = build_rows(payloads, gold, args.oracle_max_words)
    summary = summarize(rows)

    summary_path = output_dir / 'summary.json'
    csv_path = output_dir / 'evidence_diagnostics.csv'
    worst_path = output_dir / 'worst_cases.md'

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    write_csv(rows, csv_path)
    write_worst_cases(rows, worst_path, max(1, args.worst))

    print_summary(summary)
    print()
    print(f'Wrote {summary_path}')
    print(f'Wrote {csv_path}')
    print(f'Wrote {worst_path}')


if __name__ == '__main__':
    main()
