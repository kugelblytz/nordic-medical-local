#!/usr/bin/env python3
"""Classify low-tIoU cases by how far Qwen's semantic location is from gold."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_evidence import (
    _transcript_id,
    load_diagnostics,
    load_gold_rows,
    match_gold_questions,
    tiou,
)


def restricted_word_oracle(
    words: list[dict[str, Any]],
    gold_start: float,
    gold_end: float,
    allowed_segment_ids: set[int],
    max_words: int = 80,
    max_segments: int = 2,
) -> float:
    if not allowed_segment_ids:
        return 0.0

    best = 0.0
    n = len(words)

    for i in range(n):
        first_segment = int(words[i]['segment_id'])
        if first_segment not in allowed_segment_ids:
            continue

        start = float(words[i]['start'])
        seen_segments: set[int] = set()

        for j in range(i, min(n, i + max_words)):
            segment_id = int(words[j]['segment_id'])
            if segment_id not in allowed_segment_ids:
                break

            seen_segments.add(segment_id)
            if len(seen_segments) > max_segments:
                break

            score = tiou(
                start,
                float(words[j]['end']),
                gold_start,
                gold_end,
            )
            best = max(best, score)

    return best


def classify_case(
    current_tiou: float,
    selected_oracle: float,
    plus_one_oracle: float,
) -> str:
    if current_tiou >= 0.75:
        return 'already_good'

    if selected_oracle >= max(0.50, current_tiou + 0.20):
        return 'word_boundary_failure'

    if plus_one_oracle >= max(0.50, current_tiou + 0.20):
        return 'adjacent_segment_failure'

    if current_tiou <= 1e-12:
        return 'wrong_semantic_location'

    return 'mixed_or_annotation_shape'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diagnostics-dir',
        type=Path,
        default=Path('.diagnostics/word-boundary-v1'),
    )
    parser.add_argument('--question-csv', type=Path, required=True)
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Defaults to <diagnostics-dir>/analysis/failure_refinement.csv',
    )
    parser.add_argument('--max-words', type=int, default=80)
    args = parser.parse_args()

    gold = load_gold_rows(args.question_csv)
    payloads = load_diagnostics(args.diagnostics_dir)

    output = args.output or (
        args.diagnostics_dir / 'analysis' / 'failure_refinement.csv'
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for payload in payloads:
        transcript_id = _transcript_id(payload['audio_filename'])
        gold_rows = gold[transcript_id]
        all_segment_ids = {
            int(segment['id']) for segment in payload['segments']
        }

        for prediction, gold_row in match_gold_questions(payload, gold_rows):
            if not bool(gold_row['_gold_answer']):
                continue

            gold_start = gold_row['_gold_start']
            gold_end = gold_row['_gold_end']
            if gold_start is None or gold_end is None:
                continue

            pred_start = prediction.get('resolved_evidence_start')
            pred_end = prediction.get('resolved_evidence_end')
            current = tiou(
                pred_start,
                pred_end,
                gold_start,
                gold_end,
            )

            selected_ids = {
                int(value)
                for value in prediction.get('evidence_segment_ids', [])
            }
            plus_one_ids = set(selected_ids)
            for segment_id in list(selected_ids):
                if segment_id - 1 in all_segment_ids:
                    plus_one_ids.add(segment_id - 1)
                if segment_id + 1 in all_segment_ids:
                    plus_one_ids.add(segment_id + 1)

            selected_oracle = restricted_word_oracle(
                payload['words'],
                gold_start,
                gold_end,
                selected_ids,
                args.max_words,
            )
            plus_one_oracle = restricted_word_oracle(
                payload['words'],
                gold_start,
                gold_end,
                plus_one_ids,
                args.max_words,
            )

            category = classify_case(
                current,
                selected_oracle,
                plus_one_oracle,
            )

            rows.append(
                {
                    'audio_filename': payload['audio_filename'],
                    'question_id': gold_row.get('question_id', ''),
                    'question': prediction['question'],
                    'predicted_answer': bool(prediction['answer']),
                    'current_tiou': current,
                    'selected_region_oracle': selected_oracle,
                    'plus_one_segment_oracle': plus_one_oracle,
                    'category': category,
                    'selected_segment_ids': json.dumps(
                        sorted(selected_ids)
                    ),
                    'start_word_id': prediction.get(
                        'evidence_start_word_id'
                    ),
                    'end_word_id': prediction.get(
                        'evidence_end_word_id'
                    ),
                    'evidence_quote': prediction.get(
                        'evidence_quote',
                        '',
                    ),
                    'selected_word_text': prediction.get(
                        'selected_word_text',
                        '',
                    ),
                }
            )

    fields = [
        'audio_filename',
        'question_id',
        'question',
        'predicted_answer',
        'current_tiou',
        'selected_region_oracle',
        'plus_one_segment_oracle',
        'category',
        'selected_segment_ids',
        'start_word_id',
        'end_word_id',
        'evidence_quote',
        'selected_word_text',
    ]

    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    low = [row for row in rows if float(row['current_tiou']) < 0.75]
    zero_correct_yes = [
        row
        for row in rows
        if row['predicted_answer']
        and float(row['current_tiou']) <= 1e-12
    ]

    print('Failure refinement')
    print(f'  positive questions          {len(rows)}')
    print(f'  positives below 0.75       {len(low)}')
    print(f'  correct-YES zero tIoU      {len(zero_correct_yes)}')
    print()
    print('Low-tIoU categories')
    counts = Counter(row['category'] for row in low)
    for category, count in sorted(
        counts.items(),
        key=lambda pair: (-pair[1], pair[0]),
    ):
        print(f'  {category:28s} {count}')
    print()
    print('Zero-tIoU correct-YES categories')
    zero_counts = Counter(row['category'] for row in zero_correct_yes)
    for category, count in sorted(
        zero_counts.items(),
        key=lambda pair: (-pair[1], pair[0]),
    ):
        print(f'  {category:28s} {count}')

    print()
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
