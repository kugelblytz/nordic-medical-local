from experiments.evidence_candidate_lab import (
    choose_cohort,
    recall_at_k,
    tiou,
    validate_candidate,
)


def _payload():
    words = []
    for i in range(8):
        words.append(
            {
                'id': i,
                'text': f'w{i}',
                'start': float(i),
                'end': float(i) + 0.5,
                'segment_id': 0 if i < 4 else 1,
            }
        )
    return {'words': words}


def test_tiou():
    assert tiou(1.0, 3.0, 1.0, 3.0) == 1.0
    assert tiou(0.0, 1.0, 2.0, 3.0) == 0.0
    assert abs(tiou(1.0, 4.0, 2.0, 3.0) - (1.0 / 3.0)) < 1e-9


def test_validate_candidate_accepts_adjacent_segments():
    candidate = validate_candidate(
        {'start_word_id': 2, 'end_word_id': 5},
        _payload(),
        max_words=10,
    )
    assert candidate is not None
    assert candidate['segment_ids'] == [0, 1]
    assert candidate['start'] == 2.0
    assert candidate['end'] == 5.5


def test_validate_candidate_rejects_too_many_words():
    assert validate_candidate(
        {'start_word_id': 0, 'end_word_id': 7},
        _payload(),
        max_words=4,
    ) is None


def test_choose_cohorts():
    rows = [
        {'predicted_answer': True, 'baseline_tiou': 0.0},
        {'predicted_answer': True, 'baseline_tiou': 0.20},
        {'predicted_answer': True, 'baseline_tiou': 0.80},
        {'predicted_answer': False, 'baseline_tiou': 0.0},
    ]
    assert len(choose_cohort(rows, 'zero', 0.25, 0.75, False)) == 1
    assert len(choose_cohort(rows, 'low', 0.25, 0.75, False)) == 2
    assert len(choose_cohort(rows, 'control', 0.25, 0.75, False)) == 1
    assert len(choose_cohort(rows, 'all', 0.25, 0.75, False)) == 3
    assert len(choose_cohort(rows, 'zero', 0.25, 0.75, True)) == 2


def test_recall_at_k():
    rows = [
        {'candidates': [{'tiou': 0.1}, {'tiou': 0.8}]},
        {'candidates': [{'tiou': 0.9}]},
    ]
    assert recall_at_k(rows, 1, 0.75) == 0.5
    assert recall_at_k(rows, 2, 0.75) == 1.0
