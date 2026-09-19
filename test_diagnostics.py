from diagnostics.analyze_evidence import (
    _best_segment_oracle,
    _best_word_oracle,
    _failure_category,
    tiou,
)
from diagnostics.refine_failure_analysis import restricted_word_oracle


def test_tiou_exact_and_partial():
    assert tiou(10.0, 14.0, 10.0, 14.0) == 1.0
    assert abs(tiou(11.0, 15.0, 10.0, 14.0) - 0.6) < 1e-9


def test_word_oracle_recovers_gold_aligned_span():
    words = [
        {'id': 0, 'text': 'a', 'start': 9.0, 'end': 9.5, 'segment_id': 0},
        {'id': 1, 'text': 'b', 'start': 10.0, 'end': 10.5, 'segment_id': 0},
        {'id': 2, 'text': 'c', 'start': 10.5, 'end': 11.0, 'segment_id': 0},
    ]

    oracle = _best_word_oracle(words, 10.0, 11.0, 80)

    assert oracle['tiou'] == 1.0
    assert oracle['start_word_id'] == 1
    assert oracle['end_word_id'] == 2


def test_segment_oracle_checks_adjacent_pairs():
    segments = [
        {'id': 0, 'start': 9.0, 'end': 10.0, 'text': 'a'},
        {'id': 1, 'start': 10.0, 'end': 11.0, 'text': 'b'},
        {'id': 2, 'start': 11.0, 'end': 12.0, 'text': 'c'},
    ]

    oracle = _best_segment_oracle(segments, 10.0, 12.0)

    assert oracle['tiou'] == 1.0
    assert oracle['segment_ids'] == [1, 2]


def test_failure_category_detects_zero_overlap():
    assert (
        _failure_category(True, 20.0, 21.0, 10.0, 11.0, 0.0)
        == 'wrong_location_no_overlap'
    )


def test_restricted_oracle_distinguishes_selected_region():
    words = [
        {'id': 0, 'text': 'wrong', 'start': 1.0, 'end': 2.0, 'segment_id': 0},
        {'id': 1, 'text': 'gold', 'start': 10.0, 'end': 11.0, 'segment_id': 1},
    ]

    assert restricted_word_oracle(words, 10.0, 11.0, {0}) == 0.0
    assert restricted_word_oracle(words, 10.0, 11.0, {1}) == 1.0
