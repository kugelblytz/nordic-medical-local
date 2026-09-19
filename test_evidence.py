from evidence import locate_evidence, locate_quote, locate_word_evidence
from models import TranscriptSegment, WordToken


def sample_segment():
    return TranscriptSegment(
        id=7,
        start=10.0,
        end=15.0,
        text='Take 100 milligrams once daily.',
        words=[
            WordToken(text='Take', start=10.1, end=10.4),
            WordToken(text='100', start=10.5, end=10.8),
            WordToken(text='milligrams', start=10.9, end=11.5),
            WordToken(text='once', start=11.6, end=11.9),
            WordToken(text='daily.', start=12.0, end=12.4),
        ],
    )


def adjacent_segment():
    return TranscriptSegment(
        id=8,
        start=15.0,
        end=18.0,
        text='Continue this for one week.',
        words=[
            WordToken(text='Continue', start=15.1, end=15.5),
            WordToken(text='this', start=15.6, end=15.8),
            WordToken(text='for', start=15.9, end=16.1),
            WordToken(text='one', start=16.2, end=16.4),
            WordToken(text='week.', start=16.5, end=16.9),
        ],
    )


def test_exact_quote_maps_to_word_span():
    assert locate_quote(
        [sample_segment()],
        '100 milligrams once daily',
    ) == (10.5, 12.4)


def test_selected_segment_uses_tight_quote_span():
    start, end = locate_evidence(
        [sample_segment()],
        '100 milligrams once daily',
        segment_ids=[7],
    )
    assert abs(start - 10.38) < 1e-9
    assert abs(end - 12.52) < 1e-9


def test_segment_id_uses_actual_id_not_list_index():
    start, end = locate_evidence(
        [sample_segment()],
        '100 milligrams',
        segment_ids=[7],
    )
    assert abs(start - 10.38) < 1e-9
    assert abs(end - 11.62) < 1e-9


def test_bad_quote_falls_back_to_selected_segment():
    assert locate_evidence(
        [sample_segment()],
        'words that are not in the transcript',
        segment_ids=[7],
    ) == (9.8, 15.2)


def test_no_segment_id_retains_legacy_segment_mode():
    assert locate_evidence(
        [sample_segment()],
        '100 milligrams once daily',
        mode='segment',
    ) == (9.8, 15.2)


def test_word_ids_map_directly_to_whisper_timestamps():
    # Global W1..W4 are 100 through daily.
    assert locate_word_evidence(
        [sample_segment()],
        1,
        4,
        segment_ids=[7],
    ) == (10.5, 12.4)


def test_word_ids_reject_reversed_range():
    assert locate_word_evidence(
        [sample_segment()],
        4,
        1,
        segment_ids=[7],
    ) == (None, None)


def test_word_ids_reject_out_of_range_id():
    assert locate_word_evidence(
        [sample_segment()],
        1,
        99,
        segment_ids=[7],
    ) == (None, None)


def test_word_ids_ignore_redundant_segment_mismatch():
    assert locate_word_evidence(
        [sample_segment()],
        1,
        4,
        segment_ids=[8],
    ) == (10.5, 12.4)


def test_word_range_can_cross_one_adjacent_segment_boundary():
    segments = [sample_segment(), adjacent_segment()]

    # W4 is "daily." in S7 and W5.. are in S8.
    assert locate_word_evidence(
        segments,
        4,
        6,
        segment_ids=[7, 8],
    ) == (12.0, 15.8)


def test_cross_segment_word_range_is_authoritative():
    segments = [sample_segment(), adjacent_segment()]

    assert locate_word_evidence(
        segments,
        4,
        6,
        segment_ids=[7],
    ) == (12.0, 15.8)


def test_word_range_rejects_three_segments():
    third = TranscriptSegment(
        id=9,
        start=18.0,
        end=20.0,
        text='Then stop.',
        words=[
            WordToken(text='Then', start=18.1, end=18.4),
            WordToken(text='stop.', start=18.5, end=18.9),
        ],
    )
    segments = [sample_segment(), adjacent_segment(), third]

    # W4 in S7 through W10 in S9 crosses three Whisper segments.
    assert locate_word_evidence(
        segments,
        4,
        10,
        segment_ids=[7, 8],
    ) == (None, None)
