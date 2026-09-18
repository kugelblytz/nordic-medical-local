from evidence import locate_evidence, locate_quote
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
