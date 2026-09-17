from evidence import locate_evidence, locate_quote
from models import TranscriptSegment, WordToken


def sample_segment():
    return TranscriptSegment(
        id=0,
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


def test_segment_evidence_uses_segment_boundary_with_padding():
    assert locate_evidence(
        [sample_segment()],
        '100 milligrams once daily',
        mode='segment',
    ) == (9.8, 15.2)


def test_segment_id_is_preferred_over_quote_search():
    assert locate_evidence(
        [sample_segment()],
        'quote can even be imperfect here',
        segment_ids=[0],
    ) == (9.8, 15.2)
