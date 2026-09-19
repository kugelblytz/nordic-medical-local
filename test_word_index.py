from models import TranscriptSegment, WordToken
from word_index import (
    build_word_index,
    render_transcript_with_word_ids,
    selected_word_text,
)


def segments():
    return [
        TranscriptSegment(
            id=3,
            start=1.0,
            end=2.0,
            text='Hello there.',
            words=[
                WordToken(text='Hello', start=1.1, end=1.4),
                WordToken(text='there.', start=1.5, end=1.9),
            ],
        ),
        TranscriptSegment(
            id=4,
            start=2.0,
            end=3.0,
            text='Take five.',
            words=[
                WordToken(text='Take', start=2.1, end=2.3),
                WordToken(text='five.', start=2.4, end=2.8),
            ],
        ),
    ]


def test_global_word_ids_are_contiguous_across_segments():
    words = build_word_index(segments())

    assert [word.id for word in words] == [0, 1, 2, 3]
    assert [word.segment_id for word in words] == [3, 3, 4, 4]
    assert words[2].text == 'Take'


def test_renderer_uses_same_global_word_ids():
    rendered = render_transcript_with_word_ids(segments())

    assert '[S3 1.00-2.00]' in rendered
    assert '0:Hello 1:there.' in rendered
    assert '[S4 2.00-3.00]' in rendered
    assert '2:Take 3:five.' in rendered


def test_selected_word_text_uses_global_range():
    assert selected_word_text(segments(), 1, 3) == 'there. Take five.'


def test_selected_word_text_rejects_invalid_range():
    assert selected_word_text(segments(), 3, 1) == ''
