from dataclasses import dataclass

from models import TranscriptSegment


@dataclass(frozen=True)
class IndexedWord:
    id: int
    text: str
    start: float
    end: float
    segment_id: int


def build_word_index(segments: list[TranscriptSegment]) -> list[IndexedWord]:
    """Assign stable global word IDs in transcript order.

    IDs are zero-based and contiguous for all Whisper words that have timestamps.
    Both the LLM renderer and evidence resolver must use this function so the
    model's word IDs map to exactly the same Whisper timestamps.
    """
    indexed: list[IndexedWord] = []
    word_id = 0

    for segment in segments:
        for word in segment.words:
            indexed.append(
                IndexedWord(
                    id=word_id,
                    text=word.text,
                    start=float(word.start),
                    end=float(word.end),
                    segment_id=segment.id,
                )
            )
            word_id += 1

    return indexed


def render_transcript_with_word_ids(
    segments: list[TranscriptSegment],
) -> str:
    """Render one readable transcript copy with global word addresses."""
    lines: list[str] = []
    next_word_id = 0

    for segment in segments:
        lines.append(f'[S{segment.id} {segment.start:.2f}-{segment.end:.2f}]')

        if segment.words:
            addressed = []
            for word in segment.words:
                addressed.append(f'{next_word_id}:{word.text}')
                next_word_id += 1
            lines.append(' '.join(addressed))
        else:
            # Preserve untimestamped segment text for classification. Such a
            # segment cannot use direct word-boundary evidence and will fall
            # back to the existing segment/quote path.
            lines.append(segment.text)

    return '\n'.join(lines)


def selected_word_text(
    segments: list[TranscriptSegment],
    start_word_id: int,
    end_word_id: int,
) -> str:
    words = build_word_index(segments)
    if (
        start_word_id < 0
        or end_word_id < start_word_id
        or end_word_id >= len(words)
    ):
        return ''

    return ' '.join(
        word.text for word in words[start_word_id:end_word_id + 1]
    )
