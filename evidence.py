import re
from difflib import SequenceMatcher

from config import EVIDENCE_MODE, EVIDENCE_PAD_SECONDS
from models import TranscriptSegment


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _quote_tokens(quote: str) -> list[str]:
    return [t for t in (_norm_token(x) for x in quote.split()) if t]


def _flatten_words(segments: list[TranscriptSegment]):
    words = []
    for seg_idx, seg in enumerate(segments):
        for word in seg.words:
            token = _norm_token(word.text)
            if token:
                words.append((token, word.start, word.end, seg_idx))
    return words


def _find_quote_window(segments: list[TranscriptSegment], quote: str):
    query = _quote_tokens(quote)
    if not query:
        return None

    words = _flatten_words(segments)
    if not words:
        return None

    tokens = [word[0] for word in words]
    width = len(query)

    for start in range(0, len(tokens) - width + 1):
        if tokens[start:start + width] == query:
            return words, start, width

    best = None
    query_text = ' '.join(query)
    for window_width in range(
        max(1, width - 2),
        min(len(tokens), width + 3) + 1,
    ):
        for start in range(0, len(tokens) - window_width + 1):
            candidate = ' '.join(tokens[start:start + window_width])
            score = SequenceMatcher(None, query_text, candidate).ratio()
            if best is None or score > best[0]:
                best = (score, start, window_width)

    if best and best[0] >= 0.82:
        _, start, window_width = best
        return words, start, window_width

    return None


def locate_quote(segments: list[TranscriptSegment], quote: str):
    match = _find_quote_window(segments, quote)
    if match is None:
        return None, None

    words, start, width = match
    return words[start][1], words[start + width - 1][2]


def locate_evidence(
    segments: list[TranscriptSegment],
    quote: str,
    mode: str = EVIDENCE_MODE,
):
    match = _find_quote_window(segments, quote)
    if match is None:
        return None, None

    words, start, width = match
    first = words[start]
    last = words[start + width - 1]

    if mode == 'word':
        evidence_start = first[1]
        evidence_end = last[2]
    else:
        first_segment = first[3]
        last_segment = last[3]
        evidence_start = segments[first_segment].start
        evidence_end = segments[last_segment].end

    evidence_start = max(0.0, evidence_start - EVIDENCE_PAD_SECONDS)
    evidence_end = evidence_end + EVIDENCE_PAD_SECONDS
    return evidence_start, evidence_end
