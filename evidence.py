import re
from difflib import SequenceMatcher
from models import TranscriptSegment


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _quote_tokens(quote: str) -> list[str]:
    return [t for t in (_norm_token(x) for x in quote.split()) if t]


def locate_quote(segments: list[TranscriptSegment], quote: str):
    q = _quote_tokens(quote)
    if not q:
        return None, None

    words = []
    for seg in segments:
        for w in seg.words:
            n = _norm_token(w.text)
            if n:
                words.append((n, w.start, w.end))

    if not words:
        return None, None

    toks = [x[0] for x in words]
    m = len(q)

    for i in range(0, len(toks) - m + 1):
        if toks[i:i + m] == q:
            return words[i][1], words[i + m - 1][2]

    best = None
    lengths = range(max(1, m - 2), min(len(toks), m + 3) + 1)
    q_join = ' '.join(q)
    for win_len in lengths:
        for i in range(0, len(toks) - win_len + 1):
            cand = ' '.join(toks[i:i + win_len])
            score = SequenceMatcher(None, q_join, cand).ratio()
            if best is None or score > best[0]:
                best = (score, i, win_len)

    if best and best[0] >= 0.82:
        _, i, win_len = best
        return words[i][1], words[i + win_len - 1][2]

    return None, None
