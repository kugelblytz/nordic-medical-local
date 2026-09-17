import base64
import logging
from asr import transcribe
from evidence import locate_quote
from llm import answer_questions
from models import ASRQuestionRequestDto, ASRQuestionResponseDto

log = logging.getLogger(__name__)


def _fallback(n: int) -> ASRQuestionResponseDto:
    return ASRQuestionResponseDto(
        answers=[False] * n,
        evidence_start=[None] * n,
        evidence_end=[None] * n,
    )


def predict(req: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    n = len(req.questions)
    try:
        audio = base64.b64decode(req.audio_base64, validate=True)
        segments = transcribe(audio)
        batch = answer_questions(segments, req.questions)

        answers = []
        starts = []
        ends = []

        for item in batch.answers:
            answer = bool(item.answer)
            start = end = None

            if answer:
                start, end = locate_quote(segments, item.evidence_quote)

            answers.append(answer)
            starts.append(start)
            ends.append(end)

        return ASRQuestionResponseDto(
            answers=answers,
            evidence_start=starts,
            evidence_end=ends,
        )

    except Exception:
        log.exception('Prediction failed for %s; returning valid fallback', req.audio_filename)
        return _fallback(n)
