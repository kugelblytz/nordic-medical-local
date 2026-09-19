import base64
import logging

from asr import transcribe
from diagnostic_logging import write_conversation_diagnostics
from evidence import locate_evidence, locate_word_evidence
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
        if not segments:
            raise RuntimeError('ASR returned no transcript segments')

        batch = answer_questions(segments, req.questions)

        answers: list[bool] = []
        starts: list[float | None] = []
        ends: list[float | None] = []
        evidence_strategies: list[str] = []

        for item in batch.answers:
            answer = bool(item.answer)
            start = end = None
            strategy = 'negative'

            if answer:
                start, end = locate_word_evidence(
                    segments,
                    item.evidence_start_word_id,
                    item.evidence_end_word_id,
                    item.evidence_segment_ids,
                )

                if start is not None:
                    strategy = 'word_ids'
                else:
                    start, end = locate_evidence(
                        segments,
                        item.evidence_quote,
                        item.evidence_segment_ids,
                    )
                    strategy = 'quote_or_segment_fallback' if start is not None else 'no_span'

            answers.append(answer)
            starts.append(start)
            ends.append(end)
            evidence_strategies.append(strategy)

        # Best-effort only: diagnostics can never break a valid prediction.
        write_conversation_diagnostics(
            req,
            segments,
            batch,
            starts,
            ends,
            evidence_strategies,
        )

        return ASRQuestionResponseDto(
            answers=answers,
            evidence_start=starts,
            evidence_end=ends,
        )

    except Exception:
        log.exception(
            'Prediction failed for %s; returning valid fallback',
            req.audio_filename,
        )
        return _fallback(n)
