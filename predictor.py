import base64
import logging

from asr import transcribe
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
        starts = []
        ends = []

        for item in batch.answers:
            answer = bool(item.answer)
            start = end = None

            if answer:
                # Experimental primary path: Qwen points directly at the first
                # and last global Whisper word IDs. The resolver maps those IDs
                # to Whisper timestamps with no fuzzy text matching.
                start, end = locate_word_evidence(
                    segments,
                    item.evidence_start_word_id,
                    item.evidence_end_word_id,
                    item.evidence_segment_ids,
                )

                # Safety fallback: if Qwen returns missing/inconsistent word IDs,
                # preserve the known-good quote/segment evidence behavior.
                if start is None:
                    start, end = locate_evidence(
                        segments,
                        item.evidence_quote,
                        item.evidence_segment_ids,
                    )

            answers.append(answer)
            starts.append(start)
            ends.append(end)

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
