import base64
import logging
from typing import Any

from asr import transcribe
from config import EVIDENCE_REFINEMENT_ENABLED
from diagnostic_logging import write_conversation_diagnostics
from evidence import locate_evidence, locate_word_evidence
from llm import answer_questions, refine_evidence
from models import ASRQuestionRequestDto, ASRQuestionResponseDto
from performance_metrics import elapsed_ms, now_ns
from word_index import selected_word_text

log = logging.getLogger(__name__)


def _fallback(n: int) -> ASRQuestionResponseDto:
    return ASRQuestionResponseDto(
        answers=[False] * n,
        evidence_start=[None] * n,
        evidence_end=[None] * n,
    )


def _resolve_first_pass(segments, item):
    start, end = locate_word_evidence(
        segments,
        item.evidence_start_word_id,
        item.evidence_end_word_id,
        item.evidence_segment_ids,
    )
    if start is not None:
        return start, end, 'word_ids'

    start, end = locate_evidence(
        segments,
        item.evidence_quote,
        item.evidence_segment_ids,
    )
    if start is not None:
        return start, end, 'quote_or_segment_fallback'
    return None, None, 'no_span'


def _calculate_unaccounted_ms(timing: dict[str, Any]) -> float:
    accounted = float(timing.get('base64_decode_ms') or 0.0)
    accounted += float((timing.get('asr') or {}).get('total_ms') or 0.0)
    accounted += float((timing.get('pass1') or {}).get('total_wall_ms') or 0.0)
    accounted += float((timing.get('pass2') or {}).get('total_wall_ms') or 0.0)
    accounted += float(timing.get('evidence_resolution_ms') or 0.0)
    return max(
        0.0,
        float(timing.get('request_core_ms') or 0.0) - accounted,
    )


def predict(req: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    n = len(req.questions)
    request_started = now_ns()
    timing: dict[str, Any] = {}

    try:
        decode_started = now_ns()
        audio = base64.b64decode(req.audio_base64, validate=True)
        timing['base64_decode_ms'] = elapsed_ms(decode_started)

        segments = transcribe(audio, timing=timing)
        if not segments:
            raise RuntimeError('ASR returned no transcript segments')

        # Pass 1 is the frozen classifier. Its answer is authoritative.
        batch = answer_questions(
            segments,
            req.questions,
            timing=timing,
        )

        refinements = {}
        refinement_error = None
        if EVIDENCE_REFINEMENT_ENABLED:
            try:
                refinements = refine_evidence(
                    segments,
                    req.questions,
                    batch,
                    timing=timing,
                )
            except Exception as exc:
                # Evidence refinement is strictly optional. A refiner failure
                # must never turn a valid classification into the all-false
                # outer fallback.
                refinement_error = str(exc)
                log.exception(
                    'Evidence refinement failed for %s; preserving pass-1 evidence',
                    req.audio_filename,
                )
        else:
            timing['pass2'] = {
                'enabled': False,
                'skipped': True,
                'skip_reason': 'disabled',
                'true_question_count': sum(
                    bool(answer.answer) for answer in batch.answers
                ),
                'total_wall_ms': 0.0,
                'attempts': [],
                'retry_count': 0,
                'first_attempt_success': False,
            }

        answers: list[bool] = []
        starts: list[float | None] = []
        ends: list[float | None] = []
        evidence_strategies: list[str] = []

        first_pass_starts: list[float | None] = []
        first_pass_ends: list[float | None] = []
        first_pass_strategies: list[str] = []
        refinement_records: dict[int, dict] = {}

        evidence_counts = {
            'negative_count': 0,
            'first_pass_word_count': 0,
            'refined_word_count': 0,
            'quote_or_segment_fallback_count': 0,
            'no_span_count': 0,
            'refinement_valid_count': 0,
            'refinement_changed_count': 0,
        }
        evidence_started = now_ns()

        for index, item in enumerate(batch.answers):
            answer = bool(item.answer)
            start = end = None
            strategy = 'negative'

            first_start = first_end = None
            first_strategy = 'negative'

            if answer:
                first_start, first_end, first_strategy = _resolve_first_pass(
                    segments,
                    item,
                )
                start, end, strategy = (
                    first_start,
                    first_end,
                    first_strategy,
                )

                candidate = refinements.get(index)
                if candidate is not None:
                    refined_start, refined_end = locate_word_evidence(
                        segments,
                        candidate.start_word_id,
                        candidate.end_word_id,
                        None,
                    )
                    valid = refined_start is not None
                    used = False
                    if valid:
                        evidence_counts['refinement_valid_count'] += 1
                        start = refined_start
                        end = refined_end
                        strategy = 'refined_word_ids'
                        used = True

                    refined_text = ''
                    if (
                        candidate.start_word_id is not None
                        and candidate.end_word_id is not None
                    ):
                        refined_text = selected_word_text(
                            segments,
                            candidate.start_word_id,
                            candidate.end_word_id,
                        )

                    changed = bool(
                        used
                        and (
                            candidate.start_word_id
                            != item.evidence_start_word_id
                            or candidate.end_word_id
                            != item.evidence_end_word_id
                        )
                    )
                    if changed:
                        evidence_counts['refinement_changed_count'] += 1

                    refinement_records[index] = {
                        'attempted': True,
                        'start_word_id': candidate.start_word_id,
                        'end_word_id': candidate.end_word_id,
                        'selected_word_text': refined_text,
                        'resolved_evidence_start': refined_start,
                        'resolved_evidence_end': refined_end,
                        'valid': valid,
                        'used': used,
                        'error': None,
                    }
                elif EVIDENCE_REFINEMENT_ENABLED:
                    refinement_records[index] = {
                        'attempted': True,
                        'start_word_id': None,
                        'end_word_id': None,
                        'selected_word_text': '',
                        'resolved_evidence_start': None,
                        'resolved_evidence_end': None,
                        'valid': False,
                        'used': False,
                        'error': refinement_error or 'missing refinement result',
                    }
            else:
                evidence_counts['negative_count'] += 1

            if strategy == 'refined_word_ids':
                evidence_counts['refined_word_count'] += 1
            elif strategy == 'word_ids':
                evidence_counts['first_pass_word_count'] += 1
            elif strategy == 'quote_or_segment_fallback':
                evidence_counts['quote_or_segment_fallback_count'] += 1
            elif strategy == 'no_span':
                evidence_counts['no_span_count'] += 1

            answers.append(answer)
            starts.append(start)
            ends.append(end)
            evidence_strategies.append(strategy)
            first_pass_starts.append(first_start)
            first_pass_ends.append(first_end)
            first_pass_strategies.append(first_strategy)

        timing['evidence_resolution_ms'] = elapsed_ms(evidence_started)
        timing['evidence_counts'] = evidence_counts

        # request_core_ms intentionally stops before research diagnostic file
        # serialization/write. It is the closest measure of competition runtime.
        timing['request_core_ms'] = elapsed_ms(request_started)
        timing['unaccounted_ms'] = _calculate_unaccounted_ms(timing)

        write_conversation_diagnostics(
            req,
            segments,
            batch,
            starts,
            ends,
            evidence_strategies,
            first_pass_starts=first_pass_starts,
            first_pass_ends=first_pass_ends,
            first_pass_strategies=first_pass_strategies,
            refinement_records=refinement_records,
            timing=timing,
        )

        return ASRQuestionResponseDto(
            answers=answers,
            evidence_start=starts,
            evidence_end=ends,
        )

    except Exception:
        timing['failed_request_ms'] = elapsed_ms(request_started)
        log.exception(
            'Prediction failed for %s; returning valid fallback',
            req.audio_filename,
        )
        return _fallback(n)
