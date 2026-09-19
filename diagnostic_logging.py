import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DIAGNOSTICS_DIR, DIAGNOSTICS_ENABLED
from models import ASRQuestionRequestDto, LLMAnswerBatch, TranscriptSegment
from performance_metrics import elapsed_ms, now_ns
from word_index import build_word_index, selected_word_text


log = logging.getLogger(__name__)


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    safe = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in stem)
    return safe or 'conversation'


def write_conversation_diagnostics(
    req: ASRQuestionRequestDto,
    segments: list[TranscriptSegment],
    batch: LLMAnswerBatch,
    starts: list[float | None],
    ends: list[float | None],
    evidence_strategies: list[str],
    *,
    first_pass_starts: list[float | None] | None = None,
    first_pass_ends: list[float | None] | None = None,
    first_pass_strategies: list[str] | None = None,
    refinement_records: dict[int, dict] | None = None,
    timing: dict[str, Any] | None = None,
) -> None:
    """Persist one atomic JSON record per conversation.

    Diagnostics are deliberately best-effort. Any logging failure is swallowed
    so research instrumentation can never break /predict.
    """
    if not DIAGNOSTICS_ENABLED:
        return

    diagnostics_started = now_ns()
    try:
        output_dir = Path(DIAGNOSTICS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload_started = now_ns()
        indexed_words = build_word_index(segments)
        words_payload = [
            {
                'id': word.id,
                'text': word.text,
                'start': word.start,
                'end': word.end,
                'segment_id': word.segment_id,
            }
            for word in indexed_words
        ]

        segments_payload = []
        word_cursor = 0
        for segment in segments:
            word_ids = list(range(word_cursor, word_cursor + len(segment.words)))
            word_cursor += len(segment.words)
            segments_payload.append(
                {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'word_ids': word_ids,
                }
            )

        refinement_records = refinement_records or {}
        first_pass_starts = first_pass_starts or starts
        first_pass_ends = first_pass_ends or ends
        first_pass_strategies = first_pass_strategies or evidence_strategies

        questions_payload = []
        for idx, (question, item, start, end, strategy) in enumerate(
            zip(
                req.questions,
                batch.answers,
                starts,
                ends,
                evidence_strategies,
            )
        ):
            selected_text = ''
            if (
                item.evidence_start_word_id is not None
                and item.evidence_end_word_id is not None
            ):
                selected_text = selected_word_text(
                    segments,
                    item.evidence_start_word_id,
                    item.evidence_end_word_id,
                )

            refinement = refinement_records.get(idx)
            questions_payload.append(
                {
                    'question_index': idx,
                    'question': question,
                    'answer': bool(item.answer),
                    'reason_code': item.reason_code,
                    'evidence_segment_ids': item.evidence_segment_ids,
                    'evidence_start_word_id': item.evidence_start_word_id,
                    'evidence_end_word_id': item.evidence_end_word_id,
                    'evidence_quote': item.evidence_quote,
                    'selected_word_text': selected_text,
                    'first_pass_resolved_evidence_start': first_pass_starts[idx],
                    'first_pass_resolved_evidence_end': first_pass_ends[idx],
                    'first_pass_evidence_strategy': first_pass_strategies[idx],
                    'refinement': refinement,
                    'refinement_changed_range': bool(
                        refinement
                        and refinement.get('used')
                        and (
                            refinement.get('start_word_id')
                            != item.evidence_start_word_id
                            or refinement.get('end_word_id')
                            != item.evidence_end_word_id
                        )
                    ),
                    'resolved_evidence_start': start,
                    'resolved_evidence_end': end,
                    'evidence_strategy': strategy,
                }
            )

        timing_payload = dict(timing or {})
        diagnostics_timing = dict(timing_payload.get('diagnostics') or {})
        diagnostics_timing['payload_build_ms'] = elapsed_ms(payload_started)
        timing_payload['diagnostics'] = diagnostics_timing

        payload = {
            'schema_version': 3,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'audio_filename': req.audio_filename,
            'questions': questions_payload,
            'segments': segments_payload,
            'words': words_payload,
            'timing': timing_payload,
        }

        # Serialize once to measure representative JSON encoding cost, then
        # record that measurement in the final serialized payload.
        serialize_started = now_ns()
        json.dumps(payload, ensure_ascii=False)
        diagnostics_timing['json_serialize_ms'] = elapsed_ms(serialize_started)
        diagnostics_timing['pre_write_total_ms'] = elapsed_ms(
            diagnostics_started
        )
        payload['timing']['diagnostics'] = diagnostics_timing

        target = output_dir / f'{_safe_stem(req.audio_filename)}.json'
        temporary = target.with_suffix('.json.tmp')

        final_serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )

        write_started = now_ns()
        temporary.write_text(final_serialized, encoding='utf-8')
        temporary.replace(target)
        file_write_ms = elapsed_ms(write_started)

        # File-write time cannot be embedded without adding another write and
        # contaminating the measurement. Keep it in the server log instead.
        log.info(
            'Diagnostics timing for %s: build=%.2fms serialize=%.2fms write=%.2fms',
            req.audio_filename,
            diagnostics_timing['payload_build_ms'],
            diagnostics_timing['json_serialize_ms'],
            file_write_ms,
        )

    except Exception:
        log.exception(
            'Evidence diagnostics logging failed for %s',
            req.audio_filename,
        )
