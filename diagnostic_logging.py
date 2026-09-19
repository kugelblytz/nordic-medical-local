import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import DIAGNOSTICS_DIR, DIAGNOSTICS_ENABLED
from models import ASRQuestionRequestDto, LLMAnswerBatch, TranscriptSegment
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
) -> None:
    """Persist one atomic JSON record per conversation.

    Diagnostics are deliberately best-effort. Any logging failure is swallowed
    so research instrumentation can never break /predict.
    """
    if not DIAGNOSTICS_ENABLED:
        return

    try:
        output_dir = Path(DIAGNOSTICS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        payload = {
            'schema_version': 2,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'audio_filename': req.audio_filename,
            'questions': questions_payload,
            'segments': segments_payload,
            'words': words_payload,
        }

        target = output_dir / f'{_safe_stem(req.audio_filename)}.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        temporary.replace(target)

    except Exception:
        log.exception(
            'Evidence diagnostics logging failed for %s',
            req.audio_filename,
        )
