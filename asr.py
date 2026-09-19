import os
import tempfile
from typing import Any, Optional

from faster_whisper import WhisperModel

from config import ASR_COMPUTE_TYPE, ASR_DEVICE, ASR_LANGUAGE, ASR_MODEL
from models import TranscriptSegment, WordToken
from performance_metrics import elapsed_ms, now_ns


_MODEL: Optional[WhisperModel] = None


def load_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(
            ASR_MODEL,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        )
    return _MODEL


def warmup() -> None:
    # Loading here keeps model download/allocation out of the first scored request.
    load_model()


def transcribe(
    audio_bytes: bytes,
    timing: dict[str, Any] | None = None,
) -> list[TranscriptSegment]:
    asr_timing: dict[str, Any] = {}
    stage_started = now_ns()

    load_started = now_ns()
    model = load_model()
    asr_timing['model_load_ms'] = elapsed_ms(load_started)

    write_started = now_ns()
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    asr_timing['temp_file_write_ms'] = elapsed_ms(write_started)

    result: list[TranscriptSegment] = []
    cleanup_ms = 0.0

    try:
        # faster-whisper returns a lazy segment iterator. The expensive ASR
        # work happens during iteration, so this timer intentionally includes
        # both model.transcribe(...) and full iterator materialization.
        whisper_started = now_ns()
        segments_iter, _ = model.transcribe(
            path,
            language=ASR_LANGUAGE,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        for idx, seg in enumerate(segments_iter):
            words = []
            for w in (seg.words or []):
                if w.start is None or w.end is None:
                    continue
                words.append(
                    WordToken(
                        text=w.word.strip(),
                        start=float(w.start),
                        end=float(w.end),
                    )
                )

            result.append(
                TranscriptSegment(
                    id=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
        asr_timing['whisper_materialize_ms'] = elapsed_ms(whisper_started)
    finally:
        cleanup_started = now_ns()
        try:
            os.unlink(path)
        except OSError:
            pass
        cleanup_ms = elapsed_ms(cleanup_started)

    asr_timing['temp_file_cleanup_ms'] = cleanup_ms
    asr_timing['segment_count'] = len(result)
    asr_timing['word_count'] = sum(len(segment.words) for segment in result)

    audio_duration_s = max((segment.end for segment in result), default=0.0)
    asr_timing['audio_duration_s'] = audio_duration_s
    asr_timing['total_ms'] = elapsed_ms(stage_started)
    asr_timing['realtime_factor'] = (
        (asr_timing['total_ms'] / 1000.0) / audio_duration_s
        if audio_duration_s > 0
        else None
    )

    if timing is not None:
        timing['asr'] = asr_timing

    return result
