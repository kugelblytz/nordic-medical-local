import os
import tempfile
from typing import Optional

from faster_whisper import WhisperModel

from config import ASR_MODEL, ASR_DEVICE, ASR_COMPUTE_TYPE, ASR_LANGUAGE
from models import TranscriptSegment, WordToken


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


def transcribe(audio_bytes: bytes) -> list[TranscriptSegment]:
    model = load_model()

    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    try:
        segments_iter, _ = model.transcribe(
            path,
            language=ASR_LANGUAGE,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        result: list[TranscriptSegment] = []
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
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
