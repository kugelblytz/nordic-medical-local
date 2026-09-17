import os
import tempfile
from faster_whisper import WhisperModel
from config import ASR_MODEL, ASR_DEVICE, ASR_COMPUTE_TYPE, ASR_LANGUAGE
from models import TranscriptSegment, WordToken


MODEL = WhisperModel(
    ASR_MODEL,
    device=ASR_DEVICE,
    compute_type=ASR_COMPUTE_TYPE,
)


def warmup() -> None:
    pass


def transcribe(audio_bytes: bytes) -> list[TranscriptSegment]:
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    try:
        segments_iter, _ = MODEL.transcribe(
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
