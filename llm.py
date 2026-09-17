import json
import time

import httpx

from config import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    OLLAMA_WARMUP_TIMEOUT,
)
from models import LLMAnswer, LLMAnswerBatch, TranscriptSegment


SYSTEM_PROMPT = '''You are solving a medical-dialogue evidence verification task.
For every question, decide whether the proposition is explicitly established by the transcript.

Rules:
- Return true ONLY when the transcript supports the proposition.
- A near miss is false: wrong dose, duration, date, drug, body location, quantity, frequency, unit, or outcome is false.
- Distinguish discussion, possibility, prior history, and a plan that was actually agreed.
- Pay close attention to negation and corrections.
- Do not infer facts that are merely plausible.
- For each true answer, copy the smallest COMPLETE spoken sentence or utterance that proves it.
- evidence_quote must contain spoken transcript words only. Never copy the [start-end] timestamp prefix.
- For false answers, evidence_quote must be an empty string.
- Return one answer for EVERY numbered question.
- Preserve question numbering exactly.
- Output JSON only.
'''


def _client(timeout_seconds: float = OLLAMA_TIMEOUT) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            timeout_seconds,
            connect=min(5.0, timeout_seconds),
        )
    )


def _render_transcript(segments: list[TranscriptSegment]) -> str:
    return '\n'.join(f'[{s.start:.2f}-{s.end:.2f}] {s.text}' for s in segments)


def _answer_schema(question_count: int) -> dict:
    answer_schema = {
        'type': 'object',
        'properties': {
            'answer': {'type': 'boolean'},
            'evidence_quote': {'type': 'string'},
        },
        'required': ['answer', 'evidence_quote'],
        'additionalProperties': False,
    }

    properties = {
        str(i): answer_schema
        for i in range(question_count)
    }

    return {
        'type': 'object',
        'properties': properties,
        'required': list(properties.keys()),
        'additionalProperties': False,
    }


def warmup() -> None:
    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'think': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'format': {
            'type': 'object',
            'properties': {'ok': {'type': 'boolean'}},
            'required': ['ok'],
            'additionalProperties': False,
        },
        'options': {
            'temperature': 0,
            'num_predict': 16,
            'num_ctx': OLLAMA_NUM_CTX,
        },
        'messages': [{'role': 'user', 'content': 'Return {"ok": true}.'}],
    }

    last_error = None
    for attempt in range(1, 3):
        try:
            with _client(OLLAMA_WARMUP_TIMEOUT) as client:
                response = client.post(f'{OLLAMA_URL}/api/chat', json=payload)
                response.raise_for_status()
            return
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2)

    raise RuntimeError(
        f'Ollama warmup failed after 2 attempts: {last_error}'
    ) from last_error


def answer_questions(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> LLMAnswerBatch:
    transcript = _render_transcript(segments)
    numbered = '\n'.join(f'{i}. {q}' for i, q in enumerate(questions))

    user_prompt = f'''TRANSCRIPT:
{transcript}

QUESTIONS:
{numbered}

Return a JSON object with exactly these keys:
{", ".join(str(i) for i in range(len(questions)))}

Each key must contain exactly:
- answer: boolean
- evidence_quote: string

Do not omit any key.'''

    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'think': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'format': _answer_schema(len(questions)),
        'options': {
            'temperature': 0,
            'num_predict': 1800,
            'num_ctx': OLLAMA_NUM_CTX,
        },
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
    }

    with _client() as client:
        response = client.post(f'{OLLAMA_URL}/api/chat', json=payload)
        response.raise_for_status()
        content = response.json()['message']['content']

    raw = json.loads(content)

    expected_keys = [str(i) for i in range(len(questions))]
    if set(raw.keys()) != set(expected_keys):
        raise ValueError(
            f'LLM returned keys {sorted(raw.keys())}; expected {expected_keys}'
        )

    answers = [
        LLMAnswer.model_validate(raw[str(i)])
        for i in range(len(questions))
    ]
    return LLMAnswerBatch(answers=answers)
