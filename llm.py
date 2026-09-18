import json
import time

import httpx
from openai import OpenAI, OpenAIError

from config import (
    LLM_PROVIDER,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    OLLAMA_WARMUP_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_MODEL,
    OPENAI_REASONING,
    OPENAI_TIMEOUT,
)
from models import LLMAnswer, LLMAnswerBatch, TranscriptSegment


REASON_CODES = [
    'exact_support',
    'wrong_entity',
    'wrong_value',
    'wrong_time',
    'wrong_location',
    'negated',
    'contradicted',
    'discussed_not_done',
    'not_explicit',
    'off_topic',
]


SYSTEM_PROMPT = '''You verify exact factual entailment in a doctor-patient dialogue.

Each question is a proposition. Mark it TRUE only if the transcript explicitly establishes the whole proposition.

Before deciding, compare every material slot in the question with the transcript:
- person/entity or medicine
- dose/number and unit
- frequency
- duration
- date/time
- body location
- symptom/test/result
- treatment/action/status

Critical rules:
- Return TRUE when the proposition is explicitly supported, even if wording differs.
- Topic overlap alone is not enough.
- If even one material slot differs, the answer is FALSE.
- A proposed, considered, discussed, possible, or hypothetical action is not the same as an action that actually happened or was agreed.
- Past history is not the same as the current visit unless the question asks about history.
- Pay close attention to negation, corrections, and changes of plan.
- Do not infer facts that are merely plausible.
- Numbers, doses, units, dates, medication names, body sites, and frequencies must match exactly in meaning.
- Normal/abnormal, stable/unstable, continue/stop, present/absent, and positive/negative are meaningful opposites.
- Do not become overly conservative: if all material parts are explicitly supported, answer TRUE.

For every question return one reason_code:
- exact_support: every material part is explicitly supported
- wrong_entity: wrong person, medicine, test, diagnosis, or other entity
- wrong_value: wrong dose, amount, result, quantity, frequency, unit, or other value
- wrong_time: wrong date, duration, timing, or temporal status
- wrong_location: wrong body/anatomical location
- negated: transcript explicitly says the proposition is not true
- contradicted: transcript establishes an incompatible fact
- discussed_not_done: merely discussed/proposed/considered rather than actually done/agreed
- not_explicit: transcript does not explicitly establish enough to say yes
- off_topic: unrelated to the conversation

answer must be TRUE if and only if reason_code is exact_support.

Evidence rules for TRUE answers:
- Transcript utterances are labelled S0, S1, ...
- Return the smallest 1 or 2 segment IDs that directly establish the proposition.
- Copy an exact supporting spoken quote from those segment(s).
- Do not include timestamps or segment labels in evidence_quote.

For FALSE answers:
- evidence_segment_ids must be []
- evidence_quote must be ""

Return one result for EVERY numbered question. Never omit a question. Output JSON only.
'''


def _ollama_client(timeout_seconds: float = OLLAMA_TIMEOUT) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            timeout_seconds,
            connect=min(5.0, timeout_seconds),
        )
    )


def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            'LLM_PROVIDER=openai requires OPENAI_API_KEY in the environment'
        )
    return OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT,
        max_retries=1,
    )


def _render_transcript(segments: list[TranscriptSegment]) -> str:
    return '\n'.join(
        f'[S{s.id} {s.start:.2f}-{s.end:.2f}] {s.text}'
        for s in segments
    )


def _answer_schema(question_count: int) -> dict:
    answer_schema = {
        'type': 'object',
        'properties': {
            'answer': {'type': 'boolean'},
            'reason_code': {
                'type': 'string',
                'enum': REASON_CODES,
            },
            'evidence_segment_ids': {
                'type': 'array',
                'items': {'type': 'integer', 'minimum': 0},
                'maxItems': 2,
            },
            'evidence_quote': {'type': 'string'},
        },
        'required': [
            'answer',
            'reason_code',
            'evidence_segment_ids',
            'evidence_quote',
        ],
        'additionalProperties': False,
    }

    properties = {str(i): answer_schema for i in range(question_count)}

    return {
        'type': 'object',
        'properties': properties,
        'required': list(properties.keys()),
        'additionalProperties': False,
    }


def warmup() -> None:
    if LLM_PROVIDER == 'openai':
        _openai_client()
        return

    if LLM_PROVIDER != 'ollama':
        raise ValueError(f'Unsupported LLM_PROVIDER={LLM_PROVIDER!r}')

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
            with _ollama_client(OLLAMA_WARMUP_TIMEOUT) as client:
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


def _build_user_prompt(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> str:
    transcript = _render_transcript(segments)
    numbered = '\n'.join(f'{i}. {q}' for i, q in enumerate(questions))

    return f'''TRANSCRIPT:
{transcript}

QUESTIONS:
{numbered}

Return a JSON object with exactly these keys:
{", ".join(str(i) for i in range(len(questions)))}

For each question, compare EVERY material factual slot against the transcript before deciding. Choose one reason_code. answer must agree with reason_code: only exact_support is true.

Do not omit any key.'''


def _run_batch_ollama(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> dict:
    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'think': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'format': _answer_schema(len(questions)),
        'options': {
            'temperature': 0,
            'num_predict': 2200,
            'num_ctx': OLLAMA_NUM_CTX,
        },
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _build_user_prompt(segments, questions)},
        ],
    }

    with _ollama_client() as client:
        response = client.post(f'{OLLAMA_URL}/api/chat', json=payload)
        response.raise_for_status()
        return json.loads(response.json()['message']['content'])


def _run_batch_openai(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> dict:
    response = _openai_client().responses.create(
        model=OPENAI_MODEL,
        reasoning={'effort': OPENAI_REASONING},
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        store=False,
        input=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _build_user_prompt(segments, questions)},
        ],
        text={
            'format': {
                'type': 'json_schema',
                'name': 'medical_appointment_answers',
                'strict': True,
                'schema': _answer_schema(len(questions)),
            }
        },
    )

    if not response.output_text:
        raise ValueError('OpenAI Responses API returned no output_text')

    return json.loads(response.output_text)


def _normalize_batch(
    raw: dict,
    segments: list[TranscriptSegment],
    questions: list[str],
) -> LLMAnswerBatch:
    expected_keys = [str(i) for i in range(len(questions))]
    if set(raw.keys()) != set(expected_keys):
        raise ValueError(
            f'LLM returned keys {sorted(raw.keys())}; expected {expected_keys}'
        )

    valid_segment_ids = {segment.id for segment in segments}
    answers = []

    for i in range(len(questions)):
        item = LLMAnswer.model_validate(raw[str(i)])
        item.answer = item.reason_code == 'exact_support'

        if item.answer:
            item.evidence_segment_ids = [
                sid for sid in item.evidence_segment_ids
                if sid in valid_segment_ids
            ][:2]
        else:
            item.evidence_segment_ids = []
            item.evidence_quote = ''

        answers.append(item)

    return LLMAnswerBatch(answers=answers)


def _run_batch(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> LLMAnswerBatch:
    if LLM_PROVIDER == 'openai':
        raw = _run_batch_openai(segments, questions)
    elif LLM_PROVIDER == 'ollama':
        raw = _run_batch_ollama(segments, questions)
    else:
        raise ValueError(f'Unsupported LLM_PROVIDER={LLM_PROVIDER!r}')

    return _normalize_batch(raw, segments, questions)


def answer_questions(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> LLMAnswerBatch:
    last_error = None
    for attempt in range(2):
        try:
            return _run_batch(segments, questions)
        except (
            ValueError,
            json.JSONDecodeError,
            KeyError,
            httpx.TimeoutException,
            httpx.HTTPError,
            OpenAIError,
        ) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.1)

    raise RuntimeError(
        f'LLM structured output failed twice: {last_error}'
    ) from last_error
