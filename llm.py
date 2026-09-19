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
from word_index import build_word_index, render_transcript_with_word_ids


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

The transcript contains segment labels S0, S1, ... and every timestamped Whisper word is prefixed by a global integer word ID in the form 142:word.

Evidence rules for TRUE answers:
- Return the smallest 1 or 2 segment IDs that contain the complete supporting passage.
- Return evidence_start_word_id and evidence_end_word_id for one CONTIGUOUS word range.
- Select the shortest COMPLETE passage that establishes every material part of the proposition, not merely a keyword or isolated value.
- Include all words needed to establish dose, frequency, duration, date/time, body location, symptom/test/result, treatment/action, status, or negation when those details matter.
- A word range may cross one adjacent Whisper segment boundary when the complete evidence requires both segments.
- The chosen word range must lie inside the returned evidence_segment_ids.
- evidence_start_word_id must be less than or equal to evidence_end_word_id.
- Also copy the exact spoken text of the chosen word range into evidence_quote. The quote is retained as a fallback and diagnostic.

For FALSE answers:
- evidence_segment_ids must be []
- evidence_start_word_id must be null
- evidence_end_word_id must be null
- evidence_quote must be ""

Return one result for EVERY numbered question. Never omit a question. Output JSON only.
'''


def _client(timeout_seconds: float = OLLAMA_TIMEOUT) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            timeout_seconds,
            connect=min(5.0, timeout_seconds),
        )
    )


def _render_transcript(segments: list[TranscriptSegment]) -> str:
    return render_transcript_with_word_ids(segments)


def _nullable_word_id_schema() -> dict:
    return {
        'anyOf': [
            {'type': 'integer', 'minimum': 0},
            {'type': 'null'},
        ]
    }


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
            'evidence_start_word_id': _nullable_word_id_schema(),
            'evidence_end_word_id': _nullable_word_id_schema(),
            'evidence_quote': {'type': 'string'},
        },
        'required': [
            'answer',
            'reason_code',
            'evidence_segment_ids',
            'evidence_start_word_id',
            'evidence_end_word_id',
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


def _run_batch(
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

For each question:
1. compare EVERY material factual slot against the transcript;
2. choose exactly one reason_code;
3. answer true only for exact_support;
4. for true answers choose the complete contiguous evidence word range using the numbered word IDs.

Do not omit any key.'''

    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'think': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'format': _answer_schema(len(questions)),
        'options': {
            'temperature': 0,
            'num_predict': 2400,
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

    answers = []
    valid_segment_ids = {segment.id for segment in segments}
    word_count = len(build_word_index(segments))

    for i in range(len(questions)):
        item = LLMAnswer.model_validate(raw[str(i)])
        item.answer = item.reason_code == 'exact_support'

        if item.answer:
            item.evidence_segment_ids = [
                sid
                for sid in item.evidence_segment_ids
                if sid in valid_segment_ids
            ][:2]

            start_id = item.evidence_start_word_id
            end_id = item.evidence_end_word_id
            if (
                start_id is None
                or end_id is None
                or start_id < 0
                or end_id < start_id
                or end_id >= word_count
            ):
                item.evidence_start_word_id = None
                item.evidence_end_word_id = None
        else:
            item.evidence_segment_ids = []
            item.evidence_start_word_id = None
            item.evidence_end_word_id = None
            item.evidence_quote = ''

        answers.append(item)

    return LLMAnswerBatch(answers=answers)


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
        ) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.1)

    raise RuntimeError(
        f'LLM structured output failed twice: {last_error}'
    ) from last_error
