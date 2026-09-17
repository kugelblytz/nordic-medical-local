import json
import logging
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
from models import LLMAnswer, LLMAnswerBatch, MedicalFact, TranscriptSegment


log = logging.getLogger(__name__)

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


ENTAILMENT_RULES = '''You verify exact factual entailment in a doctor-patient dialogue.

A question is TRUE only when the transcript explicitly establishes the complete proposition.

For every question compare all material slots:
- person, medicine, diagnosis, test, or other entity
- number, dose, result, quantity, and unit
- frequency
- duration, date, and timing
- body/anatomical location
- symptom, finding, test result, or outcome
- action and status: historical, current, proposed, agreed, started, completed, stopped

Rules:
- Topic overlap is never enough.
- One material mismatch makes the proposition FALSE.
- Proposed/considered/discussed/possible is not the same as agreed, started, or completed.
- Past history is not the current visit unless the question asks about history.
- Respect negation, corrections, uncertainty, and changes of plan. A later correction overrides an earlier statement.
- Do not fill missing facts using medical common sense.
- Numbers, units, medication names, dates, durations, body sites, and frequencies must match exactly in meaning.
- Normal versus abnormal, stable versus unstable, continue versus stop, and positive versus negative are material opposites.

Reason codes:
- exact_support: every material part is explicitly supported
- wrong_entity: wrong person, medicine, test, diagnosis, or entity
- wrong_value: wrong dose, amount, result, quantity, frequency, unit, or polarity/value
- wrong_time: wrong date, duration, timing, or temporal status
- wrong_location: wrong anatomical/body location
- negated: transcript explicitly says the proposition is not true
- contradicted: transcript establishes an incompatible fact
- discussed_not_done: merely discussed/proposed/considered rather than done/agreed
- not_explicit: insufficient explicit support
- off_topic: unrelated to the conversation

Examples of the required strictness:
- Transcript says "100 mg once daily"; question says "200 mg once daily" -> FALSE, wrong_value.
- Transcript says "for two weeks"; question says "for six weeks" -> FALSE, wrong_time.
- Transcript says "we could consider starting it later"; question says it was started -> FALSE, discussed_not_done.
- Transcript says "lungs are clear"; question says abnormal lung sounds were found -> FALSE, contradicted.
'''


def _client(timeout_seconds: float = OLLAMA_TIMEOUT) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            timeout_seconds,
            connect=min(5.0, timeout_seconds),
        )
    )


def _render_transcript(segments: list[TranscriptSegment]) -> str:
    return '\n'.join(
        f'[S{s.id} {s.start:.2f}-{s.end:.2f}] {s.text}'
        for s in segments
    )


def _numbered_questions(questions: list[str]) -> str:
    return '\n'.join(f'{i}. {q}' for i, q in enumerate(questions))


def _fact_schema() -> dict:
    return {
        'type': 'object',
        'properties': {
            'facts': {
                'type': 'array',
                'maxItems': 32,
                'items': {
                    'type': 'object',
                    'properties': {
                        'fact': {'type': 'string'},
                        'segment_ids': {
                            'type': 'array',
                            'items': {'type': 'integer', 'minimum': 0},
                            'minItems': 1,
                            'maxItems': 2,
                        },
                    },
                    'required': ['fact', 'segment_ids'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['facts'],
        'additionalProperties': False,
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


def _chat_json(
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    num_predict: int,
) -> dict:
    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'think': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'format': schema,
        'options': {
            'temperature': 0,
            'num_predict': num_predict,
            'num_ctx': OLLAMA_NUM_CTX,
        },
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    }

    with _client() as client:
        response = client.post(f'{OLLAMA_URL}/api/chat', json=payload)
        response.raise_for_status()
        return json.loads(response.json()['message']['content'])


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


def _extract_fact_ledger(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> list[MedicalFact]:
    transcript = _render_transcript(segments)
    numbered = _numbered_questions(questions)

    prompt = f'''TRANSCRIPT:
{transcript}

QUESTIONS:
{numbered}

Build a compact ledger of explicit facts that are useful for deciding these questions.

Extract exact facts, including exact numbers/units, medication names, durations, dates, body locations, findings, negations, and whether an action was only discussed versus agreed/started/completed.

Important:
- Record what the transcript actually says, not what a question suggests.
- Preserve opposites explicitly: normal/abnormal, stable/unstable, continue/stop, positive/negative.
- If the transcript corrects itself, record the final corrected fact.
- Each fact must cite the smallest 1 or 2 transcript segment IDs that establish it.
- Do not answer the questions in this pass.
'''

    raw = _chat_json(
        system_prompt=(
            'You extract a precise factual ledger from medical dialogue. '
            'Never infer missing facts.'
        ),
        user_prompt=prompt,
        schema=_fact_schema(),
        num_predict=1200,
    )

    max_segment_id = len(segments) - 1
    facts = []
    for raw_fact in raw.get('facts', []):
        fact = MedicalFact.model_validate(raw_fact)
        fact.segment_ids = [
            sid for sid in fact.segment_ids
            if 0 <= sid <= max_segment_id
        ][:2]
        if fact.fact.strip() and fact.segment_ids:
            facts.append(fact)

    return facts[:32]


def _render_facts(facts: list[MedicalFact]) -> str:
    if not facts:
        return '(No helper facts available; use the transcript directly.)'

    lines = []
    for i, fact in enumerate(facts):
        ids = ','.join(f'S{sid}' for sid in fact.segment_ids)
        lines.append(f'F{i} [{ids}] {fact.fact}')
    return '\n'.join(lines)


def _normalize_answers(
    raw: dict,
    segments: list[TranscriptSegment],
    question_count: int,
) -> LLMAnswerBatch:
    expected_keys = [str(i) for i in range(question_count)]
    if set(raw.keys()) != set(expected_keys):
        raise ValueError(
            f'LLM returned keys {sorted(raw.keys())}; expected {expected_keys}'
        )

    max_segment_id = len(segments) - 1
    answers = []

    for i in range(question_count):
        item = LLMAnswer.model_validate(raw[str(i)])

        # The semantic reason is authoritative. This also catches inconsistent
        # booleans produced by the model.
        item.answer = item.reason_code == 'exact_support'

        if item.answer:
            item.evidence_segment_ids = [
                sid for sid in item.evidence_segment_ids
                if 0 <= sid <= max_segment_id
            ][:2]
        else:
            item.evidence_segment_ids = []
            item.evidence_quote = ''

        answers.append(item)

    return LLMAnswerBatch(answers=answers)


def _classify_questions(
    segments: list[TranscriptSegment],
    questions: list[str],
    facts: list[MedicalFact],
) -> LLMAnswerBatch:
    transcript = _render_transcript(segments)
    numbered = _numbered_questions(questions)
    ledger = _render_facts(facts)

    prompt = f'''TRANSCRIPT:
{transcript}

HELPER FACT LEDGER:
{ledger}

QUESTIONS:
{numbered}

Classify every question independently against the transcript. The helper ledger may be incomplete; the transcript is authoritative.

For each question:
1. Identify the proposition's material slots.
2. Find the most relevant transcript evidence.
3. Compare every material slot exactly.
4. Choose one reason_code.
5. answer is TRUE only for exact_support.

For TRUE answers:
- return the smallest 1 or 2 supporting segment IDs;
- copy an exact spoken quote from those segments.

For FALSE answers:
- evidence_segment_ids = []
- evidence_quote = ""

Return exactly the keys {", ".join(str(i) for i in range(len(questions)))}.
'''

    raw = _chat_json(
        system_prompt=ENTAILMENT_RULES,
        user_prompt=prompt,
        schema=_answer_schema(len(questions)),
        num_predict=1800,
    )
    return _normalize_answers(raw, segments, len(questions))


def _render_draft(
    questions: list[str],
    draft: LLMAnswerBatch,
) -> str:
    lines = []
    for i, (question, item) in enumerate(zip(questions, draft.answers)):
        evidence = ','.join(f'S{x}' for x in item.evidence_segment_ids) or '-'
        lines.append(
            f'{i}. {question}\n'
            f'   DRAFT: {"TRUE" if item.answer else "FALSE"}; '
            f'reason={item.reason_code}; evidence={evidence}; '
            f'quote={item.evidence_quote!r}'
        )
    return '\n'.join(lines)


def _verify_answers(
    segments: list[TranscriptSegment],
    questions: list[str],
    facts: list[MedicalFact],
    draft: LLMAnswerBatch,
) -> LLMAnswerBatch:
    transcript = _render_transcript(segments)
    ledger = _render_facts(facts)
    draft_text = _render_draft(questions, draft)

    prompt = f'''TRANSCRIPT:
{transcript}

HELPER FACT LEDGER:
{ledger}

DRAFT JUDGMENTS:
{draft_text}

Act as an adversarial verifier. The draft judgments are hypotheses, not authority.

Re-decide ALL questions from the transcript.

Verification procedure:
1. For every draft TRUE, actively try to falsify it by finding any mismatch in entity, dose/value/unit, frequency, duration/date/time, body location, polarity, negation, or action/status.
2. For every draft FALSE, check whether there is direct exact support that the first pass missed. Rescue it to TRUE only when the whole proposition is explicitly established.
3. Compare questions with each other for likely contrast pairs. If two claims differ only by a material value or opposite status, do not allow both to be TRUE unless the transcript explicitly establishes both in their respective contexts.
4. Treat normal/abnormal, stable/unstable, continue/discontinue, present/absent, and proposed/completed as meaningful opposites.
5. The transcript is authoritative. Do not preserve a draft answer merely for consistency.

For final TRUE answers, return the smallest 1 or 2 supporting segment IDs and an exact spoken quote.
For final FALSE answers, return no evidence.

Return exactly the keys {", ".join(str(i) for i in range(len(questions)))}.
'''

    raw = _chat_json(
        system_prompt=ENTAILMENT_RULES,
        user_prompt=prompt,
        schema=_answer_schema(len(questions)),
        num_predict=1800,
    )
    return _normalize_answers(raw, segments, len(questions))


def answer_questions(
    segments: list[TranscriptSegment],
    questions: list[str],
) -> LLMAnswerBatch:
    started = time.perf_counter()

    facts: list[MedicalFact] = []
    try:
        fact_started = time.perf_counter()
        facts = _extract_fact_ledger(segments, questions)
        log.info(
            'Fact ledger: %d facts in %.2fs',
            len(facts),
            time.perf_counter() - fact_started,
        )
    except Exception:
        # The ledger is a reasoning aid, not a dependency. Classification can
        # still use the transcript directly.
        log.exception('Fact-ledger extraction failed; continuing without it')

    last_error = None
    draft = None
    for attempt in range(2):
        try:
            classify_started = time.perf_counter()
            draft = _classify_questions(segments, questions, facts)
            log.info(
                'Initial classification completed in %.2fs',
                time.perf_counter() - classify_started,
            )
            break
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

    if draft is None:
        raise RuntimeError(
            f'LLM classification failed twice: {last_error}'
        ) from last_error

    try:
        verify_started = time.perf_counter()
        verified = _verify_answers(segments, questions, facts, draft)
        log.info(
            'Adversarial verification completed in %.2fs; total LLM %.2fs',
            time.perf_counter() - verify_started,
            time.perf_counter() - started,
        )
        return verified
    except Exception:
        # A verifier failure must never erase a valid first-pass result.
        log.exception('Verifier failed; using initial classification')
        return draft
