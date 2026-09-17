import json
import httpx
from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from models import TranscriptSegment, LLMAnswerBatch

SYSTEM_PROMPT = '''You are solving a medical-dialogue evidence verification task.
For every question, decide whether the proposition is explicitly established by the transcript.

Rules:
- Return true ONLY when the transcript supports the proposition.
- A near miss is false: wrong dose, duration, date, drug, body location, quantity, frequency, unit, or outcome is false.
- Distinguish discussion, possibility, prior history, and a plan that was actually agreed.
- Pay close attention to negation and corrections.
- Do not infer facts that are merely plausible.
- For each true answer, copy the SHORTEST exact contiguous quote from the transcript that is sufficient to prove it.
- For false answers, evidence_quote must be an empty string.
- Preserve question order.
- Output JSON only.
'''


def _render_transcript(segments: list[TranscriptSegment]) -> str:
    return '\n'.join(f'[{s.start:.2f}-{s.end:.2f}] {s.text}' for s in segments)


def answer_questions(segments: list[TranscriptSegment], questions: list[str]) -> LLMAnswerBatch:
    transcript = _render_transcript(segments)
    numbered = '\n'.join(f'{i}. {q}' for i, q in enumerate(questions))
    schema_example = {
        'answers': [
            {'answer': True, 'evidence_quote': 'exact words copied from transcript'},
            {'answer': False, 'evidence_quote': ''},
        ]
    }

    user_prompt = f'''TRANSCRIPT:\n{transcript}\n\nQUESTIONS:\n{numbered}\n\nReturn exactly {len(questions)} answers in this shape:\n{json.dumps(schema_example)}'''

    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0, 'num_predict': 1800},
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
    }

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        r = client.post(f'{OLLAMA_URL}/api/chat', json=payload)
        r.raise_for_status()
        content = r.json()['message']['content']

    batch = LLMAnswerBatch.model_validate(json.loads(content))
    if len(batch.answers) != len(questions):
        raise ValueError(f'LLM returned {len(batch.answers)} answers for {len(questions)} questions')
    return batch
