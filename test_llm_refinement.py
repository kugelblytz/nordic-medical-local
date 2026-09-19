import json

import llm
from models import LLMAnswer, LLMAnswerBatch, TranscriptSegment, WordToken


def segments():
    return [
        TranscriptSegment(
            id=0,
            start=0.0,
            end=2.0,
            text='alpha beta gamma',
            words=[
                WordToken(text='alpha', start=0.1, end=0.4),
                WordToken(text='beta', start=0.5, end=0.8),
                WordToken(text='gamma', start=0.9, end=1.2),
            ],
        )
    ]


def test_refiner_skips_when_no_true_answers(monkeypatch):
    batch = LLMAnswerBatch(
        answers=[
            LLMAnswer(answer=False, reason_code='wrong_value'),
            LLMAnswer(answer=False, reason_code='off_topic'),
        ]
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError('Ollama should not be called')

    monkeypatch.setattr(llm, '_run_evidence_refinement', should_not_run)

    assert llm.refine_evidence(segments(), ['q0', 'q1'], batch) == {}


def test_run_refinement_uses_original_question_indexes(monkeypatch):
    batch = LLMAnswerBatch(
        answers=[
            LLMAnswer(answer=False, reason_code='wrong_value'),
            LLMAnswer(
                answer=True,
                reason_code='exact_support',
                evidence_segment_ids=[0],
                evidence_start_word_id=0,
                evidence_end_word_id=0,
                evidence_quote='alpha',
            ),
            LLMAnswer(
                answer=True,
                reason_code='exact_support',
                evidence_segment_ids=[0],
                evidence_start_word_id=1,
                evidence_end_word_id=1,
                evidence_quote='beta',
            ),
        ]
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'total_duration': 2_000_000_000,
                'load_duration': 0,
                'prompt_eval_count': 1000,
                'prompt_eval_duration': 500_000_000,
                'eval_count': 100,
                'eval_duration': 1_000_000_000,
                'message': {
                    'content': json.dumps(
                        {
                            '1': {
                                'start_word_id': 1,
                                'end_word_id': 2,
                            },
                            '2': {
                                'start_word_id': 2,
                                'end_word_id': 2,
                            },
                        }
                    )
                }
            }

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            assert set(json['format']['required']) == {'1', '2'}
            return FakeResponse()

    monkeypatch.setattr(llm, '_client', lambda *args, **kwargs: FakeClient())

    attempt_timing = {}
    result = llm._run_evidence_refinement(
        segments(),
        ['q0', 'q1', 'q2'],
        batch,
        attempt_timing=attempt_timing,
    )

    assert set(result.keys()) == {1, 2}
    assert result[1].start_word_id == 1
    assert result[1].end_word_id == 2
    assert result[2].start_word_id == 2
    assert result[2].end_word_id == 2
    assert attempt_timing['ollama']['total_ms'] == 2000.0
    assert attempt_timing['ollama']['prompt_tokens_per_s'] == 2000.0
    assert attempt_timing['ollama']['output_tokens_per_s'] == 100.0
