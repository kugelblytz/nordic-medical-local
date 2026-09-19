import base64

import predictor
from models import (
    ASRQuestionRequestDto,
    LLMAnswer,
    LLMAnswerBatch,
    RefinedEvidence,
    TranscriptSegment,
    WordToken,
)


def sample_segments():
    return [
        TranscriptSegment(
            id=0,
            start=0.0,
            end=4.0,
            text='alpha beta gamma delta epsilon zeta',
            words=[
                WordToken(text='alpha', start=0.1, end=0.4),
                WordToken(text='beta', start=0.5, end=0.8),
                WordToken(text='gamma', start=0.9, end=1.2),
                WordToken(text='delta', start=1.3, end=1.6),
                WordToken(text='epsilon', start=1.7, end=2.0),
                WordToken(text='zeta', start=2.1, end=2.4),
            ],
        )
    ]


def sample_batch():
    return LLMAnswerBatch(
        answers=[
            LLMAnswer(
                answer=True,
                reason_code='exact_support',
                evidence_segment_ids=[0],
                evidence_quote='alpha',
                evidence_start_word_id=0,
                evidence_end_word_id=0,
            ),
            LLMAnswer(
                answer=False,
                reason_code='wrong_value',
            ),
            LLMAnswer(
                answer=True,
                reason_code='exact_support',
                evidence_segment_ids=[0],
                evidence_quote='delta',
                evidence_start_word_id=3,
                evidence_end_word_id=3,
            ),
        ]
    )


def request():
    return ASRQuestionRequestDto(
        audio_base64=base64.b64encode(b'x').decode(),
        audio_filename='conversation_test.mp3',
        questions=['q0', 'q1', 'q2'],
    )


def test_refinement_changes_only_evidence(monkeypatch):
    monkeypatch.setattr(predictor, 'EVIDENCE_REFINEMENT_ENABLED', True)
    monkeypatch.setattr(predictor, 'transcribe', lambda _, timing=None: sample_segments())
    monkeypatch.setattr(
        predictor,
        'answer_questions',
        lambda segments, questions, timing=None: sample_batch(),
    )
    monkeypatch.setattr(
        predictor,
        'refine_evidence',
        lambda segments, questions, batch, timing=None: {
            0: RefinedEvidence(start_word_id=1, end_word_id=2),
            2: RefinedEvidence(start_word_id=4, end_word_id=5),
        },
    )
    monkeypatch.setattr(
        predictor,
        'write_conversation_diagnostics',
        lambda *args, **kwargs: None,
    )

    response = predictor.predict(request())

    assert response.answers == [True, False, True]
    assert response.evidence_start == [0.5, None, 1.7]
    assert response.evidence_end == [1.2, None, 2.4]


def test_invalid_refinement_preserves_first_pass(monkeypatch):
    monkeypatch.setattr(predictor, 'EVIDENCE_REFINEMENT_ENABLED', True)
    monkeypatch.setattr(predictor, 'transcribe', lambda _, timing=None: sample_segments())
    monkeypatch.setattr(
        predictor,
        'answer_questions',
        lambda segments, questions, timing=None: sample_batch(),
    )
    monkeypatch.setattr(
        predictor,
        'refine_evidence',
        lambda segments, questions, batch, timing=None: {
            0: RefinedEvidence(start_word_id=1, end_word_id=999),
            2: RefinedEvidence(start_word_id=4, end_word_id=5),
        },
    )
    monkeypatch.setattr(
        predictor,
        'write_conversation_diagnostics',
        lambda *args, **kwargs: None,
    )

    response = predictor.predict(request())

    assert response.answers == [True, False, True]
    assert response.evidence_start[0] == 0.1
    assert response.evidence_end[0] == 0.4
    assert response.evidence_start[2] == 1.7
    assert response.evidence_end[2] == 2.4


def test_refiner_exception_preserves_classification_and_first_pass(monkeypatch):
    monkeypatch.setattr(predictor, 'EVIDENCE_REFINEMENT_ENABLED', True)
    monkeypatch.setattr(predictor, 'transcribe', lambda _, timing=None: sample_segments())
    monkeypatch.setattr(
        predictor,
        'answer_questions',
        lambda segments, questions, timing=None: sample_batch(),
    )

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic refiner failure')

    monkeypatch.setattr(predictor, 'refine_evidence', fail)
    monkeypatch.setattr(
        predictor,
        'write_conversation_diagnostics',
        lambda *args, **kwargs: None,
    )

    response = predictor.predict(request())

    assert response.answers == [True, False, True]
    assert response.evidence_start == [0.1, None, 1.3]
    assert response.evidence_end == [0.4, None, 1.6]
