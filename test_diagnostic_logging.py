import json

import diagnostic_logging
from diagnostic_logging import write_conversation_diagnostics
from models import (
    ASRQuestionRequestDto,
    LLMAnswer,
    LLMAnswerBatch,
    TranscriptSegment,
    WordToken,
)


def test_diagnostic_logger_writes_conversation_json(tmp_path):
    diagnostic_logging.DIAGNOSTICS_ENABLED = True
    diagnostic_logging.DIAGNOSTICS_DIR = str(tmp_path)

    req = ASRQuestionRequestDto(
        audio_base64='',
        audio_filename='conversation_sample_4.mp3',
        questions=['Was the dose 100 mg?'],
    )
    segments = [
        TranscriptSegment(
            id=0,
            start=1.0,
            end=2.0,
            text='Take 100 mg.',
            words=[
                WordToken(text='Take', start=1.0, end=1.2),
                WordToken(text='100', start=1.2, end=1.5),
                WordToken(text='mg.', start=1.5, end=1.8),
            ],
        )
    ]
    batch = LLMAnswerBatch(
        answers=[
            LLMAnswer(
                answer=True,
                reason_code='exact_support',
                evidence_segment_ids=[0],
                evidence_quote='100 mg',
                evidence_start_word_id=1,
                evidence_end_word_id=2,
            )
        ]
    )

    write_conversation_diagnostics(
        req,
        segments,
        batch,
        [1.2],
        [1.8],
        ['word_ids'],
    )

    path = tmp_path / 'conversation_sample_4.json'
    payload = json.loads(path.read_text(encoding='utf-8'))

    assert payload['schema_version'] == 2
    assert payload['audio_filename'] == 'conversation_sample_4.mp3'
    assert payload['words'][1]['id'] == 1
    assert payload['questions'][0]['selected_word_text'] == '100 mg.'
    assert payload['questions'][0]['evidence_strategy'] == 'word_ids'
    assert payload['questions'][0]['first_pass_resolved_evidence_start'] == 1.2
    assert payload['questions'][0]['refinement'] is None
