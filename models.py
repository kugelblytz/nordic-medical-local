from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ASRQuestionRequestDto(BaseModel):
    audio_base64: str
    audio_filename: str
    questions: List[str]


class ASRQuestionResponseDto(BaseModel):
    answers: List[bool]
    evidence_start: List[Optional[float]]
    evidence_end: List[Optional[float]]

    @model_validator(mode='after')
    def validate_lengths(self):
        n = len(self.answers)
        if len(self.evidence_start) != n or len(self.evidence_end) != n:
            raise ValueError('All response arrays must have the same length')
        return self


class WordToken(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: List[WordToken] = Field(default_factory=list)


class MedicalFact(BaseModel):
    fact: str
    segment_ids: List[int] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    answer: bool
    reason_code: str = 'not_explicit'
    evidence_segment_ids: List[int] = Field(default_factory=list)
    evidence_quote: str = ''


class LLMAnswerBatch(BaseModel):
    answers: List[LLMAnswer]
