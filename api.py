import logging

import uvicorn
from fastapi import FastAPI

from asr import warmup as warmup_asr
from config import HOST, LLM_PROVIDER, PORT, WARMUP_ON_START
from llm import warmup as warmup_llm
from models import ASRQuestionRequestDto, ASRQuestionResponseDto
from predictor import predict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title='Nordic AI Cup Medical Appointment - Local Starter')


@app.on_event('startup')
def startup() -> None:
    if not WARMUP_ON_START:
        return

    log.info('Loading Whisper...')
    warmup_asr()
    log.info('Initializing %s LLM backend...', LLM_PROVIDER)
    warmup_llm()
    log.info('Models ready.')


@app.get('/')
def root():
    return {'status': 'ok'}


@app.get('/api')
def api_status():
    return {'service': 'medical-appointment-usecase', 'status': 'ok', 'llm_provider': LLM_PROVIDER}


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(req: ASRQuestionRequestDto):
    response = predict(req)

    expected = len(req.questions)
    if not (
        len(response.answers) == expected
        and len(response.evidence_start) == expected
        and len(response.evidence_end) == expected
    ):
        raise RuntimeError('predict() returned arrays with incorrect lengths')

    return response


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
