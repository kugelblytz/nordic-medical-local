import logging
from fastapi import FastAPI
import uvicorn
from config import HOST, PORT
from models import ASRQuestionRequestDto, ASRQuestionResponseDto
from predictor import predict

logging.basicConfig(level=logging.INFO)
app = FastAPI(title='Nordic AI Cup Medical Appointment - Local Starter')


@app.get('/')
def root():
    return {'status': 'ok'}


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(req: ASRQuestionRequestDto):
    return predict(req)


if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
