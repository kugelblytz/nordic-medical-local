import os

ASR_MODEL = os.getenv('ASR_MODEL', 'large-v3-turbo')
ASR_DEVICE = os.getenv('ASR_DEVICE', 'cuda')
ASR_COMPUTE_TYPE = os.getenv('ASR_COMPUTE_TYPE', 'float16')
ASR_LANGUAGE = os.getenv('ASR_LANGUAGE', 'en')

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '35'))

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '9054'))
