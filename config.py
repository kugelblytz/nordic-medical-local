import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


ASR_MODEL = os.getenv('ASR_MODEL', 'large-v3-turbo')
ASR_DEVICE = os.getenv('ASR_DEVICE', 'cuda')
ASR_COMPUTE_TYPE = os.getenv('ASR_COMPUTE_TYPE', 'float16')
ASR_LANGUAGE = os.getenv('ASR_LANGUAGE', 'en')

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '35'))
OLLAMA_KEEP_ALIVE = os.getenv('OLLAMA_KEEP_ALIVE', '30m')
OLLAMA_NUM_CTX = int(os.getenv('OLLAMA_NUM_CTX', '8192'))

EVIDENCE_MODE = os.getenv('EVIDENCE_MODE', 'segment').strip().lower()
EVIDENCE_PAD_SECONDS = float(os.getenv('EVIDENCE_PAD_SECONDS', '0.20'))

WARMUP_ON_START = _env_bool('WARMUP_ON_START', True)

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '9054'))
