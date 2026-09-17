#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p .vast/logs .vast/ollama

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Rent a Vast NVIDIA GPU instance with CUDA support." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl git python3 python3-venv python3-pip
  else
    echo "ERROR: curl, git, and python3 are required." >&2
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-gpu.txt

if ! command -v ollama >/dev/null 2>&1; then
  echo "== Installing Ollama =="
  curl -fsSL https://ollama.com/install.sh | sh
fi

export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="$ROOT/.vast/ollama"
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_MAX_LOADED_MODELS="1"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:14b}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"
export ASR_MODEL="${ASR_MODEL:-large-v3-turbo}"
export ASR_DEVICE="${ASR_DEVICE:-cuda}"
export ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-float16}"
export HOST="0.0.0.0"
export PORT="${PORT:-9054}"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "== Starting Ollama =="
  nohup ollama serve > .vast/logs/ollama.log 2>&1 &
  echo $! > .vast/ollama.pid
fi

echo -n "Waiting for Ollama"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo
  echo "ERROR: Ollama did not start. See .vast/logs/ollama.log" >&2
  exit 1
fi

echo "== Pulling $OLLAMA_MODEL =="
ollama pull "$OLLAMA_MODEL"

echo "== Starting Nordic Medical API on 0.0.0.0:$PORT =="
echo "Vast will expose this through the host port mapped to container port $PORT."
echo "Logs from Ollama: $ROOT/.vast/logs/ollama.log"
echo
exec ./run_gpu.sh
