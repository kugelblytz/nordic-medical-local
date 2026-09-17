#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-9054}"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "Ollama:"
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
echo "OK"

echo
echo "Competition API:"
curl -fsS "http://127.0.0.1:${PORT}/api"
echo
