#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo "Run: export OPENAI_API_KEY='your-key'" >&2
  exit 1
fi

export LLM_PROVIDER=openai
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6-sol}"
export OPENAI_REASONING="${OPENAI_REASONING:-high}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-50}"
export OPENAI_MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-8000}"
export ASR_MODEL="${ASR_MODEL:-large-v3-turbo}"

exec bash vast_start.sh
