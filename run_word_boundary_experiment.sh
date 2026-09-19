#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:27b}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-90}"

export ASR_MODEL="${ASR_MODEL:-large-v3-turbo}"
export ASR_DEVICE="${ASR_DEVICE:-cuda}"
export ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-float16}"

# Clean first A/B: direct word timestamps with no added padding.
export EVIDENCE_WORD_PAD_SECONDS="${EVIDENCE_WORD_PAD_SECONDS:-0.00}"
export EVIDENCE_MAX_WORD_SPAN="${EVIDENCE_MAX_WORD_SPAN:-80}"

exec bash vast_start.sh
