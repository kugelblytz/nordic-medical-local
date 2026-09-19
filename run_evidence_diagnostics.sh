#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export DIAGNOSTICS_ENABLED=true
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-$ROOT/.diagnostics/word-boundary-v1}"

mkdir -p "$DIAGNOSTICS_DIR"

echo "Evidence diagnostics enabled"
echo "  output: $DIAGNOSTICS_DIR"
echo
echo "Existing conversation JSON files with the same audio filename will be overwritten."
echo "Delete the directory first if you want a completely clean run."
echo

exec bash run_word_boundary_experiment.sh
