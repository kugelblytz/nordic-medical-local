#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export EVIDENCE_REFINEMENT_ENABLED=true
export EVIDENCE_REFINEMENT_TIMEOUT="${EVIDENCE_REFINEMENT_TIMEOUT:-90}"
export EVIDENCE_REFINEMENT_NUM_PREDICT="${EVIDENCE_REFINEMENT_NUM_PREDICT:-900}"
export DIAGNOSTICS_ENABLED=true
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-$ROOT/.diagnostics/evidence-refinement-v2}"

mkdir -p "$DIAGNOSTICS_DIR"
echo "Evidence refinement v2 + diagnostics enabled"
echo "  output: $DIAGNOSTICS_DIR"
echo

exec bash run_word_boundary_experiment.sh
