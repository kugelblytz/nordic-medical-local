#!/usr/bin/env bash
set -euo pipefail

export EVIDENCE_REFINEMENT_ENABLED=true
export EVIDENCE_REFINEMENT_TIMEOUT="${EVIDENCE_REFINEMENT_TIMEOUT:-90}"
export EVIDENCE_REFINEMENT_NUM_PREDICT="${EVIDENCE_REFINEMENT_NUM_PREDICT:-900}"

exec bash run_word_boundary_experiment.sh
