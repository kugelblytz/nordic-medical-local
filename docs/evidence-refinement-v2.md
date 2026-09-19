# Evidence refinement v2

This experiment keeps the high-accuracy first-pass classifier unchanged and adds
one optional evidence-only Qwen call for questions already classified TRUE.

## Architecture

\`\`\`text
audio
  ↓
Whisper large-v3-turbo + word timestamps
  ↓
Pass 1: existing Qwen3.5 27B classifier (unchanged)
  ↓
TRUE/FALSE is frozen
  ↓
Pass 2: one batched full-transcript evidence refinement call for TRUE questions
  ↓
validate refined contiguous word IDs
  ↓
valid refinement → use it
invalid/missing refinement → keep Pass-1 evidence
Pass-1 direct range invalid → existing quote/segment fallback
\`\`\`

Pass 2 has no answer or reason-code field, so it cannot alter classification.

## Important implementation choices

- The refiner receives the full word-numbered transcript.
- All TRUE questions are refined in one Ollama call.
- The original Pass-1 range and selected text are provided as hints.
- The refiner searches for the speech act that best matches the question:
  patient report, examination finding, clinician assessment, explicit test
  result, completed/committed action, or final plan.
- Refined evidence may cross at most two adjacent Whisper segments.
- A refiner exception is caught locally and preserves Pass-1 evidence.
- Direct word IDs are authoritative when they form a valid contiguous range;
  redundant Pass-1 segment IDs no longer invalidate a good word span.
- Refinement is disabled by default and enabled only by the experiment launcher.

## Run on Vast with diagnostics

\`\`\`bash
cd /workspace/nordic-medical-local

git stash push -m "vast changes before evidence refinement v2" || true
git fetch origin
git switch experiment/evidence-refinement-v2 || \
  git switch -c experiment/evidence-refinement-v2 \
  --track origin/experiment/evidence-refinement-v2
git pull --ff-only

rm -rf .diagnostics/evidence-refinement-v2

pkill -f "python3 api.py" || true
bash run_evidence_refinement_diagnostics.sh
\`\`\`

Verify:

\`\`\`bash
curl http://127.0.0.1:9054/api
\`\`\`

Expected:

\`\`\`json
"evidence_strategy": "qwen-evidence-refinement-v2"
\`\`\`

The research launcher uses the existing 90-second Ollama classifier timeout plus:

\`\`\`text
EVIDENCE_REFINEMENT_TIMEOUT=90
EVIDENCE_REFINEMENT_NUM_PREDICT=900
\`\`\`

For the current slower development GPU, use a long local-evaluator timeout
(e.g. 180 seconds) for this research run. That is not a competition-valid
latency measurement; it prevents latency from contaminating the evidence A/B.

## Run tests

\`\`\`bash
cd /workspace/nordic-medical-local
source .venv/bin/activate

pytest -q
python -m py_compile *.py diagnostics/*.py
\`\`\`

## Analyze the 39-conversation run

Copy the official training CSV to:

\`\`\`text
.diagnostics/question_train.csv
\`\`\`

Then:

\`\`\`bash
python diagnostics/analyze_evidence.py \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --question-csv .diagnostics/question_train.csv \
  --worst 40
\`\`\`

The summary now reports both the original Pass-1 evidence and the final refined
evidence from the same requests:

\`\`\`text
first-pass mean tIoU
final mean tIoU
mean delta
improved / unchanged / degraded positives
improved by >= 0.10
degraded by >= 0.10
zero-tIoU before / after
baseline-good mean delta
\`\`\`

The CSV also records:

\`\`\`text
first_pass_pred_start
first_pass_pred_end
first_pass_tiou
refinement_delta_tiou
refinement_attempted
refinement_valid
refinement_used
refinement_changed_range
refined_start_word_id
refined_end_word_id
refined_selected_word_text
\`\`\`

## Promotion criteria

The first-pass classifier must remain identical because Pass 2 never writes
answers.

A useful first evidence target is mean tIoU >= 0.68. A strong result is >= 0.70.
Also inspect regression on positives whose Pass-1 tIoU was already >= 0.75;
overall mean improvement should not come from damaging many already-good spans.

Do not combine this A/B with batched Whisper, beam-size changes, model-size
changes, global padding, or a different classification prompt.
