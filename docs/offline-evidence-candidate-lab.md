# Offline evidence candidate lab

This experiment isolates evidence retrieval from the end-to-end medical pipeline.

It does **not** run Whisper and does **not** call \`/predict\`. It reads cached
conversation diagnostics, reconstructs the exact word-numbered Whisper
transcript, asks Ollama to find multiple supporting occurrences, then scores
those candidates against the training evidence timestamps.

The branch is based on \`experiment/evidence-diagnostics\`, not on the failed
V2 refinement pipeline.

## Experimental invariant

The candidate finder sees:

- the cached full transcript with segment and global word IDs;
- the question;
- the fact that upstream classification already said YES.

It does **not** see:

- Pass-1 evidence;
- Pass-1 word IDs;
- Pass-1 quote;
- gold timestamps.

This prevents the anchoring observed in evidence-refinement-v2.

## Cohorts

- \`zero\`: correctly predicted-YES positives with first-pass tIoU = 0.
- \`low\`: predicted-YES positives with first-pass tIoU <= a threshold
  (default 0.25).
- \`control\`: predicted-YES positives with first-pass tIoU >= a threshold
  (default 0.75).
- \`all\`: all predicted-YES positives.

Classification misses are excluded by default because this downstream stage
would only run after a YES decision. Use \`--include-classification-misses\`
for diagnostic-only experiments.

For schema-v2/v3 diagnostics, cohort selection and baseline scoring always use
the **first-pass** timestamps, not the V2 refined timestamps.

## First run: zero-IoU target set

On the Vast machine:

\`\`\`bash
cd /workspace/nordic-medical-local
git fetch origin
git switch experiment/offline-evidence-candidates
git pull --ff-only
source .venv/bin/activate
\`\`\`

The cached V2 diagnostics and training CSV can remain where they already are.

First verify the selected examples without calling Ollama:

\`\`\`bash
python experiments/evidence_candidate_lab.py \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --question-csv .diagnostics/question_train.csv \
  --cohort zero \
  --dry-run
\`\`\`

Then run independent candidate discovery:

\`\`\`bash
python experiments/evidence_candidate_lab.py \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --question-csv .diagnostics/question_train.csv \
  --cohort zero \
  --candidate-count 4 \
  --recall-threshold 0.75
\`\`\`

No Whisper inference happens in either command.

## Outputs

The run writes:

\`\`\`text
.diagnostics/evidence-candidate-lab/
  cache/
    sample_XX__<hash>.json
  zero/
    summary.json
    candidate_results.csv
    candidate_results.md
\`\`\`

Model outputs are cached by transcript, question set, model, prompt version,
candidate count, and max-word limit. Re-running the same experiment should
reuse the cache. Use \`--refresh\` only when you intentionally want to call
the model again.

## Metrics

The report separates retrieval from ranking:

- top-1 candidate mean tIoU;
- best-returned-candidate oracle mean tIoU;
- recall@1 through recall@K for tIoU >= the requested threshold;
- top-1 improved/degraded/unchanged counts;
- mean tIoU on the full 195-positive set if only the selected cohort were
  replaced by top-1 candidates;
- mean tIoU on the full positive set if an oracle could select the best
  returned candidate.

The key first question is candidate recall, not selector performance. If
recall@4 is high on the zero-IoU set, a separate selector/ranker experiment is
worth building. If recall@4 is low, candidate discovery itself needs work.

## Control experiment

After the zero set works, check whether independent retrieval also proposes
sensible candidates for already-good examples:

\`\`\`bash
python experiments/evidence_candidate_lab.py \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --question-csv .diagnostics/question_train.csv \
  --cohort control \
  --control-min-tiou 0.75 \
  --candidate-count 4
\`\`\`

Do not integrate this into \`/predict\` until candidate recall and later
selector behavior are validated offline.
