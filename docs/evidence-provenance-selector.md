# Evidence provenance selector experiment

This branch tests a specific hypothesis about the Nordic AI Cup evidence labels:

> when several transcript passages all support a proposition, the gold span may
> reflect the question's annotation provenance rather than the strongest
> standalone semantic entailment.

The experiment is fully offline with respect to ASR. It consumes cached
candidate-discovery results plus cached diagnostics and calls only the local
Ollama model.

## Branch

```text
experiment/evidence-provenance-selector
```

It branches from `experiment/offline-evidence-candidates`, not from the failed
V2 refinement pipeline.

## Controlled A/B design

Both selectors see exactly the same:

- question;
- independently discovered candidates;
- one-segment local context around each candidate;
- deterministic anonymized candidate order;
- model, temperature, context size, and output schema.

Neither selector sees:

- gold timestamps;
- tIoU;
- candidate discovery rank;
- Pass-1 score.

Only the objective changes.

### Semantic selector

Chooses the candidate that most directly, explicitly, self-containedly, and
completely proves the proposition.

### Provenance selector

Chooses the candidate most likely to be the designated source evidence in an
annotated doctor-patient dataset, considering discourse role such as initial
report, clarification, clinician assessment, explicit confirmation, informed
preference, final plan, completed action, result/follow-up, or summary.

It is explicitly told not to blindly prefer later text or clinician speech.

## Outputs and metrics

The script reports:

- selected mean tIoU;
- candidate-oracle mean tIoU;
- mean regret to candidate oracle;
- oracle-choice rate;
- selected tIoU >= .75 rate;
- improvement/degradation vs Pass 1;
- pairwise provenance-minus-semantic tIoU;
- deterministic-order permutation stability;
- results broken into zero, difficult, medium, and control baseline buckets.

Each selection also records a discourse-role label for diagnosis.

## First wiring test: the existing 18 zero-IoU cases

The existing candidate results should already be at:

```text
.diagnostics/evidence-candidate-lab/zero/candidate_results.csv
```

Run one deterministic ordering first:

```bash
python experiments/evidence_provenance_selector.py \
  --candidate-results .diagnostics/evidence-candidate-lab/zero/candidate_results.csv \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --selector both \
  --permutations 1
```

This uses only the independently discovered candidates. It does not add Pass 1
as a candidate, which makes the first comparison directly about ranking the
candidate-finder outputs.

Then test robustness to display order:

```bash
python experiments/evidence_provenance_selector.py \
  --candidate-results .diagnostics/evidence-candidate-lab/zero/candidate_results.csv \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --selector both \
  --permutations 3
```

Cached permutation 0 results are reused automatically.

Outputs:

```text
.diagnostics/evidence-provenance-selector/
  summary.json
  selector_results.csv
  selector_results.md
  cache/
    semantic/
    provenance/
```

## Production-oriented variant

Once candidate discovery has been run for a broader cohort, add the original
Pass-1 span as another anonymized candidate:

```bash
python experiments/evidence_provenance_selector.py \
  --candidate-results .diagnostics/evidence-candidate-lab/all/candidate_results.csv \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --selector both \
  --include-pass1 \
  --permutations 3 \
  --output-dir .diagnostics/evidence-provenance-selector-all
```

This tests whether the selector can improve difficult cases while choosing to
retain Pass 1 on already-good controls.

## Broader candidate generation

To make the real held-out test, first generate candidates for all correctly
classified positive questions:

```bash
python experiments/evidence_candidate_lab.py \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --question-csv .diagnostics/question_train.csv \
  --cohort all \
  --candidate-count 4 \
  --recall-threshold 0.75
```

That still does not rerun Whisper. Candidate discovery is cached by
conversation.

Then run the production-oriented selector command above.

## Interpretation

The provenance hypothesis is supported only if it beats the semantic selector
consistently on held-out difficult/medium cases and does not materially damage
the control bucket.

A tiny aggregate difference is not enough. The useful signal is:

- lower regret to the candidate oracle;
- higher high-quality selection rate;
- positive provenance-minus-semantic tIoU across unseen conversations;
- stable choices across candidate-order permutations;
- small degradation on baseline-good controls.

Boundary refinement is intentionally excluded from this experiment so ranking
and annotation geometry remain separable.
