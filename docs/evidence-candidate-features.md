# Independent candidate-feature experiment

This branch removes the listwise candidate-choice behavior that showed strong
candidate-order sensitivity in the provenance-selector experiment.

## Branch

```text
experiment/evidence-candidate-features
```

It branches from `experiment/evidence-provenance-selector` and keeps all
earlier offline tooling. It does not modify the production API.

## Hypothesis

The previous listwise selectors saw all candidates simultaneously and changed
their choice when candidate order changed. This experiment instead asks Qwen
to describe every candidate independently, then performs ranking in
deterministic Python code.

The model never sees two candidates in the same feature-extraction call.

## Stage 1: question intent

Once per question, Qwen assigns:

- `question_target`:
  - diagnosis_or_assessment
  - patient_symptom_or_experience
  - patient_preference_or_request
  - test_or_result
  - treatment_plan_or_continuation
  - completed_treatment_or_action
  - medication_effect
  - history_or_timing
  - absence_or_normal_state
  - other
- `expects_finalized_state`: boolean

No transcript or candidate is supplied for this call.

## Stage 2: candidate-local feature extraction

Each candidate is sent in a separate call with only:

- the question;
- that candidate's selected range;
- its local context (default ±1 Whisper segment).

The model returns:

- speaker_role
- discourse_role
- support_form
- temporal_role
- action_status
- explicitness 0–3
- question_alignment 0–3
- self_contained
- finalized

The candidate never sees another candidate, discovery rank, gold timestamps,
tIoU, Pass-1 score, or a request to choose a winner.

All question and candidate features are cached.

## Stage 3: deterministic ranking

Three rankers are reported from the same candidate pool.

### discovery_top1

The original candidate-discovery ordering. No additional decision logic.

### neutral

Uses only:

- question_alignment;
- explicitness;
- self_containment.

It deliberately ignores provenance/discourse-role features.

### provenance

Starts from the neutral score and adds deterministic compatibility between:

- question target;
- discourse role;
- action status;
- finalized state;
- temporal role;
- support form.

For example:

- diagnosis/assessment questions favor clinician assessment and explicit
  confirmation over initial patient report;
- preference/request questions favor patient-preference utterances;
- treatment-plan questions favor final plans;
- completed-action questions strongly favor completed actions and penalize
  merely planned/discussed actions;
- medication-effect questions favor explicit confirmation, patient experience,
  and follow-up outcome.

These are explicit Python weights in
`experiments/evidence_candidate_features.py`; once features are cached they
can be edited and re-evaluated without any Qwen calls.

## First run: existing 18 zero-IoU cases

Switch branch:

```bash
cd /workspace/nordic-medical-local
git fetch origin
git switch experiment/evidence-candidate-features
git pull --ff-only
source .venv/bin/activate
```

Verify:

```bash
python -m py_compile experiments/evidence_candidate_features.py
python -m pytest -q tests/test_evidence_candidate_features.py
```

Run:

```bash
python experiments/evidence_candidate_features.py \
  --candidate-results .diagnostics/evidence-candidate-lab/zero/candidate_results.csv \
  --diagnostics-dir .diagnostics/evidence-refinement-v2
```

This does not run Whisper, `/predict`, or candidate discovery.

For the current 18-case set it will make approximately:

- 18 question-intent calls;
- one independent feature call for each returned candidate.

Every result is cached, so rerunning the same command should make zero model
calls.

## Outputs

```text
.diagnostics/evidence-candidate-features/
  summary.json
  question_features.csv
  candidate_features.csv
  ranking_results.csv
  cache/
    questions/
    candidates/
```

The most useful file for analysis is `candidate_features.csv`. It lets us
inspect every candidate's extracted role and score alongside post-hoc tIoU.

## Metrics

The summary compares:

- discovery_top1;
- neutral feature ranker;
- provenance feature ranker;
- candidate oracle.

For each ranker:

- mean selected tIoU;
- mean candidate-oracle tIoU;
- mean regret;
- oracle-choice rate;
- tIoU >= .75 rate;
- mean delta vs Pass 1;
- zero/difficult/medium/control buckets.

Pairwise blocks report:

- neutral minus discovery_top1;
- provenance minus discovery_top1;
- provenance minus neutral.

A provenance signal is useful only if it beats both discovery ordering and the
neutral feature ranker. If neutral and provenance remain similar, discourse
features are not adding meaningful ranking information.

## Include Pass 1 later

For a production-oriented broader experiment:

```bash
python experiments/evidence_candidate_features.py \
  --candidate-results .diagnostics/evidence-candidate-lab/all/candidate_results.csv \
  --diagnostics-dir .diagnostics/evidence-refinement-v2 \
  --include-pass1 \
  --output-dir .diagnostics/evidence-candidate-features-all
```

Pass 1 is deduplicated against discovery candidates and independently
feature-extracted just like every other candidate.

## Iterating on ranker rules

Feature extraction is the expensive part. After the first run, all features
are in CSV and JSON cache files.

The current script recomputes ranking after loading cached features. Therefore
editing only `ROLE_BONUSES`, `ACTION_BONUSES`, `neutral_score`, or
`provenance_score` and rerunning the command will reuse cached model
features.

Do not tune weights against the 18 zero cases and then claim generalization.
Those cases are a development set. Any promising rule must later be evaluated
on unseen difficult/medium/control conversations.
