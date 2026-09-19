# Word-boundary evidence experiment

Branch: `experiment/word-boundary-evidence`

## Goal

Improve temporal IoU without changing the part of the system that already works well: Qwen3.5 27B classification.

The current best hidden-validation system is:

```text
MP3
  ↓
faster-whisper large-v3-turbo
  ↓
timestamped transcript segments
  ↓
Qwen3.5 27B single-pass entailment
  ↓
answer + reason_code + evidence segment IDs + exact quote
  ↓
quote matcher inside selected segment(s)
  ↓
Whisper word timestamps
  ↓
evidence_start / evidence_end
```

Best hidden-validation score observed:

```text
0.7414863491598819
```

The recent local Qwen run was contaminated by five 60-second timeouts. On completed conversations, classification was approximately 98.8%. Evidence quality on questions actually answered yes was approximately:

```text
tIoU when answered yes ≈ 0.593
```

The experiment therefore focuses on evidence localization rather than changing the classifier.

---

## Hypothesis

The current evidence pipeline can lose IoU because it asks Qwen to choose a segment and quote, then uses fuzzy/exact text matching to recover timestamps.

Instead, give every Whisper word a stable integer ID and ask Qwen to return the exact first and last word IDs that cover the complete supporting evidence.

Example transcript representation:

```text
[S17 43.20-46.85]
142:The 143:patient 144:started 145:taking 146:100
147:milligrams 148:once 149:daily 150:last 151:week
```

Example positive answer:

```json
{
  "answer": true,
  "reason_code": "exact_support",
  "evidence_segment_ids": [17],
  "evidence_start_word_id": 144,
  "evidence_end_word_id": 151,
  "evidence_quote": "started taking 100 milligrams once daily last week"
}
```

The timestamp conversion becomes deterministic:

```text
W144.start → evidence_start
W151.end   → evidence_end
```

No fuzzy quote matching is required for the primary path.

---

## Experimental principles

1. Keep the classifier unchanged.
2. Change only evidence localization at first.
3. Preserve the current quote-based evidence logic as a fallback.
4. Use the supplied 39 training conversations for detailed diagnostics.
5. Use a longer local timeout during evidence research so latency does not contaminate the comparison.
6. Restore the official 60-second timeout before any competition-like benchmark.
7. Change one evidence variable at a time.

---

## Phase 1 — Preserve the baseline

The experiment branch must be based directly on `main`.

Do not modify the best-known competition baseline on `main`.

Record the baseline metrics used for comparison:

```text
Hidden validation:
  score = 0.7414863491598819

Recent local Qwen diagnostic:
  completed-request classification ≈ 98.8%
  tIoU when answered yes          ≈ 0.593
  official tIoU with timeouts      = 0.514
```

The local `0.514` should not be treated as clean evidence quality because five timed-out conversations returned no usable evidence.

---

## Phase 2 — Stable global Whisper word IDs

Whisper already returns word-level timestamps.

Create a flattened word index in transcript order:

```text
S0:
  W0
  W1
  W2

S1:
  W3
  W4
  ...
```

Each word ID must map to:

```python
word_id -> {
    "text": ...,
    "start": ...,
    "end": ...,
    "segment_id": ...
}
```

Requirements:

- IDs are deterministic for one transcription.
- IDs increase monotonically through the conversation.
- Every timestamped Whisper word gets exactly one ID.
- Segment IDs remain available.
- Word IDs never alter ASR output or timing.

Suggested implementation locations:

- extend the rendered transcript logic in `llm.py`;
- optionally add a small helper/dataclass in `models.py` if it makes mapping cleaner;
- keep original `WordToken` timestamps unchanged.

---

## Phase 3 — Compact word-addressed transcript rendering

Do not duplicate the transcript.

Replace the current plain segment rendering with a compact word-addressed form such as:

```text
[S17 43.20-46.85]
142:The 143:patient 144:started 145:taking 146:100 147:milligrams
148:once 149:daily 150:last 151:week
```

Design constraints:

- preserve `S#` segment IDs;
- preserve readable sentence flow;
- keep token overhead small;
- use global word IDs rather than resetting numbering inside every segment;
- do not include unnecessary decimal timestamps for every word because the application already owns the word-to-time mapping.

The LLM needs word identities, not the raw word timestamps.

---

## Phase 4 — Extend the structured output schema

Keep all current answer fields and add:

```json
{
  "evidence_start_word_id": 144,
  "evidence_end_word_id": 151
}
```

For TRUE answers:

- both IDs should be integers;
- start must be less than or equal to end;
- both IDs must exist;
- selected words must form one contiguous span;
- the span should contain all material evidence required by the proposition.

For FALSE answers:

```json
{
  "evidence_start_word_id": null,
  "evidence_end_word_id": null
}
```

Keep `evidence_quote` and `evidence_segment_ids` during the experiment because they are useful for diagnostics and fallback behavior.

---

## Phase 5 — Prompt changes

Do not change the classification rules.

Add evidence-specific instructions similar to:

```text
For TRUE answers:

1. Identify the complete contiguous spoken evidence needed to establish
   every material part of the proposition.

2. Return the first and last WORD IDs covering that evidence.

3. Do not return only a keyword or isolated value.
   Include the words needed to establish all relevant material slots,
   including dose, frequency, duration, time, location, status,
   negation, treatment/action, and result where applicable.

4. evidence_start_word_id must be <= evidence_end_word_id.

5. The selected word range must correspond to the evidence_segment_ids.

6. Prefer the shortest COMPLETE span, not the shortest fragment.

For FALSE answers:
- evidence_start_word_id must be null
- evidence_end_word_id must be null
```

Important: preserve the existing reason-code logic and hard-negative instructions.

The answer must still be:

```python
answer = reason_code == "exact_support"
```

---

## Phase 6 — Deterministic timestamp conversion

Add a direct word-boundary evidence path.

Primary path:

```text
valid Qwen start/end word IDs
        ↓
lookup first-word start timestamp
        ↓
lookup last-word end timestamp
        ↓
return span
```

Initial experiment:

```text
padding = 0.00 s
```

Do not mix padding optimization into the first A/B.

---

## Phase 7 — Preserve the current evidence logic as fallback

Do not delete `evidence.py` quote matching.

Use:

```text
TRUE answer
  ↓
valid word IDs?
  ├─ yes → direct word timestamps
  └─ no  → current segment + quote matcher
```

Validation rules before accepting word IDs:

- both IDs exist;
- start <= end;
- selected range is contiguous;
- selected range is not absurdly long;
- selected words are compatible with the selected segment IDs;
- start/end are not outside the transcript.

If validation fails, fall back to today's `locate_evidence()` behavior.

This prevents experimental evidence output from turning a good classification result into a missing span.

---

## Phase 8 — First controlled A/B

Compare:

### A — Current baseline

```text
segment IDs
+ evidence quote
+ exact/fuzzy quote matching
+ 0.12 s quote padding
```

### B — Word-boundary experiment

```text
Qwen start/end word IDs
+ direct Whisper timestamps
+ 0.00 s padding
```

Do not change:

- Qwen3.5 27B;
- Whisper `large-v3-turbo`;
- ASR beam size;
- VAD;
- context length;
- temperature;
- reasoning prompt;
- reason codes;
- TRUE/FALSE decision rule.

This isolates the effect of the evidence representation.

---

## Phase 9 — Diagnostic evaluation timeout

For local evidence research only, increase the official local evaluator request timeout from 60 seconds to 120 seconds.

Also allow the Vast Ollama call enough time to complete.

Example development settings:

```text
local evaluator request timeout = 120 s
OLLAMA_TIMEOUT                 = 90 s
```

Purpose:

- measure evidence quality without five whole-conversation timeouts distorting the result.

Before any competition-like test, restore:

```text
request timeout = 60 s
```

The actual competition constraint does not change.

---

## Phase 10 — Evidence diagnostic report

For every annotated positive question, save:

```text
question_id
question
question_type

gold_start
gold_end
gold_duration

predicted_start
predicted_end
predicted_duration

tIoU

answer
reason_code
evidence_segment_ids
evidence_start_word_id
evidence_end_word_id
evidence_quote
selected_word_text

nearby previous segment
selected segment(s)
nearby next segment
```

Preferred output format:

- machine-readable JSONL or CSV;
- optionally a human-readable Markdown report for the worst cases.

Sort or bucket examples by tIoU.

Suggested buckets:

```text
0.000
0.000–0.250
0.250–0.500
0.500–0.750
0.750–0.900
0.900–1.000
```

---

## Phase 11 — Diagnose low-IoU failure modes

For bad positives, assign one failure category:

1. wrong semantic occurrence;
2. correct occurrence, start too late;
3. correct occurrence, start too early;
4. correct occurrence, end too early;
5. correct occurrence, end too late;
6. evidence requires an adjacent segment;
7. ASR transcription/timing error;
8. correct semantic evidence but annotation convention differs;
9. invalid/missing word-boundary output caused fallback.

This matters because padding only helps boundary errors.

Padding cannot fix a completely wrong semantic location.

---

## Phase 12 — Offline span-shape optimization

Once a Qwen run has produced stable word IDs, save them.

Do not rerun Qwen for every padding experiment.

From the same saved word selections, compute multiple deterministic span variants offline:

```text
raw word boundaries
±0.05 s
±0.10 s
±0.15 s
±0.20 s
±0.30 s
minimum 1.5 s
minimum 2.0 s
minimum 2.5 s
whole containing segment
adaptive expansion toward segment/clause boundaries
```

Compare each against the gold training spans.

Known training evidence-span statistics:

```text
minimum ≈ 0.16 s
median  ≈ 2.88 s
mean    ≈ 3.21 s
p90     ≈ 5.44 s
maximum ≈ 14.2 s
```

These statistics suggest very short predicted spans may systematically under-cover some gold annotations.

Do not blindly force every span to the median; use the training set to understand when expansion is useful.

---

## Phase 13 — Adjacent-segment evidence

Whisper segmentation is not semantic segmentation.

A proposition may require the end of one segment and the beginning of another.

Example:

```text
S17: "The pain started about three weeks ago,"
S18: "and it's mainly on the left side."
```

Question:

```text
Has the patient had left-sided pain for three weeks?
```

The complete evidence requires both utterances.

The word-boundary representation should permit one contiguous range crossing adjacent segments when necessary.

Do not permit arbitrary non-adjacent spans in the first version.

---

## Phase 14 — Classification regression guard

Adding word IDs may distract the LLM.

Every experiment must report:

```text
overall accuracy
positive accuracy
hard-negative accuracy
off-topic accuracy
```

Current completed-request Qwen classification is approximately 98.8%.

Target:

```text
classification degradation <= ~1 percentage point
```

Ideally there is no measurable regression.

If adding word IDs materially harms classification, do not continue with a single-pass word-addressed transcript.

Move to the two-stage alternative below.

---

## Phase 15 — Two-stage fallback experiment

Only try this if word IDs clearly improve evidence but harm classification.

Stage 1 remains today's clean transcript:

```text
clean transcript
    ↓
Qwen classification
    ↓
answer + reason_code + evidence_segment_ids
```

Stage 2 runs only for TRUE questions and receives a small local context:

```text
question
selected segment
previous adjacent segment
next adjacent segment

with numbered words
```

It returns only:

```json
{
  "evidence_start_word_id": 144,
  "evidence_end_word_id": 151
}
```

Batch all TRUE questions from one conversation into one evidence-refinement request.

Do not allow stage 2 to change TRUE/FALSE answers.

Risk: extra latency. The current 27B deployment is already near the 60-second request limit on some hardware, so this is not the first implementation to test.

---

## Phase 16 — Success criteria

Current clean diagnostic evidence reference:

```text
tIoU when answered yes ≈ 0.593
```

Interpretation targets:

```text
0.59 → 0.60    too small to justify much complexity
0.59 → 0.65    meaningful improvement
0.59 → 0.70    strong improvement
0.59 → 0.75+   major improvement
```

Because tIoU contributes 60% of the final score, an evidence increase from:

```text
0.593 → 0.700
```

would correspond to roughly:

```text
(0.700 - 0.593) * 0.6 ≈ +0.064
```

before accounting for classification or timeout changes.

---

## Implementation checklist

- [ ] Add stable global word IDs.
- [ ] Add word-ID transcript renderer.
- [ ] Add `evidence_start_word_id` to the structured schema/model.
- [ ] Add `evidence_end_word_id` to the structured schema/model.
- [ ] Preserve current reason-code classification.
- [ ] Update prompt with complete-span word-boundary instructions.
- [ ] Add word-ID validation.
- [ ] Add direct word-ID → timestamp conversion.
- [ ] Keep current quote matcher as fallback.
- [ ] Add tests for valid word ranges.
- [ ] Add tests for invalid/reversed/missing IDs.
- [ ] Add tests for ranges crossing adjacent segments.
- [ ] Add tests proving fallback behavior remains intact.
- [ ] Add evidence diagnostic logging/report generation.
- [ ] Run 39-conversation diagnostic with long timeout.
- [ ] Compare classification against baseline.
- [ ] Compare tIoU against ~0.593 answered-positive baseline.
- [ ] Save selected word IDs for offline span-shape experiments.
- [ ] Sweep padding/minimum-duration strategies offline.
- [ ] Analyze zero/low-IoU failure modes.
- [ ] Restore 60-second evaluator timeout.
- [ ] Measure real latency and timeout rate.
- [ ] Only then consider a hidden-validation run.

---

## Recommended experiment sequence

1. Implement the single-pass word-boundary version.
2. Run unit tests and syntax checks.
3. Run the 39-conversation training evaluator with a 120-second research timeout.
4. Save a structured evidence report.
5. Verify classification stayed effectively unchanged.
6. Compare raw word-boundary tIoU against the current evidence method.
7. Use saved word IDs to test padding and adaptive expansion offline.
8. Inspect every zero-IoU and very-low-IoU positive.
9. Select a robust span-construction rule rather than the maximum-overfit training rule.
10. Restore the official 60-second timeout.
11. Optimize latency if necessary.
12. Only after the system passes runtime requirements, test on hidden validation.

---

## Non-goals for the first experiment

Do not change these at the same time:

- ASR model;
- Qwen model size;
- classification prompt;
- number of reasoning passes;
- temperature;
- context window;
- VAD behavior;
- hard-negative logic.

The first experiment should answer exactly one question:

> Does having Qwen select explicit Whisper word boundaries improve evidence tIoU without materially hurting classification?
