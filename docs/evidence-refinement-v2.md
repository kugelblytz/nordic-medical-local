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


## Performance timing diagnostics

Diagnostics schema v3 records performance measurements for every conversation.
The primary runtime metric is `request_core_ms`, which stops immediately before
research diagnostic-file serialization and therefore approximates competition
runtime without diagnostic disk I/O.

Per conversation, the JSON includes:

```text
request_core_ms
base64_decode_ms

asr:
  total_ms
  model_load_ms
  temp_file_write_ms
  whisper_materialize_ms
  temp_file_cleanup_ms
  audio_duration_s
  realtime_factor
  segment_count
  word_count

pass1 / pass2:
  total_wall_ms
  prompt_build_ms
  http_wall_ms
  parse_validate_ms
  retry_count
  first_attempt_success
  attempts[]

  ollama:
    total_ms
    load_ms
    prompt_eval_ms
    eval_ms
    prompt_tokens
    output_tokens
    prompt_tokens_per_s
    output_tokens_per_s

evidence_resolution_ms
evidence_counts
unaccounted_ms

diagnostics:
  payload_build_ms
  json_serialize_ms
  pre_write_total_ms
```

Whisper's timer covers full iteration of faster-whisper's lazy segment iterator,
not only the initial `transcribe()` call.

Ollama durations come from the completed local Ollama response and are converted
from nanoseconds to milliseconds. Retries remain separate attempt records.

Run the aggregate timing report after the evaluator:

```bash
python diagnostics/analyze_timing.py \
  --diagnostics-dir .diagnostics/evidence-refinement-v2
```

This writes:

```text
.diagnostics/evidence-refinement-v2/analysis/timing_summary.json
.diagnostics/evidence-refinement-v2/analysis/timing_by_conversation.csv
```

and prints mean/p50/p90/max stage latency, percentage of total request time,
Ollama prompt/decode throughput, retry counts, correlations, and the ten slowest
conversations.

For an external GPU-utilization trace, run this in a second SSH session before
starting the evaluator:

```bash
cd /workspace/nordic-medical-local
nvidia-smi dmon -s pucvmet -d 1 > gpu-dmon.log
```

Stop it with Ctrl+C after the benchmark.

### Current RTX 4090 Vast endpoint

```text
SSH:     95.253.220.115:60346 -> 22/tcp
API:     95.253.220.115:61806 -> 9054/tcp
Health:  http://95.253.220.115:61806/api
Predict: http://95.253.220.115:61806/predict
```

SSH:

```bash
ssh -p 60346 root@95.253.220.115
```

Evaluator:

```cmd
python local_evaluator.py --url http://95.253.220.115:61806/predict --verbose
```

For the first V2 quality/performance benchmark, keep the research evaluator
timeout long enough to avoid contaminating evidence quality with timeouts. Use
the resulting p50/p90/max `request_core_ms` values to decide whether the 4090
already meets the official runtime envelope and which stage to optimize first.
