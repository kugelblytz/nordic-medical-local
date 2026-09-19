# Evidence diagnostics

This branch adds opt-in evidence diagnostics around the successful
\`qwen-word-boundary-v1\` pipeline.

The normal \`/predict\` response is unchanged. When diagnostics are enabled, the
service additionally writes one JSON file per conversation containing:

- the timestamped Whisper segments;
- every global Whisper word ID and timestamp;
- all ten questions;
- Qwen reason code;
- selected segment IDs;
- selected start/end word IDs;
- Qwen evidence quote;
- selected word text;
- final resolved evidence timestamps;
- whether the span came from direct word IDs or the quote/segment fallback.

## 1. Run the diagnostic API on Vast

\`\`\`bash
cd /workspace/nordic-medical-local

git fetch origin
git switch experiment/evidence-diagnostics
git pull

rm -rf .diagnostics/word-boundary-v1

pkill -f "python3 api.py" || true
bash run_evidence_diagnostics.sh
\`\`\`

The normal word-boundary model configuration is used. Diagnostics are written
under:

\`\`\`text
/workspace/nordic-medical-local/.diagnostics/word-boundary-v1/
\`\`\`

The HTTP response remains competition-compatible.

## 2. Run the normal official evaluator

From the Windows machine:

\`\`\`cmd
cd C:\Users\joris\Nordic-AI-Cup-2026\medical-appointment
.venv\Scripts\activate

python local_evaluator.py --url http://151.237.25.16:26401/predict --verbose
\`\`\`

For research, a longer evaluator timeout is fine. These diagnostics are focused
on evidence quality rather than runtime.

After all 39 conversations have completed, verify on Vast:

\`\`\`bash
cd /workspace/nordic-medical-local
find .diagnostics/word-boundary-v1 -maxdepth 1 -name '*.json' | wc -l
\`\`\`

A complete training run should print:

\`\`\`text
39
\`\`\`

## 3. Copy the official training CSV to Vast

From Windows PowerShell or CMD:

\`\`\`cmd
scp -P 25615 "C:\Users\joris\Nordic-AI-Cup-2026\medical-appointment\data\question_train.csv" root@151.237.25.16:/workspace/nordic-medical-local/.diagnostics/question_train.csv
\`\`\`

If the Vast instance changes, replace the SSH host and port with the current
instance values.

## 4. Run the main analyzer

On Vast:

\`\`\`bash
cd /workspace/nordic-medical-local
source .venv/bin/activate

python diagnostics/analyze_evidence.py \
  --diagnostics-dir .diagnostics/word-boundary-v1 \
  --question-csv .diagnostics/question_train.csv \
  --worst 40
\`\`\`

It prints a summary and writes:

\`\`\`text
.diagnostics/word-boundary-v1/analysis/
├── summary.json
├── evidence_diagnostics.csv
└── worst_cases.md
\`\`\`

### Important metrics

\`mean_tiou\`
: Exact reproduction of evidence quality from the saved predictions.

\`word_oracle.mean_tiou_all_positives\`
: Best tIoU achievable by choosing any contiguous Whisper word range up to 80
  words. This approximates the ceiling of the word-timestamp representation.

\`segment_oracle.mean_tiou_all_positives\`
: Best tIoU achievable using one Whisper segment or two adjacent segments.

\`recoverable_mean_tiou_gap_with_current_classifier\`
: Approximate localization headroom if the current classifier stays unchanged
  but chooses optimal Whisper word boundaries.

\`failure_categories\`
: Geometric span errors such as no overlap, too narrow, too wide, or a late/early
  boundary.

\`span_shape\`
: Gold vs predicted duration plus systematic start/end timing bias.

## 5. Run semantic-location refinement diagnostics

The second analyzer distinguishes bad word boundaries from bad segment
selection:

\`\`\`bash
python diagnostics/refine_failure_analysis.py \
  --diagnostics-dir .diagnostics/word-boundary-v1 \
  --question-csv .diagnostics/question_train.csv
\`\`\`

It writes:

\`\`\`text
.diagnostics/word-boundary-v1/analysis/failure_refinement.csv
\`\`\`

Each low-tIoU positive is classified as one of:

\`word_boundary_failure\`
: Qwen selected the correct segment region, but its start/end word IDs are poor.
  Deterministic boundary shaping or a word-only refinement pass should help.

\`adjacent_segment_failure\`
: The gold evidence is recoverable within one segment of Qwen's chosen region.
  Expanding the evidence search to neighboring segments is promising.

\`wrong_semantic_location\`
: Even Qwen's chosen segment plus immediate neighbors cannot recover the gold
  span. A second semantic evidence-search pass over the full transcript is the
  likely fix.

\`mixed_or_annotation_shape\`
: Some overlap exists, but neither simple boundary correction nor one-segment
  expansion explains enough of the gap.

\`already_good\`
: Current tIoU is at least 0.75.

The two oracle thresholds used for failure classification are deliberately
conservative: a proposed fix must normally reach at least 0.50 tIoU and improve
the current span by at least 0.20.

## 6. Inspect the outputs

Useful commands on Vast:

\`\`\`bash
cat .diagnostics/word-boundary-v1/analysis/summary.json

sed -n '1,260p' \
  .diagnostics/word-boundary-v1/analysis/worst_cases.md

head -n 25 \
  .diagnostics/word-boundary-v1/analysis/failure_refinement.csv
\`\`\`

To download all reports to Windows, first archive them on Vast:

\`\`\`bash
cd /workspace/nordic-medical-local
tar -czf evidence-diagnostics.tar.gz \
  .diagnostics/word-boundary-v1/analysis
\`\`\`

Then from Windows:

\`\`\`cmd
scp -P 25615 root@151.237.25.16:/workspace/nordic-medical-local/evidence-diagnostics.tar.gz .
\`\`\`

## How to decide the next experiment

Use the diagnostic distribution rather than changing padding blindly.

- Many \`word_boundary_failure\` cases → tune deterministic span expansion,
  clause boundaries, padding, or a tiny word-boundary refinement pass.
- Many \`adjacent_segment_failure\` cases → allow Qwen/evidence refinement to
  inspect neighboring segments.
- Many \`wrong_semantic_location\` cases → add an evidence-only semantic search
  over the complete transcript.
- High word oracle but much lower current tIoU → substantial localization
  headroom remains.
- Low word oracle → Whisper timestamps or annotation conventions are limiting,
  so changing Qwen alone will not solve the problem.
