# Nordic AI Cup 2026 — Medical Appointment local starter

A practical starter implementation for the **Medical Appointment** use case:

**MP3 → faster-whisper with word timestamps → local Qwen via Ollama → yes/no + exact evidence quote → word-level evidence timestamps**

No cloud API is used during `/predict`.

## Why this architecture

The challenge score is:

`0.4 * accuracy + 0.6 * mean temporal IoU`

So this repo keeps word timestamps and asks the local LLM to copy the shortest exact supporting quote. The quote is mapped back to Whisper words to produce a tighter interval than returning an entire ASR segment.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
ollama serve
```

Then, in another terminal:

```bash
source .venv/bin/activate
python api.py
```

Endpoint:

```text
POST http://localhost:9054/predict
```

## Test with the official evaluator

```bash
cd Nordic-AI-Cup-2026/medical-appointment
python local_evaluator.py --url http://127.0.0.1:9054/predict --verbose
```

## What to optimize first

1. Measure latency and stay safely below the 60-second budget.
2. Inspect hard-negative errors (dose, duration, date, drug, negation).
3. Compare evidence-span quality against the labeled positives.
4. Try Qwen 8B vs 14B and Whisper turbo vs full large-v3 if hardware permits.
5. Tune using the supplied training CSV rather than only the headline score.

The official competition rules prohibit cloud APIs during inference, so this starter keeps both transcription and reasoning local.
