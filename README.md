

A local inference implementation for the **Medical Appointment** use case.

**Best-known competition configuration:**

**MP3 → faster-whisper large-v3-turbo → local Qwen3.5 27B via Ollama → strict single-pass entailment → segment selection + exact quote → Whisper word timestamps**

No cloud API is used during `/predict` on the competition configuration.

## Best-known result: 0.741486 hidden validation

The strongest online validation result observed so far is:

```text
score: 0.7414863491598819
attempt: a21da8ecbab444d9a24f29b810608b75
validation conversations: 19
errors: []
```

The validation run started at `2026-09-18T00:34:55Z` and finished at `2026-09-18T00:42:14Z`, or about **23.1 seconds per conversation on average**. This was comfortably within the competition's 60-second-per-conversation average budget.

### Architecture that produced the 0.741486 score

1. **ASR — faster-whisper `large-v3-turbo`**
   - CUDA + float16
   - English forced
   - beam size 5
   - VAD enabled
   - word timestamps enabled
   - each Whisper segment is assigned an `S#` ID

2. **Reasoning — Qwen3.5 27B through Ollama**
   - one request for all 10 questions from a conversation
   - 8192-token context
   - temperature 0
   - Ollama thinking disabled
   - strict JSON schema with one result per question
   - the model must classify each question using one of:
     - `exact_support`
     - `wrong_entity`
     - `wrong_value`
     - `wrong_time`
     - `wrong_location`
     - `negated`
     - `contradicted`
     - `discussed_not_done`
     - `not_explicit`
     - `off_topic`
   - a question is TRUE only when the reason code is `exact_support`

3. **Hard-negative handling**
   - the prompt explicitly compares material slots such as medicine/entity, dose, unit, frequency, duration, date/time, body location, symptom/test/result, treatment/action, and status
   - one material mismatch is enough to make the proposition false
   - proposed/discussed treatment is not treated as completed treatment
   - past history is not treated as current state
   - negation, corrections, and plan changes are handled explicitly

4. **Evidence selection**
   - for TRUE answers, Qwen returns the smallest 1–2 supporting Whisper segment IDs plus an exact spoken quote
   - the evidence matcher first searches for that quote **inside the selected segment(s)**, preventing repeated phrases elsewhere in the conversation from stealing the timestamp
   - exact token matching is attempted first, followed by a conservative fuzzy match
   - when a quote matches, Whisper word timestamps produce a tight evidence span with `0.12 s` padding
   - if quote alignment fails, the selected segment span is used as a safe fallback with `0.20 s` padding

5. **Failure behavior**
   - malformed/missing structured output is retried once
   - if the full prediction pipeline fails, the API still returns arrays of the correct length using an all-false/null fallback

### Why this became the best configuration

The development path gave several useful signals:

- the original broken/failed path scored exactly **0.200**, which exposed an all-false fallback caused by malformed model output
- moving to strict keyed structured output and a larger Qwen model produced about **0.69625** on hidden validation
- switching Whisper from `large-v3-turbo` to full `large-v3` did not produce a meaningful improvement in the observed validation result
- a multi-pass fact-ledger / adversarial-verifier reasoning pipeline **reduced** validation performance, so the system returned to one strict entailment pass
- the combination of **Qwen3.5 27B + the tighter evidence pipeline** produced the current best **0.741486** score

This is not a pure model-size A/B test because the 27B run also included evidence-localization improvements. The safest interpretation is that the current combination is the best observed system, not that every point of improvement came from the larger LLM alone.

### Important distinction: hidden validation vs local training evaluation

The `0.741486` score above is from the competition's **hidden online validation set**.

A separate development experiment using GPT-5.6 Sol on the supplied 39-conversation training set achieved very high classification accuracy but only moderate evidence tIoU. That experiment is useful for diagnosing the remaining bottleneck, but its `0.697` local score is **not directly comparable** with the hidden-validation `0.741486` because the datasets differ.

The strongest current lesson is that **classification can become nearly saturated while evidence localization still limits the final score**, which matters because tIoU contributes 60% of the competition metric.

## Competition contract

The endpoint accepts one conversation plus all questions and returns exactly:

```json
{
  "answers": [true, false],
  "evidence_start": [12.4, null],
  "evidence_end": [15.8, null]
}
```

The score is `0.4 * accuracy + 0.6 * mean temporal IoU`.

The supplied training annotations have positive evidence spans averaging about 3.2 seconds (median about 2.9 seconds). The model now selects explicit Whisper segment IDs for positive answers, with quote matching retained as a fallback. The prompt explicitly checks hard-negative differences in entity, dose/value, time, location, negation, and discussed-vs-completed actions.

## Azure / NVIDIA setup

Current faster-whisper/CTranslate2 GPU execution requires CUDA 12 cuBLAS and cuDNN 9 in addition to a working NVIDIA driver.

On an Ubuntu NVIDIA VM:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip curl

git clone https://github.com/kugelblytz/nordic-medical-local.git
cd nordic-medical-local

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Install Ollama and pull Qwen3:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:27b
sudo systemctl enable --now ollama
```

Confirm the GPU and local LLM:

```bash
nvidia-smi
ollama list
```

Start the endpoint with the CUDA library path configured:

```bash
chmod +x run_gpu.sh
./run_gpu.sh
```

Startup loads Whisper and sends a tiny warm-up request to Ollama so first-request model loading is kept out of the scored request.

Check:

```bash
curl http://127.0.0.1:9054/
curl http://127.0.0.1:9054/api
```

## Official local evaluator

From the organizer repository:

```bash
cd Nordic-AI-Cup-2026/medical-appointment
python local_evaluator.py --url http://127.0.0.1:9054/predict --verbose
```

For an Azure VM:

```bash
python local_evaluator.py \
  --url http://<VM_PUBLIC_IP>:9054/predict \
  --verbose
```

Open TCP 9054 in the VM's Network Security Group before testing remotely.

## Configuration

Important defaults:

- `ASR_MODEL=large-v3-turbo`
- `ASR_DEVICE=cuda`
- `ASR_COMPUTE_TYPE=float16`
- `OLLAMA_MODEL=qwen3.5:27b`
- `EVIDENCE_MODE=segment`
- `EVIDENCE_PAD_SECONDS=0.20`
- `WARMUP_ON_START=true`

The app reads environment variables directly. `.env.example` is a reference file; it is not automatically loaded.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
python -m py_compile *.py
```

## Verification status

The API request/response shape matches the official Medical Appointment DTOs.

A mocked end-to-end request has been exercised through FastAPI, including:

- base64 audio decoding
- timestamped ASR segments
- one batched answer for all questions
- structured LLM JSON parsing
- evidence quote matching
- timestamp generation
- exact response-array ordering and lengths
- valid fallback response on model failure

Real GPU/model throughput still needs to be measured on the target VM because it depends on the GPU, driver, CUDA libraries, model download, and Ollama runtime.

## Optimization order

1. Preserve the current Qwen3.5 27B / Whisper turbo configuration as the hidden-validation baseline.
2. Measure local errors separately for classification and evidence localization.
3. Inspect zero/low-tIoU positives with their selected segment IDs, evidence quote, predicted span, and gold span.
4. Tune evidence alignment/padding before changing the classifier, because evidence carries 60% of the score.
5. Change one variable at a time when testing larger local models, alternate prompts, or ASR variants.

The competition rules prohibit cloud APIs during inference, so both ASR and question answering stay local.


## Vast.ai deployment

Vast.ai normally gives you a Docker container on a rented GPU host. You do not need to build or publish a custom image for this repo.

### Recommended instance

Choose an **on-demand** instance with one of:

- RTX 3090 24 GB
- RTX 4090 24 GB
- A10 24 GB
- T4 16 GB only if it is much cheaper

Use a standard CUDA/PyTorch SSH template. At least 40 GB disk is recommended.

### Required port mapping

Expose container port **9054/tcp** in the Vast template/container settings.

The external host port may be different from 9054. Vast will show the actual public mapping after the instance starts.

### Deploy

SSH into the instance using the command shown by Vast, then:

```bash
git clone git@github.com:kugelblytz/nordic-medical-local.git
cd nordic-medical-local
chmod +x vast_start.sh run_gpu.sh vast_health.sh
./vast_start.sh
```

The script will:

1. verify the NVIDIA GPU
2. create the Python virtual environment
3. install Python and CUDA runtime dependencies
4. install Ollama if needed
5. start Ollama in the background
6. pull `qwen3.5:27b`
7. load/warm Whisper and Qwen
8. start FastAPI on `0.0.0.0:9054`

First startup downloads several GB of model weights.

### Verify inside the container

In another SSH session:

```bash
cd nordic-medical-local
chmod +x vast_health.sh
./vast_health.sh
```

You should see the GPU plus successful Ollama and competition API health checks.

### Verify from your laptop

In the Vast instance page, find the public IP and host port that map to container port 9054.

For example, if Vast shows:

```text
203.0.113.20:32145 -> 9054/tcp
```

test:

```bash
curl http://203.0.113.20:32145/api
```

and use this as the evaluator URL:

```text
http://203.0.113.20:32145/predict
```

### Run the official evaluator

From the organizer's `medical-appointment` directory on your laptop:

```bash
python local_evaluator.py \
  --url http://<VAST_PUBLIC_IP>:<VAST_HOST_PORT>/predict \
  --verbose
```

### Cleanup

When finished, **Delete/Destroy the Vast instance** rather than only stopping it.

Stopping can leave storage allocated depending on the rental/setup. Destroying the instance is the clean end state for a disposable benchmark.

Do not keep important data only on the instance; destroying it removes the container's local data.
