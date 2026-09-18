# OpenAI reasoning experiment branch

This branch adds a **development-only** OpenAI reasoning backend while keeping the normal Ollama backend available. Use it only with the official **local evaluator**. Do not use the OpenAI backend for the competition's online validation/evaluation, because the competition rules prohibit cloud API calls during `/predict`.

## Vast quickstart for the OpenAI experiment

Checkout this branch on Vast:

```bash
cd /workspace/nordic-medical-local
git fetch origin
git switch experiment/openai-sol
git pull
```

Set the API key only in the shell environment. **Do not put a real key in Git, `.env.example`, a command screenshot, or chat.**

```bash
export OPENAI_API_KEY='YOUR_KEY_HERE'
export OPENAI_MODEL=gpt-5.6-sol
export OPENAI_REASONING=high

pkill -f "python3 api.py" || true
bash run_openai_experiment.sh
```

The experiment keeps Whisper `large-v3-turbo` on the Vast GPU, sends only the resulting transcript/questions to the OpenAI Responses API, and preserves the same structured answer/evidence schema used by the local model.

Confirm the active backend:

```bash
curl http://127.0.0.1:9054/api
```

It should include:

```json
{"service":"medical-appointment-usecase","status":"ok","llm_provider":"openai"}
```

Then run the official evaluator from your own computer:

```cmd
cd Nordic-AI-Cup-2026\medical-appointment
python local_evaluator.py --url http://194.26.196.159:15983/predict --verbose
```

To leave experiment mode, switch back to `main` and restart the normal Ollama deployment.

---

# Nordic AI Cup 2026 — Medical Appointment local starter

A local inference implementation for the **Medical Appointment** use case:

**MP3 → faster-whisper → local Qwen3 14B via Ollama → contrastive entailment → segment evidence → timestamp span**

No cloud API is used during `/predict`.

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

- `ASR_MODEL=large-v3`
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

1. Run the 390 supplied training questions and record score + worst-case latency.
2. Inspect errors by `positive`, `hard_negative`, and `off_topic`.
3. Compare `EVIDENCE_MODE=segment` against `word`.
4. Benchmark Whisper `large-v3` against `large-v3-turbo` if latency becomes tight.
5. Test Qwen3 8B against a larger local model only if latency and VRAM allow it.

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
