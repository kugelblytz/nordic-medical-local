# Nordic AI Cup 2026 — Medical Appointment local starter

A local inference implementation for the **Medical Appointment** use case:

**MP3 → faster-whisper → local Qwen3 via Ollama → yes/no → evidence quote → timestamp span**

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

The supplied training annotations have positive evidence spans averaging about 3.2 seconds (median about 2.9 seconds), so the default `EVIDENCE_MODE=segment` returns the Whisper utterance containing the supporting quote. You can validate `EVIDENCE_MODE=word` as an alternative.

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
ollama pull qwen3:8b
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
- `OLLAMA_MODEL=qwen3:8b`
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
4. Test Whisper `large-v3-turbo` vs `large-v3`.
5. Test Qwen3 8B against a larger local model only if latency and VRAM allow it.

The competition rules prohibit cloud APIs during inference, so both ASR and question answering stay local.
