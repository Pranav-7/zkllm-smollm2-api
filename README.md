# zkLLM SmolLM2-360M — FastAPI + Docker

Serves SmolLM2-360M through a FastAPI endpoint that generates text **and** a zkLLM zero-knowledge proof of the forward pass, running on H200 (Hopper, `sm_90`). Based on [`jvhs0706/zkllm-ccs2024`](https://github.com/jvhs0706/zkllm-ccs2024).

## ⚠️ Time cost — read this first

Proof generation is **expensive**:

| `max_tokens` | Approx time on H200 | Approx disk used |
|---|---|---|
| 1  | ~15–30 min  | ~2 GB  |
| 5  | ~1.5–2.5 hr | ~10 GB |
| 10 | ~3–5 hr     | ~20 GB |

One job occupies the GPU fully. Queue is strict FIFO, one worker. No concurrency.

## Architecture

```
Client ──HTTP──▶ FastAPI (async handlers)
                     │
                     ▼
              In-memory job dict ◀──── Background worker thread
                                               │
                                               ▼
                         ┌────────── ownership ───────────┐
                         │      HF SmolLM2-360M (GPU)     │
                         │      + zkllm CUDA binaries     │
                         └────────────────────────────────┘
```

Jobs are held in memory — **they are lost on container restart** (by design, per your spec).

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Queue a job. Returns `job_id` immediately. |
| `GET`  | `/status/{job_id}` | Poll job progress (`pending` / `running` / `done` / `failed`). |
| `GET`  | `/result/{job_id}` | Fetch final text + proof artifacts. 409 until `done`. |
| `POST` | `/verify` | Re-check commitments for a completed job. |
| `GET`  | `/health` | Liveness + CUDA availability. |
| `GET`  | `/docs` | Swagger UI. |

### Example

```bash
# 1. Submit a job
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The French Revolution began in", "max_tokens":1}'
# → {"job_id":"a1b2c3...","status":"pending","estimated_seconds":960, ...}

# 2. Poll
curl http://localhost:8000/status/a1b2c3...
# → {"status":"running","progress":{"tokens_done":0,"layer":7,"phase":"tok0_layer7_ffn"}, ...}

# 3. Fetch result
curl http://localhost:8000/result/a1b2c3...
# → {"generated_text":" 1789","tokens":[{..."commitment_files":[...]}], ...}

# 4. Verify
curl -X POST http://localhost:8000/verify \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"a1b2c3..."}'
# → {"verified":true,"total_layers_checked":224,"mismatches":0, ...}
```

## What `/verify` actually checks

The upstream zkllm-ccs2024 interleaves prover and verifier inside the same CUDA binary, so we cannot expose a standalone SNARK verifier without rewriting the library. `/verify` therefore does the strongest thing currently separable:

1. SHA-256 the stored `*-commitment.bin` files from the job.
2. Re-run `./commit-param` on the stored int-quantized weight files.
3. Byte-compare the recomputed commitment against the stored one.

If byte-equal, the Pedersen (BLS12-381) commitment binds the same weights that were proven. This catches any tampering with weights or commitments after the fact. It does **not** verify the sumcheck transcripts end-to-end.

## Setup

### 1. Vendor zkllm once (local machine)

```bash
git clone <this-repo> zkllm-smollm2-api
cd zkllm-smollm2-api
bash scripts/vendor_zkllm.sh        # clones + patches zkllm-ccs2024 into ./zkllm/
git add zkllm && git commit -m "Vendor patched zkllm"
```

### 2. Set Docker Hub secrets in GitHub

In your repo → **Settings → Secrets and variables → Actions → New repository secret**:

- `DOCKERHUB_USERNAME` — your Docker Hub username
- `DOCKERHUB_TOKEN` — a Docker Hub access token (not your password)

### 3. Push

```bash
git push origin main
```

The workflow `.github/workflows/docker.yml` will:

1. Free disk on the runner (the image is 8–10 GB).
2. Build `linux/amd64` with `CUDA_VERSION=12.4.1`, `SM_ARCH=sm_90`.
3. Push to `pranav6773/zkllm-smollm2` with tags:
   - `main`, `<short-sha>` on main pushes
   - `vX.Y.Z`, `vX.Y`, `latest` on semver tags

### 4. Run on H200

```bash
docker run -d --name zkllm \
  --gpus all \
  -p 8000:8000 \
  -v zkllm-data:/data \
  pranav6773/zkllm-smollm2:latest

# Smoke test (CUDA must show true)
curl http://localhost:8000/health
```

The image includes the SmolLM2-360M weights baked in, so no download at first run. `/data` volume holds proof artifacts; size it for ~20 GB per concurrent max-tokens-10 job plus ~2 GB of one-time public parameters.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CARD` | `HuggingFaceTB/SmolLM2-360M` | HF model id |
| `ZKLLM_WORKDIR` | `/data/workdir` | proof artifact directory |
| `JOB_DIR` | `/data/jobs` | job state directory (currently unused — in-memory) |
| `MAX_TOKENS_HARD_CAP` | `10` | upper bound on `max_tokens` in requests |
| `DEFAULT_MAX_TOKENS` | `1` | default `max_tokens` when omitted |
| `N_LAYERS_PROVE` | `32` | layers to prove per forward pass (32 = full model) |

## Repo layout

```
.
├── Dockerfile                       # multi-stage, builder compiles zkllm
├── docker-compose.yml               # local GPU test
├── .dockerignore
├── .github/workflows/docker.yml     # build + push to Docker Hub
├── requirements.txt
├── README.md
├── app/
│   ├── main.py                      # FastAPI entrypoint
│   ├── config.py                    # constants from notebook
│   ├── schemas.py                   # Pydantic models
│   ├── routes/{generate,verify}.py
│   ├── worker/
│   │   ├── worker.py                # FIFO queue + thread
│   │   ├── pipeline.py              # per-token proving loop
│   │   ├── prover.py                # port of notebook cells 6,8,11
│   │   └── verifier.py              # commitment re-check
│   └── storage/jobs.py              # in-memory job registry
├── scripts/
│   ├── vendor_zkllm.sh              # run once, pre-patches upstream repo
│   └── build_zkllm.sh               # `make` inside the Docker build
└── zkllm/                           # vendored, patched zkllm-ccs2024
```

## Known limits (honest list)

- **In-memory jobs**: a container restart during a 4-hour job loses the job. You accepted this trade-off.
- **GAP-1 RoPE approximation**: upstream zkllm's CUDA binary does not apply rotary embeddings inside the attention proof. The Python reference applies RoPE for logits the user sees, but the proven computation uses un-rotated Q/K. Logit cosine-similarity is ~0.95 vs exact HF (see notebook cell 18).
- **GAP-2 GQA expansion**: KV heads are expanded 5→16 before commitment. Inference-equivalent, but what's committed is a widened MHA form.
- **GAP-3 padding**: 6.25% of hidden, 37.5% of intermediate dims are zero-padded. Committed, but semantically null.
- **Serial**: one GPU, one job. For concurrent users, add more containers behind a load balancer — do **not** share a GPU across jobs (zkllm uses most of VRAM).
- **`max_tokens ≤ 10`** hard cap — see time table above.

## Local development (no GPU)

You can't actually run the prover without CUDA, but you can exercise the FastAPI layer:

```bash
pip install -r requirements.txt fastapi[standard]
# Comment out prover imports in app/worker/pipeline.py for a dry run,
# or use the /docs UI to inspect schemas without calling endpoints.
```

For real testing, use `docker compose up --build` on a GPU host.
