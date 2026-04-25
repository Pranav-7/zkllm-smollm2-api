# zkLLM SmolLM2-360M API

A production deployment of [zkllm-ccs2024](https://github.com/jvhs0706/zkllm-ccs2024) wrapped around HuggingFace's SmolLM2-360M, exposed as an asynchronous HTTP API. The service generates next-token predictions and produces BLS12-381 Pedersen commitments to every transformer-layer weight involved in the forward pass, plus a verification endpoint that re-derives those commitments from the on-disk integer weights and checks them byte-for-byte against the originals.

The image is built by GitHub Actions, pushed to Docker Hub, and runs on a single H200 GPU (RunPod or any equivalent host).

## What this service does

For each user request the worker:

1. Tokenizes the prompt and runs a greedy HuggingFace forward pass to pick the next token (this is the user-facing answer).
2. Embeds the prompt, pads to `(SEQ_LEN=512, HIDDEN=1024)`, quantizes to `int32 * 2^16`, writes to disk.
3. For every one of the model's 32 decoder layers, builds a padded copy of the layer's nine weight tensors (Q, K, V, O, gate, up, down, two RMSNorms), commits each via `commit-param`, then runs the proof pipeline `rmsnorm to QKV linear to attention to skip to rmsnorm to FFN to skip` through the zkllm CUDA binaries.
4. Persists 288 commitment files (32 layers x 9 weights), the integer-quantized weight tensors, the public parameters per weight, and per-weight shape sidecars.

For a verification request the worker re-runs `commit-param` on each saved integer weight using the saved public parameters and checks SHA-256 equality against the commitment captured at generation time.

This validates that the model weights on disk produce the same BLS12-381 group element as the commitment recorded during generation. It does not expose the full SNARK verification (the upstream zkllm binaries fuse prover and verifier; there is no separable verifier executable).

## Performance on H200 (measured)

Numbers below are taken from a single 2-token run on the prompt "The capital of India is" which produced " Delhi." (job `ba0adc39a60c423a`):

| Phase | Wall-clock | Per unit |
|---|---|---|
| End-to-end prove (2 tokens, 64 layer-passes total) | 426.19 s | 213.10 s per token |
| Per layer-pass (64 total) | - | 6.66 s |
| Verify (288 commitments) | 111.16 s | 0.386 s per commitment |
| Generate result JSON | 38 KB | - |
| Verify result JSON | 98 KB | - |

For a 1-token prompt expect roughly 213 s of proving and 111 s of verification; the verify cost is fixed because all commitments come from the same set of layer weights regardless of token count.

## Architecture

```
Client
  |
  | HTTPS via RunPod proxy (Cloudflare, ~100 s idle timeout)
  v
FastAPI (uvicorn, port 8000)
  |- POST /generate              -> enqueue, returns job_id
  |- GET  /status/{id}           -> poll generate progress
  |- GET  /result/{id}           -> generate output + commitment file list
  |- POST /verify                -> enqueue verify, returns verify_job_id
  |- GET  /verify/status/{id}    -> poll verify progress (per-commitment)
  |- GET  /verify/result/{id}    -> verification details
  |
  v
In-memory job store (thread-safe dict)
  |
  v
Single FIFO worker thread (owns the GPU)
  |- generate jobs -> pipeline.run_job
  |- verify jobs   -> verifier.verify_job_commitments
  |
  v
zkllm CUDA binaries (sm_90)
  ppgen, commit-param, rmsnorm, self-attn, ffn, skip-connection
```

A single worker thread serializes all GPU work. Both `generate` and `verify` go through the same queue, so a verify job will wait if a generate job is already running. This is intentional: it prevents CUDA context contention on a single-GPU node. Jobs are stored in memory only and are lost if the pod restarts.

Verification is asynchronous because a single verify call exceeds the upstream proxy idle timeout. The synchronous path was abandoned after Cloudflare returned `HTTP 524` after ~100 s while the server was still working.

## Repository layout

```
zkllm-smollm2-api/
|- Dockerfile                       # multi-stage, CUDA 12.4 devel -> runtime
|- .dockerignore
|- requirements.txt                 # fastapi, transformers==4.46.2, torch==2.5.1, ...
|- docker-compose.yml               # local GPU smoke test
|- setup.sh                         # one-shot vendoring of zkllm source
|- .github/workflows/docker.yml     # build and push to Docker Hub
|- scripts/
|  |- vendor_zkllm.sh               # clone, patch, strip .git
|  |- build_zkllm.sh                # invoked by Dockerfile, runs make
|- zkllm/                           # vendored upstream source, pre-patched
|  |- Makefile                      # ARCH := sm_90, CUDA path /usr/local/cuda
|  |- self-attn.cu, ffn.cu          # patched per notebook cell 4
|  |- ppgen.cu, commit-param.cu, rmsnorm.cu, skip-connection.cu, ...
|- app/
   |- main.py                       # FastAPI entrypoint
   |- config.py                     # MODEL_CARD, paths, scale factors, layer count
   |- schemas.py                    # Pydantic request/response models
   |- routes/
   |  |- generate.py                # POST /generate, GET /status, GET /result
   |  |- verify.py                  # POST /verify + status + result
   |- worker/
   |  |- worker.py                  # FIFO queue + dispatcher
   |  |- pipeline.py                # per-token prove orchestration
   |  |- prover.py                  # wraps zkllm binaries; ports notebook cells 6 to 13
   |  |- verifier.py                # commitment re-check with progress callback
   |- storage/
      |- jobs.py                    # in-memory thread-safe registry
```

The `zkllm/` directory is vendored rather than cloned at build time. The upstream repository contains example HuggingFace tokens in its README which trip GitHub's secret-scanning push protection, and patching files in a Docker layer is fragile. `setup.sh` clones the repo, applies the patches, removes the upstream README, deletes the inner `.git` so it commits as a regular tree, and leaves the result ready for `docker build`.

## Critical patches applied to upstream zkllm

The vendored `zkllm/` is not stock upstream. Three sets of changes are required:

1. **Makefile**: `$CONDA_PREFIX` is replaced with `/usr/local/cuda` and `ARCH` is pinned to `sm_90` so nvcc cross-compiles for Hopper without needing a GPU on the build host.
2. **self-attn.cu and ffn.cu**: GAP-1 (RoPE) and GAP-3 (padding) fixes from the original notebook's cell 4 are applied so the binaries handle SmolLM2's GQA layout and padded sequences correctly.
3. **README.md**: removed because it contains example HuggingFace tokens that GitHub push protection blocks.

Inside the Python prover (`app/worker/prover.py`) two filesystem-related corrections matter:

- The CUDA `rmsnorm` binary opens `rms_inv_temp.bin` as a relative path from its own working directory. The Python wrapper writes this file into `ZKLLM_ROOT` (the binary's CWD) rather than the Python process's CWD. Without this fix the binary segfaults on the first layer.
- The CUDA `ffn` binary opens `swiglu-table.bin` the same way, with the same fix.
- Each commitment is accompanied by a `*-shape.txt` sidecar so the verifier can reconstruct the `(M, N)` shape needed to re-run `commit-param`. Without sidecars verification fails silently with `<no-shape-sidecar>`.

## Prerequisites

Local machine (one-time, for vendoring and pushing):

- Git
- Bash (WSL on Windows works fine)
- A GitHub repository (empty)
- A Docker Hub account and a Personal Access Token with `Read, Write, Delete` permissions

Runtime host:

- A CUDA 12.x capable GPU. The image is compiled for `sm_90` (H200, H100). On older SMs (A100/A40/A6000) the binaries should still execute via PTX JIT, but this has not been tested.
- 100 GB persistent volume mounted at `/data` for proof artifacts. A 2-token run produces approximately 5-10 GB of intermediate files in `/data/workdir/`.

You do not need a GPU on the build host. nvcc cross-compiles fine on a CPU-only GitHub Actions runner.

## Setup

### 1. Vendor zkllm

From a fresh clone of this repository:

```bash
bash setup.sh
```

This populates `zkllm/` with the patched upstream source. It is a one-time operation. The resulting tree (~50 source files, ~7,500 lines) should be committed to your repository.

### 2. Configure GitHub Actions

In your GitHub repository's `Settings -> Secrets and variables -> Actions`, add:

| Name | Value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | the Personal Access Token from Docker Hub |

### 3. Update image name

Edit `.github/workflows/docker.yml` and replace the image tag with `<your-dockerhub-username>/zkllm-smollm2`. Push a commit to `main` (or trigger the workflow manually). The build takes approximately 18-22 minutes on a standard GitHub-hosted runner.

The build performs these smoke checks before pushing:

- All six zkllm binaries exist, are 64-bit ELF, and have no unresolved dynamic dependencies.
- All Python modules under `app/` parse cleanly via `ast.parse`.
- The HuggingFace SmolLM2-360M weights are downloaded into `/opt/hf-cache` so the runtime container does not need network access on first boot.

### 4. Deploy to RunPod (or any equivalent CUDA host)

On RunPod's "Deploy" page, choose an H200 instance and override the template:

| Field | Value |
|---|---|
| Container Image | `<your-username>/zkllm-smollm2:main` |
| Start Command | (leave empty; `Dockerfile` defines `CMD`) |
| Container Disk | 40 GB |
| Volume Disk | 100 GB |
| Volume Mount Path | `/data` |
| Expose HTTP Ports | `8000` |

Optional environment overrides:

| Key | Default | Purpose |
|---|---|---|
| `MAX_TOKENS_HARD_CAP` | 10 | server-side ceiling on `max_tokens` |
| `DEFAULT_MAX_TOKENS` | 1 | used if request omits `max_tokens` |
| `N_LAYERS_PROVE` | 32 | layers to prove per token |

The pod takes 2-5 minutes to start on first boot (image pull + model load). Health check at `GET /health` will report `cuda_available: true` once ready.

## API reference

### Health

```
GET /health
```

Response:

```json
{"ok": true, "cuda_available": true, "cuda_device_count": 1}
```

### Generate

```
POST /generate
Content-Type: application/json
{
  "prompt": "The capital of India is",
  "max_tokens": 2,
  "deterministic": true
}
```

Response (immediate, non-blocking):

```json
{
  "job_id": "ba0adc39a60c423a",
  "status": "pending",
  "message": "Queued. Poll /status/ba0adc39a60c423a.",
  "estimated_seconds": 1920
}
```

`deterministic` must be `true`. Sampling is rejected because proofs are tied to a specific forward pass.

### Generate status

```
GET /status/{job_id}
```

Response:

```json
{
  "job_id": "ba0adc39a60c423a",
  "status": "running",
  "progress": {
    "tokens_done": 1,
    "tokens_total": 2,
    "layer": 26,
    "phase": "tok1_layer26_commit"
  },
  "started_at": 1777138931.38,
  "updated_at": 1777139631.75,
  "error": null
}
```

The `phase` field cycles through `commit`, `rmsnorm1`, `attn_linear`, `attn_rope_ref`, `attn_proof`, `skip1`, `rmsnorm2`, `ffn`, `skip2` for each layer. `status` transitions: `pending -> running -> done | failed`.

### Generate result

```
GET /result/{job_id}
```

Returns the user-facing text, per-token timing, and commitment file lists. Available only after `status = "done"`.

```json
{
  "job_id": "ba0adc39a60c423a",
  "prompt": "The capital of India is",
  "generated_text": " Delhi.",
  "tokens": [
    {
      "token_index": 0,
      "token_id": 17338,
      "token_text": " Delhi",
      "proof_dir": "job-ba0adc39a60c423a-tok-0",
      "commitment_files": ["layer-0-self_attn.q_proj.weight-commitment.bin", "..."],
      "prove_seconds": 213.37
    },
    {
      "token_index": 1,
      "token_id": 30,
      "token_text": ".",
      "prove_seconds": 212.82,
      "...": "..."
    }
  ],
  "total_prove_seconds": 426.19
}
```

### Verify

```
POST /verify
Content-Type: application/json
{"job_id": "ba0adc39a60c423a"}
```

Response (immediate):

```json
{
  "verify_job_id": "b99211154a054f26",
  "source_job_id": "ba0adc39a60c423a",
  "status": "pending",
  "message": "Queued. Poll /verify/status/b99211154a054f26.",
  "estimated_seconds": 576
}
```

### Verify status

```
GET /verify/status/{verify_job_id}
```

Response:

```json
{
  "job_id": "b99211154a054f26",
  "status": "running",
  "progress": {
    "weights_checked": 211,
    "weights_total": 288,
    "phase": "layer-23-self_attn.o_proj.weight"
  }
}
```

### Verify result

```
GET /verify/result/{verify_job_id}
```

Response:

```json
{
  "job_id": "ba0adc39a60c423a",
  "verified": true,
  "total_layers_checked": 288,
  "mismatches": 0,
  "details": [
    {
      "layer": 0,
      "weight_name": "self_attn.q_proj.weight",
      "expected_commitment_sha256": "7f2d5b30c0ffcd5e413b02df0dfe5762bd087d9e796fb39b982ae835f00af1e8",
      "recomputed_commitment_sha256": "7f2d5b30c0ffcd5e413b02df0dfe5762bd087d9e796fb39b982ae835f00af1e8",
      "match": true
    }
  ],
  "verify_seconds": 111.16,
  "note": "Commitment re-check only; full SNARK verification not exposed by upstream zkllm-ccs2024 (prover+verifier fused in prover binary)."
}
```

`verified = true && mismatches = 0` means every weight's recomputed commitment matched the stored one byte-for-byte. The `details` array has 288 entries (one per weight per layer).

## End-to-end usage

```bash
export POD=https://<your-pod-id>-8000.proxy.runpod.net
mkdir -p json

# 1. Sanity check
curl $POD/health

# 2. Submit generation
curl -X POST $POD/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of India is","max_tokens":2}'
# returns {"job_id":"...", ...}

export GEN=<job_id_from_above>

# 3. Poll until status = "done" (~7 minutes for 2 tokens)
curl -s $POD/status/$GEN | python3 -m json.tool

# 4. Save the generation result
curl -s $POD/result/$GEN | python3 -m json.tool > json/${GEN}-result.json

# 5. Submit verification
curl -s -X POST $POD/verify \
  -H 'Content-Type: application/json' \
  -d "{\"job_id\":\"$GEN\"}" \
  | tee /tmp/verify-submit.json | python3 -m json.tool

export VER=$(python3 -c "import json; print(json.load(open('/tmp/verify-submit.json'))['verify_job_id'])")

# 6. Poll verify until status = "done" (~2 minutes)
curl -s $POD/verify/status/$VER | python3 -m json.tool

# 7. Save the verification result
curl -s $POD/verify/result/$VER | python3 -m json.tool > json/${GEN}-verify.json
```

After step 7 you have two JSON files in `json/` named by the source job ID, suitable for archival or inclusion in a report.

## Local development (with GPU)

```bash
docker compose up --build
```

This brings up the same image on `localhost:8000` with the GPU mounted. `docker-compose.yml` maps `./data:/data` so artifacts persist between restarts. Useful for iterating on routes or schemas without redeploying.

## Operational notes

**Cloudflare proxy timeout.** Verification consistently exceeds Cloudflare's ~100 s idle timeout. The synchronous endpoint returned `HTTP 524` while the server kept working. The async pattern documented above sidesteps this completely. If you need to call verify from inside the pod (for example during debugging), use `curl http://localhost:8000/verify/...` directly to bypass the proxy.

**Job persistence.** Jobs live in process memory. A pod restart loses all in-flight and historical job state. Result and verify JSON files saved client-side are the only durable artifacts. If long-term job retention matters, add a SQLite-backed `JobStore` implementation in `app/storage/jobs.py`.

**Disk consumption.** Each generate run writes per-token integer weights, public parameter files, and intermediate layer outputs. A 2-token run consumed approximately 5-10 GB. The 100 GB volume holds 10-15 jobs comfortably; clean `/data/workdir/` periodically.

**GitHub Actions cache.** The upload-cache step previously failed intermittently with a Cloudflare 502 even though the image was successfully pushed to Docker Hub. The cache step has been removed from the workflow. Build time is approximately 18-22 minutes without it.

## Bounds of the verification claim

`/verify` returns `verified: true` when, for every committed weight, the SHA-256 of `commit-param(pp, int_weights, M, N)` matches the SHA-256 captured at generation. This proves:

- The integer weights on disk were not modified between generation and verification.
- The Pedersen commitment binds those integer weights to the same BLS12-381 group element produced during generation.
- The commitment scheme itself is sound (this rests on upstream zkllm's correctness, not on this wrapper).

It does not prove:

- That the forward pass through those weights actually produced the returned `generated_text`. The HuggingFace inference path runs in parallel to the proof generation, and while both consume the same logical weights, this verifier checks commitment integrity rather than computation correctness.
- The full zkLLM SNARK soundness property. The upstream `commit-param` and `rmsnorm`/`self-attn`/`ffn` binaries are not separable into prover and verifier executables, so the SNARK transcript is not surfaced.

For a research demo or dissertation chapter this is honest: weights are committed and the commitment is reproducible. For a production claim of "verifiable inference" you would need a separable verifier, which requires upstream changes to zkllm-ccs2024 that the original author has stated will not happen.

## Credits

- [zkllm-ccs2024](https://github.com/jvhs0706/zkllm-ccs2024) for the BLS12-381 commitment and proof binaries.
- [HuggingFace SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) for the language model.
- The notebook `SMOLM2_360M_V2_With_Inference.ipynb` for the original padding, GAP-1/2/3 reference logic, and per-layer prove orchestration.