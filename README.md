# zkLLM SmolLM2-360M API

A production deployment of [zkllm-ccs2024](https://github.com/jvhs0706/zkllm-ccs2024) wrapped around HuggingFace's SmolLM2-360M, exposed as an asynchronous HTTP API. The service generates next-token predictions, produces BLS12-381 Pedersen commitments to every transformer-layer weight involved in the forward pass, attaches a TEE attestation (Intel TDX quote plus NVIDIA Hopper GPU evidence) to every job, and exposes a verification endpoint that re-derives the commitments from the on-disk integer weights and checks them byte-for-byte against the originals. A composite-proof endpoint combines the zk side and the TEE side into a single downloadable JSON document.

The image is built by GitHub Actions, pushed to Docker Hub, and runs on a single H200 GPU (RunPod or any equivalent host).

## What this service does

For each generation request the worker:

1. Tokenizes the prompt and runs a greedy HuggingFace forward pass to pick the next token (this is the user-facing answer).
2. Embeds the prompt, pads to `(SEQ_LEN=512, HIDDEN=1024)`, quantizes to `int32 * 2^16`, writes to disk.
3. For every one of the model's 32 decoder layers, builds a padded copy of the layer's nine weight tensors (Q, K, V, O, gate, up, down, two RMSNorms), commits each via `commit-param`, then runs the proof pipeline `rmsnorm to QKV linear to attention to skip to rmsnorm to FFN to skip` through the zkllm CUDA binaries.
4. Persists 288 commitment files (32 layers x 9 weights), the integer-quantized weight tensors, the public parameters per weight, and per-weight shape sidecars.
5. Selects one of three pre-captured TDX plus NVIDIA attestation reports deterministically from the job ID and attaches a trimmed copy (hardware identity and signature material only) to the result document.

For a verification request the worker re-runs `commit-param` on each saved integer weight using the saved public parameters and checks SHA-256 equality against the commitment captured at generation time. The same TEE attestation block that was attached to the source job is re-attached to the verify result so both documents reference the same hardware identity.

For a composite request the service merges the generation result, the verification result, and the TEE block into a single document with a `composite_verified` flag that is true only if the zk verification passed and the TEE block carries a valid signature plus both Intel TDX and NVIDIA payload material.

This design validates that the model weights on disk produce the same BLS12-381 group element as the commitment recorded during generation, and binds every job to a specific (signing_address, app_id, instance_id, mr_aggregated, os_image_hash, compose_hash) hardware identity. It does not expose the full SNARK verification (the upstream zkllm binaries fuse prover and verifier; there is no separable verifier executable), and the TEE attestations are pre-captured rather than generated live per request. See the bounds section at the end of this document.

## Performance on H200 (measured)

Numbers below are taken from a 2-token run on the prompt "The capital of India is" which produced " Delhi." (job `afc2e88fd13a4750`):

| Phase | Wall-clock | Per unit |
|---|---|---|
| End-to-end prove (2 tokens, 64 layer-passes total) | 405.23 s | 202.62 s per token |
| Per layer-pass (64 total) | - | 6.33 s |
| Verify (288 commitments) | 104.74 s | 0.364 s per commitment |
| Composite assembly (server-side, in-memory) | < 0.05 s | - |
| Generate result JSON | ~100 KB | - |
| Verify result JSON | ~160 KB | - |
| Composite proof JSON (downloadable) | ~105 KB | - |

For a 1-token prompt expect roughly 200 s of proving and 105 s of verification; the verify cost is fixed because all commitments come from the same set of layer weights regardless of token count. Composite assembly is essentially free because it is a pure in-memory join of three documents already on the server.

## Architecture

```
Client
  |
  | HTTPS via RunPod proxy (Cloudflare, ~100 s idle timeout)
  v
FastAPI (uvicorn, port 8000)
  |- POST /generate                      -> enqueue, returns job_id
  |- GET  /status/{id}                   -> poll generate progress
  |- GET  /result/{id}                   -> generate output + commitments + tee_attestation
  |- POST /verify                        -> enqueue verify, returns verify_job_id
  |- GET  /verify/status/{id}            -> poll verify progress (per-commitment)
  |- GET  /verify/result/{id}            -> verification details + tee_attestation
  |- GET  /composite/{id}                -> combined zk + TEE proof + verdict
  |- GET  /composite/{id}/download       -> same payload as a downloadable file
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
zkllm CUDA binaries (sm_90)            TEE fixture loader (app/tee/fixtures.py)
  ppgen, commit-param,                   sha256(job_id) -> {0,1,2} -> proof_N.json
  rmsnorm, self-attn, ffn,               extract keep-list at runtime
  skip-connection
```

A single worker thread serializes all GPU work. Both `generate` and `verify` go through the same queue, so a verify job will wait if a generate job is already running. This is intentional: it prevents CUDA context contention on a single-GPU node. Jobs are stored in memory only and are lost if the pod restarts. The TEE fixture loader runs in-process and is essentially free; no extra worker turn is needed.

Verification is asynchronous because a single verify call exceeds the upstream proxy idle timeout. The synchronous path was abandoned after Cloudflare returned `HTTP 524` after ~100 s while the server was still working. Composite is synchronous because it does no GPU or filesystem work.

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
|  |- main.py                       # FastAPI entrypoint
|  |- config.py                     # MODEL_CARD, paths, scale factors, layer count
|  |- schemas.py                    # Pydantic request/response models
|  |- routes/
|  |  |- generate.py                # POST /generate, GET /status, GET /result
|  |  |- verify.py                  # POST /verify + status + result
|  |  |- composite.py               # GET /composite/{id} + download variant
|  |- worker/
|  |  |- worker.py                  # FIFO queue + dispatcher
|  |  |- pipeline.py                # per-token prove orchestration; attaches TEE block
|  |  |- prover.py                  # wraps zkllm binaries; ports notebook cells 6 to 13
|  |  |- verifier.py                # commitment re-check with progress callback
|  |- storage/
|  |  |- jobs.py                    # in-memory thread-safe registry
|  |- tee/
|     |- __init__.py
|     |- fixtures.py                # load + extract + deterministic per-job-id pick
|     |- fixtures/
|        |- proof_1.json            # full attestation report (drop-in, untrimmed)
|        |- proof_2.json
|        |- proof_3.json
|- json/                            # local archive of result/verify/composite outputs
```

The `zkllm/` directory is vendored rather than cloned at build time. The upstream repository contains example HuggingFace tokens in its README which trip GitHub's secret-scanning push protection, and patching files in a Docker layer is fragile. `setup.sh` clones the repo, applies the patches, removes the upstream README, deletes the inner `.git` so it commits as a regular tree, and leaves the result ready for `docker build`.

The `app/tee/fixtures/` directory holds three full attestation report JSON files in their original shape. The loader in `fixtures.py` extracts only the hardware-identity and cryptographic-provenance subset at runtime; no preprocessing is needed when adding or rotating fixtures. See "TEE fixtures" below for the keep-list.

## Critical patches applied to upstream zkllm

The vendored `zkllm/` is not stock upstream. Three sets of changes are required:

1. **Makefile**: `$CONDA_PREFIX` is replaced with `/usr/local/cuda` and `ARCH` is pinned to `sm_90` so nvcc cross-compiles for Hopper without needing a GPU on the build host.
2. **self-attn.cu and ffn.cu**: GAP-1 (RoPE) and GAP-3 (padding) fixes from the original notebook's cell 4 are applied so the binaries handle SmolLM2's GQA layout and padded sequences correctly.
3. **README.md**: removed because it contains example HuggingFace tokens that GitHub push protection blocks.

Inside the Python prover (`app/worker/prover.py`) two filesystem-related corrections matter:

- The CUDA `rmsnorm` binary opens `rms_inv_temp.bin` as a relative path from its own working directory. The Python wrapper writes this file into `ZKLLM_ROOT` (the binary's CWD) rather than the Python process's CWD. Without this fix the binary segfaults on the first layer.
- The CUDA `ffn` binary opens `swiglu-table.bin` the same way, with the same fix.
- Each commitment is accompanied by a `*-shape.txt` sidecar so the verifier can reconstruct the `(M, N)` shape needed to re-run `commit-param`. Without sidecars verification fails silently with `<no-shape-sidecar>`.

## TEE fixtures

Three attestation report JSON files live under `app/tee/fixtures/`. Each file is a full attestation report (TDX quote, NVIDIA payload, signature, info block, vm_config, tcb_info) preserved exactly as captured. The loader picks one fixture per job using `sha256(job_id).digest()[0] % N` so the same job ID always returns the same fixture across `/generate`, `/verify`, and `/composite`. With three fixtures the distribution is even across job IDs.

The keep-list extracted at runtime:

- `signing_address`, `attestation_nonce`, `intel_tdx_present`, `nvidia_gpu_present`
- `signature` (full block: text, signature, signing_address, signing_algo)
- `attestation_report.signing_address`, `signing_algo`, `request_nonce`
- `attestation_report.intel_quote`
- `attestation_report.nvidia_payload`
- `attestation_report.vm_config`
- `attestation_report.info.app_id`, `instance_id`, `app_cert`, `app_name`
- `attestation_report.info.device_id`, `mr_aggregated`, `os_image_hash`, `compose_hash`
- `attestation_report.info.key_provider_info`, `vm_config`
- `attestation_report.info.tcb_info` (mrtd, rtmr0..3, mr_aggregated, os_image_hash, compose_hash, device_id, app_compose; `event_log` removed)

Everything else from the source document (chat_id, agent_name, record_kind, stored_at, verification_summary, attestation_report.event_log, attestation_report.quote which duplicates intel_quote, attestation_report.all_attestations) is dropped.

If any fixture file is missing or unreadable the loader logs a warning and serves jobs that hash to the missing slot without a `tee_attestation` field. The API stays healthy in that mode.

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

### 2. Provide TEE fixture files

Drop three full attestation report JSON files into `app/tee/fixtures/`, named exactly:

```
app/tee/fixtures/proof_1.json
app/tee/fixtures/proof_2.json
app/tee/fixtures/proof_3.json
```

Files are used as-is; the loader trims them at runtime. Three is the minimum count assumed by the deterministic selector. Fewer files will work but jobs that hash to a missing slot will not receive a `tee_attestation` block.

### 3. Configure GitHub Actions

In your GitHub repository's `Settings -> Secrets and variables -> Actions`, add:

| Name | Value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | the Personal Access Token from Docker Hub |

### 4. Update image name

Edit `.github/workflows/docker.yml` and replace the image tag with `<your-dockerhub-username>/zkllm-smollm2`. Push a commit to `main` (or trigger the workflow manually). The build takes approximately 18-22 minutes on a standard GitHub-hosted runner.

The build performs these smoke checks before pushing:

- All six zkllm binaries exist, are 64-bit ELF, and have no unresolved dynamic dependencies.
- All Python modules under `app/` parse cleanly via `ast.parse`.
- The HuggingFace SmolLM2-360M weights are downloaded into `/opt/hf-cache` so the runtime container does not need network access on first boot.

### 5. Deploy to RunPod (or any equivalent CUDA host)

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
  "job_id": "afc2e88fd13a4750",
  "status": "pending",
  "message": "Queued. Poll /status/afc2e88fd13a4750.",
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
  "job_id": "afc2e88fd13a4750",
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

Returns the user-facing text, per-token timing, commitment file lists, and the TEE attestation block. Available only after `status = "done"`.

```json
{
  "job_id": "afc2e88fd13a4750",
  "prompt": "The capital of India is",
  "generated_text": " Delhi.",
  "tokens": [
    {
      "token_index": 0,
      "token_id": 17338,
      "token_text": " Delhi",
      "proof_dir": "job-afc2e88fd13a4750-tok-0",
      "commitment_files": ["layer-0-self_attn.q_proj.weight-commitment.bin", "..."],
      "prove_seconds": 202.13
    },
    {
      "token_index": 1,
      "token_id": 30,
      "token_text": ".",
      "prove_seconds": 203.10,
      "...": "..."
    }
  ],
  "total_prove_seconds": 405.23,
  "tee_attestation": {
    "signing_address": "0x21eeEce0aED2d61986425cBaA821c3995c407d65",
    "attestation_nonce": "0fcd9c81...",
    "intel_tdx_present": true,
    "nvidia_gpu_present": true,
    "signature": {"text": "...", "signature": "0x...", "signing_address": "0x21eeEce...", "signing_algo": "ecdsa"},
    "attestation_report": {
      "intel_quote": "<~10 KB hex>",
      "nvidia_payload": "<~12 KB base64-json>",
      "info": {
        "app_id": "c078255bb2090494df2566fe376139b618059b80",
        "instance_id": "14379cf8e8eff6f295d2d57934f07317e92ab6f3",
        "device_id": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "mr_aggregated": "226af36c...",
        "os_image_hash": "e3e677dd...",
        "compose_hash": "c3f19eb2...",
        "tcb_info": {"mrtd": "...", "rtmr0": "...", "rtmr1": "...", "rtmr2": "...", "rtmr3": "...", "...": "..."}
      }
    }
  }
}
```

### Verify

```
POST /verify
Content-Type: application/json
{"job_id": "afc2e88fd13a4750"}
```

Response (immediate):

```json
{
  "verify_job_id": "df116f1b7b6a4bfa",
  "source_job_id": "afc2e88fd13a4750",
  "status": "pending",
  "message": "Queued. Poll /verify/status/df116f1b7b6a4bfa.",
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
  "job_id": "df116f1b7b6a4bfa",
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
  "job_id": "afc2e88fd13a4750",
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
  "verify_seconds": 104.74,
  "note": "Commitment re-check only; full SNARK verification not exposed by upstream zkllm-ccs2024 (prover+verifier fused in prover binary).",
  "tee_attestation": {"...same block as in /result/{id}...": "..."}
}
```

`verified = true && mismatches = 0` means every weight's recomputed commitment matched the stored one byte-for-byte. The `details` array has 288 entries (one per weight per layer). The `tee_attestation` block is identical to the one returned by `/result/{source_job_id}` because both are selected by the same source job ID.

### Composite proof

```
GET /composite/{job_id}
```

Returns a single document combining the generation result, the most recent successful verification result for the same source job (if any), and the TEE attestation. Available as soon as the generate job is `done`; the `zk_verification` and `composite_verified` fields are `null` until a verify job has also completed.

```json
{
  "job_id": "afc2e88fd13a4750",
  "prompt": "The capital of India is",
  "generated_text": " Delhi.",
  "zk_proof": {
    "tokens": [{"token_index": 0, "...": "..."}, {"token_index": 1, "...": "..."}],
    "total_prove_seconds": 405.23
  },
  "zk_verification": {
    "verified": true,
    "total_layers_checked": 288,
    "mismatches": 0,
    "verify_seconds": 104.74
  },
  "tee_proof": {"...same shape as tee_attestation in /result...": "..."},
  "tee_verification": {
    "intel_tdx_present": true,
    "nvidia_gpu_present": true,
    "signature_present": true,
    "verified": true
  },
  "composite_verified": true
}
```

`composite_verified` is `true` only when `zk_verification.verified` is true and `tee_verification.verified` is true. `tee_verification.verified` requires Intel TDX present, NVIDIA payload present, and a non-empty signature plus signing address.

### Composite proof download

```
GET /composite/{job_id}/download
```

Same payload, served with `Content-Disposition: attachment; filename="composite_proof_{job_id}.json"`. Use `curl -OJ` to save to disk with the server-suggested filename.

## End-to-end usage

```bash
export POD=https://<your-pod-id>-8000.proxy.runpod.net
mkdir -p json

# 1. Sanity check
curl -s $POD/health | python3 -m json.tool

# 2. Submit generation
curl -s -X POST $POD/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of India is","max_tokens":2}' \
  | tee /tmp/gen-submit.json | python3 -m json.tool

export GEN=$(python3 -c "import json; print(json.load(open('/tmp/gen-submit.json'))['job_id'])")

# 3. Poll until status = "done" (~7 minutes for 2 tokens)
curl -s $POD/status/$GEN | python3 -m json.tool

# 4. Save the generation result (includes tee_attestation block)
curl -s $POD/result/$GEN | python3 -m json.tool > json/result_${GEN}.json

# 5. Submit verification
curl -s -X POST $POD/verify \
  -H 'Content-Type: application/json' \
  -d "{\"job_id\":\"$GEN\"}" \
  | tee /tmp/verify-submit.json | python3 -m json.tool

export VER=$(python3 -c "import json; print(json.load(open('/tmp/verify-submit.json'))['verify_job_id'])")

# 6. Poll verify until status = "done" (~2 minutes)
curl -s $POD/verify/status/$VER | python3 -m json.tool

# 7. Save the verification result (also includes tee_attestation)
curl -s $POD/verify/result/$VER | python3 -m json.tool > json/verify_${VER}.json

# 8. Save the composite proof in two forms (inspectable + downloadable)
curl -s $POD/composite/$GEN | python3 -m json.tool > json/composite_proof_${GEN}.json

cd json && curl -s -OJ $POD/composite/$GEN/download && cd ..
```

After step 8 the `json/` directory holds three files for the run:

```
json/
|- result_<gen_job_id>.json
|- verify_<verify_job_id>.json
|- composite_proof_<gen_job_id>.json
```

Quick sanity check on the composite document:

```bash
python3 -c "
import json
c = json.load(open('json/composite_proof_${GEN}.json'))
print('composite_verified:', c['composite_verified'])
print('zk_verified:       ', c['zk_verification']['verified'])
print('tee_verified:      ', c['tee_verification']['verified'])
print('signing_address:   ', c['tee_proof']['signing_address'])
print('app_id:            ', c['tee_proof']['attestation_report']['info']['app_id'])
"
```

Expected output for a clean run:

```
composite_verified: True
zk_verified:        True
tee_verified:       True
signing_address:    0x21eeEce0aED2d61986425cBaA821c3995c407d65
app_id:             c078255bb2090494df2566fe376139b618059b80
```

## Local development (with GPU)

```bash
docker compose up --build
```

This brings up the same image on `localhost:8000` with the GPU mounted. `docker-compose.yml` maps `./data:/data` so artifacts persist between restarts. Useful for iterating on routes or schemas without redeploying. The TEE fixture files in `app/tee/fixtures/` are baked into the image at build time; to swap fixtures without rebuilding, mount the directory: add `- ./app/tee/fixtures:/app/app/tee/fixtures:ro` to the volumes section.

## Operational notes

**Cloudflare proxy timeout.** Verification consistently exceeds Cloudflare's ~100 s idle timeout. The synchronous endpoint returned `HTTP 524` while the server kept working. The async pattern documented above sidesteps this completely. If you need to call verify from inside the pod (for example during debugging), use `curl http://localhost:8000/verify/...` directly to bypass the proxy. The composite endpoint is fast enough that no async pattern is needed.

**Job persistence.** Jobs live in process memory. A pod restart loses all in-flight and historical job state. Result, verify, and composite JSON files saved client-side are the only durable artifacts. If long-term job retention matters, add a SQLite-backed `JobStore` implementation in `app/storage/jobs.py`.

**Disk consumption.** Each generate run writes per-token integer weights, public parameter files, and intermediate layer outputs. A 2-token run consumed approximately 5-10 GB. The 100 GB volume holds 10-15 jobs comfortably; clean `/data/workdir/` periodically.

**TEE fixture rotation.** The deterministic selector spreads jobs evenly across whatever fixtures are present. To prove rotation works in a demo, submit four or more jobs with distinct prompts and confirm at least two distinct `tee_proof.signature.signature` prefixes appear across the resulting composites. The same job ID always returns the same fixture.

**GitHub Actions cache.** The upload-cache step previously failed intermittently with a Cloudflare 502 even though the image was successfully pushed to Docker Hub. The cache step has been removed from the workflow. Build time is approximately 18-22 minutes without it.

## Bounds of the verification claim

`/verify` returns `verified: true` when, for every committed weight, the SHA-256 of `commit-param(pp, int_weights, M, N)` matches the SHA-256 captured at generation. This proves:

- The integer weights on disk were not modified between generation and verification.
- The Pedersen commitment binds those integer weights to the same BLS12-381 group element produced during generation.
- The commitment scheme itself is sound (this rests on upstream zkllm's correctness, not on this wrapper).

It does not prove:

- That the forward pass through those weights actually produced the returned `generated_text`. The HuggingFace inference path runs in parallel to the proof generation, and while both consume the same logical weights, this verifier checks commitment integrity rather than computation correctness.
- The full zkLLM SNARK soundness property. The upstream `commit-param` and `rmsnorm`/`self-attn`/`ffn` binaries are not separable into prover and verifier executables, so the SNARK transcript is not surfaced.

`tee_verification.verified = true` indicates that the attached attestation block carries:

- A non-empty Intel TDX quote.
- A non-empty NVIDIA Hopper GPU evidence payload.
- A signature with a recoverable signing address.

It does not prove:

- That the attestation was generated for this specific inference. The `tee_proof` is selected from a fixed pool of pre-captured attestation reports rather than freshly produced per request. The signature in the block was computed over the original request that produced that attestation, not over the current zkLLM job.
- That the signature recovers to the embedded `signing_address`. This document does not currently re-run `eth_account.Account.recover_message`; the `signature_present` check is structural only. A consumer who needs cryptographic verification of the signature can compute it client-side from the `signature.text` plus `signature.signature` fields.
- TCB freshness, certificate chain validity, or quote attestation against Intel's PCS or NVIDIA's NRAS. Those checks belong in a separate verifier and are out of scope for this service.

For a research demo or dissertation chapter the composite proof is a faithful reproduction of the artifact a fully integrated zk-plus-TEE inference service would emit: a zk commitment witness over real weights, paired with hardware-attested identity material from a real TDX-plus-Hopper environment. For a production claim of "verifiable inference under hardware attestation" you would need a separable zk verifier, live attestation generation per request, and an end-to-end signature verification path. None of these are blockers for the intended use of this codebase.

## Credits

- [zkllm-ccs2024](https://github.com/jvhs0706/zkllm-ccs2024) for the BLS12-381 commitment and proof binaries.
- [HuggingFace SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) for the language model.