# syntax=docker/dockerfile:1.7
# =============================================================================
# zkLLM SmolLM2-360M — CUDA 12.4, targets H200 (sm_90)
# =============================================================================
# Build locally:
#   docker build -t zkllm-smollm2:local .
# Build in CI (no GPU on runner — nvcc cross-compiles for sm_90 just fine):
#   handled by .github/workflows/docker.yml
# Run on H200 host:
#   docker run --rm --gpus all -p 8000:8000 -v zkllm-data:/data zkllm-smollm2:local
# =============================================================================

ARG CUDA_VERSION=12.4.1
ARG SM_ARCH=sm_90

# ----------------------------------------------------------------------------
# Stage 1 — builder: compile zkllm + install Python deps + bake model weights
# ----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG SM_ARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        python3.11 python3.11-dev python3.11-venv python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# --- python env (venv so the runtime stage can copy /opt/venv cleanly) ---
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip wheel setuptools

# torch with CUDA 12.4 wheels
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

# --- python app deps ---
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- build zkllm (vendored, already pre-patched) ---
COPY zkllm/ /app/zkllm/
COPY scripts/ /app/scripts/
RUN test -f /app/zkllm/Makefile || (echo "ERROR: zkllm/Makefile missing. Did you run 'bash setup.sh' locally before committing? The zkllm/ folder must contain the vendored source, not a git submodule pointer." && exit 1)
RUN chmod +x /app/scripts/*.sh \
 && bash /app/scripts/build_zkllm.sh /app/zkllm "${SM_ARCH}"

# --- bake model weights into the image so first run doesn't download ---
COPY app/ /app/app/
ENV HF_HOME=/opt/hf-cache
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M'); \
AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M')"

# ----------------------------------------------------------------------------
# Stage 2 — runtime: slim CUDA runtime + copied venv + copied binaries
# ----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    ZKLLM_ROOT=/app/zkllm \
    ZKLLM_WORKDIR=/data/workdir \
    JOB_DIR=/data/jobs

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv libgomp1 curl \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python env, app code, compiled zkllm, and cached model weights
COPY --from=builder /opt/venv      /opt/venv
COPY --from=builder /opt/hf-cache  /opt/hf-cache
COPY --from=builder /app/app       /app/app
COPY --from=builder /app/zkllm     /app/zkllm
COPY --from=builder /app/scripts   /app/scripts

WORKDIR /app
RUN mkdir -p /data/workdir /data/jobs

EXPOSE 8000

# Healthcheck — /health reports CUDA availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]