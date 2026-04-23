#!/usr/bin/env bash
set -euxo pipefail

# Build the 6 zkllm CUDA binaries. Assumes:
#   - we are in /app/zkllm (or path passed as $1)
#   - CUDA is at /usr/local/cuda
#   - target SM arch is passed as $2 (default sm_90 for H200)
# The Makefile was pre-patched at vendor-time to use /usr/local/cuda paths.

ZKLLM_DIR="${1:-/app/zkllm}"
ARCH="${2:-sm_90}"

cd "$ZKLLM_DIR"

# Ensure makefile has the right arch (idempotent)
sed -i "s/^ARCH := .*/ARCH := ${ARCH}/" Makefile || true

make clean
make ppgen commit-param rmsnorm self-attn ffn skip-connection
ls -lh ppgen commit-param rmsnorm self-attn ffn skip-connection
