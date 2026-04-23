#!/usr/bin/env bash
# =============================================================================
# One-shot setup: vendor zkllm-ccs2024 into ./zkllm/ with all notebook patches
# applied. Run this ONCE after unzipping. No git operations — you handle those.
#
# Requires: git, python3, internet.
#
# Usage:
#   bash setup.sh
# =============================================================================
set -euo pipefail

echo "==> Vendoring zkllm-ccs2024 into ./zkllm/ ..."
bash scripts/vendor_zkllm.sh

echo ""
echo "============================================================"
echo " Vendoring complete. ./zkllm/ is now populated and patched."
echo " Next steps are git commands — see instructions."
echo "============================================================"