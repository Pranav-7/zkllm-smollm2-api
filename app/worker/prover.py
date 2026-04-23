"""
zkLLM prover pipeline for SmolLM2-360M.

Direct port of the notebook cells 6-13 into callable functions:
  - build_padded_weights_for_layer (cell 11)
  - commit_layer_weights           (cell 11)
  - prove_one_layer                (cell 11)

We keep the model + tokenizer loaded once at module import (expensive).
Public params (ppgen) are generated on-demand and cached per weight name.
"""
from __future__ import annotations
import os
import sys
import math
import time
import subprocess
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import (
    MODEL_CARD, ZKLLM_ROOT, WORKDIR,
    HIDDEN_TRUE, N_HEADS_TRUE, KV_HEADS_TRUE, HEAD_DIM, N_GROUPS,
    HIDDEN, N_HEADS, INTER, SEQ_LEN, LOG_SF, SCALE, LOG_OFF,
)

# Make zkllm's fileio_utils importable (it's a loose .py in the repo root)
sys.path.insert(0, str(ZKLLM_ROOT))

# =============================================================================
# Module-level: load model once
# =============================================================================
print(f"[prover] Loading {MODEL_CARD} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
_model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, torch_dtype=torch.float32)
_model.eval()
_cfg = _model.config
assert _cfg.hidden_size == HIDDEN_TRUE, f"Expected {HIDDEN_TRUE}, got {_cfg.hidden_size}"
_ROPE_THETA = getattr(_cfg, "rope_theta", 10000.0)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_DEVICE)
print(f"[prover] Model on {_DEVICE}. rope_theta={_ROPE_THETA}", flush=True)

# fileio_utils is loaded only after sys.path tweak
import fileio_utils  # noqa: E402

# =============================================================================
# Public accessors
# =============================================================================

def get_tokenizer():
    return _tokenizer

def get_model():
    return _model

def get_cfg():
    return _cfg


# =============================================================================
# Weight padding (cell 8 of the notebook)
# =============================================================================

def _pad_2d(W: torch.Tensor, out_rows: int, out_cols: int) -> torch.Tensor:
    r, c = W.shape
    P = torch.zeros(out_rows, out_cols, dtype=W.dtype)
    P[:r, :c] = W
    return P


def _pad_1d(v: torch.Tensor, out_len: int) -> torch.Tensor:
    P = torch.zeros(out_len, dtype=v.dtype)
    P[: v.shape[0]] = v
    return P


def _expand_and_pad_kv(weight: torch.Tensor) -> torch.Tensor:
    """GQA: (KV*HEAD_DIM, HIDDEN_TRUE) → (N_HEADS*HEAD_DIM, HIDDEN) via repeat+pad."""
    w = weight.view(KV_HEADS_TRUE, HEAD_DIM, HIDDEN_TRUE)
    w = w.repeat_interleave(N_GROUPS, dim=0)
    w = w.reshape(N_HEADS_TRUE * HEAD_DIM, HIDDEN_TRUE)
    return _pad_2d(w, N_HEADS * HEAD_DIM, HIDDEN)


def _pad_q_or_o(weight: torch.Tensor) -> torch.Tensor:
    return _pad_2d(weight, HIDDEN, HIDDEN)


def build_padded_weights_for_layer(layer_idx: int) -> dict:
    layer = _model.model.layers[layer_idx]
    w = {}
    for name, p in layer.named_parameters():
        p = p.detach().float().cpu()
        if "k_proj.weight" in name or "v_proj.weight" in name:
            w[name] = _expand_and_pad_kv(p)
        elif "q_proj.weight" in name or "o_proj.weight" in name:
            w[name] = _pad_q_or_o(p)
        elif "gate_proj.weight" in name or "up_proj.weight" in name:
            w[name] = _pad_2d(p, INTER, HIDDEN)
        elif "down_proj.weight" in name:
            w[name] = _pad_2d(p, HIDDEN, INTER)
        elif "layernorm.weight" in name:
            w[name] = _pad_1d(p, HIDDEN)
    return w


# =============================================================================
# Public parameters (ppgen) — cached across calls
# =============================================================================

_pp_ready = False

def ensure_public_params() -> None:
    """One-time ppgen for all weight names. Idempotent."""
    global _pp_ready
    if _pp_ready:
        return
    print("[prover] Ensuring public parameters ...", flush=True)
    ref = build_padded_weights_for_layer(0)
    ppgen = str(ZKLLM_ROOT / "ppgen")
    for name, w in ref.items():
        pp_path = WORKDIR / f"{name}-pp.bin"
        if pp_path.exists():
            continue
        pp_size = (w.shape[0] << LOG_OFF) if w.ndim == 2 else w.shape[0]
        r = subprocess.run(
            f"{ppgen} {pp_size} {pp_path}",
            shell=True, capture_output=True, text=True, cwd=str(ZKLLM_ROOT),
        )
        if r.returncode != 0:
            raise RuntimeError(f"ppgen failed for {name}: {r.stderr}")
    _pp_ready = True
    print("[prover] Public params ready.", flush=True)


# =============================================================================
# Commit (cell 11: commit_layer_weights)
# =============================================================================

def commit_layer_weights(weights: dict, prefix: str) -> tuple[float, int]:
    t0 = time.time()
    commit_bin = str(ZKLLM_ROOT / "commit-param")
    total_bytes = 0
    for name, tensor in weights.items():
        w_orig = tensor.float().T if tensor.ndim == 2 else tensor.float()
        w_int = torch.round(w_orig * SCALE).to(torch.int32)
        pp_path     = WORKDIR / f"{name}-pp.bin"
        int_path    = WORKDIR / f"{prefix}-{name}-int.bin"
        commit_path = WORKDIR / f"{prefix}-{name}-commitment.bin"
        w_int.cpu().numpy().astype(np.int32).tofile(int_path)
        shape_path = WORKDIR / f"{prefix}-{name}-shape.txt"
        if w_int.ndim == 2:
            M, N = w_int.shape
            shape_path.write_text(f"{M} {N}\n")
            cmd = f"{commit_bin} {pp_path} {int_path} {commit_path} {M} {N}"
        else:
            shape_path.write_text(f"{w_int.shape[0]}\n")
            cmd = f"{commit_bin} {pp_path} {int_path} {commit_path} {w_int.shape[0]} 1"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=str(ZKLLM_ROOT))
        if r.returncode != 0:
            raise RuntimeError(f"commit-param failed for {name}: {r.stderr}")
        if commit_path.exists():
            total_bytes += commit_path.stat().st_size
    return time.time() - t0, total_bytes


# =============================================================================
# Prove one layer (cell 11: prove_one_layer)
# =============================================================================

def _run(cmd: str, tag: str, cwd: str) -> None:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        # SIGSEGV = 139, SIGABRT = 134, etc. Add the signal name for diagnosis.
        import signal as _sig
        sig_name = ""
        if r.returncode > 128:
            try:
                sig_name = f" (signal {_sig.Signals(r.returncode - 128).name})"
            except (ValueError, KeyError):
                sig_name = f" (signal {r.returncode - 128})"
        raise RuntimeError(
            f"[{tag}] FAILED (rc={r.returncode}{sig_name}): {cmd}\n"
            f"CWD: {cwd}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )


def prove_one_layer(
    layer_idx: int,
    input_file: Path,
    output_file: Path,
    on_phase: Optional[Callable[[str], None]] = None,
) -> float:
    """
    ZK proof pipeline for one decoder layer.
    Follows the documented gaps (GAP-1 RoPE, GAP-2 GQA expansion, GAP-3 padding).
    """
    VALUE_LOGSF, ACCU_LOGSF = 16, 20
    prefix   = f"layer-{layer_idx}"
    rn1_out  = WORKDIR / f"{prefix}-rn1-out.bin"
    attn_out = WORKDIR / f"{prefix}-attn-out.bin"
    skip1    = WORKDIR / f"{prefix}-skip1.bin"
    rn2_out  = WORKDIR / f"{prefix}-rn2-out.bin"
    ffn_out  = WORKDIR / f"{prefix}-ffn-out.bin"

    cwd = str(ZKLLM_ROOT)
    bin_rmsnorm = str(ZKLLM_ROOT / "rmsnorm")
    bin_selfattn = str(ZKLLM_ROOT / "self-attn")
    bin_ffn = str(ZKLLM_ROOT / "ffn")
    bin_skip = str(ZKLLM_ROOT / "skip-connection")

    t0 = time.time()

    # STEP 1: input RMSNorm
    if on_phase: on_phase("rmsnorm1")
    X = (torch.tensor(np.fromfile(input_file, dtype=np.int32).reshape(SEQ_LEN, HIDDEN),
                      device="cuda", dtype=torch.float64) / (1 << LOG_SF))
    rms_inv = 1.0 / torch.sqrt((X ** 2).mean(dim=-1) + _cfg.rms_norm_eps)
    # The rmsnorm CUDA binary reads "rms_inv_temp.bin" from its own CWD.
    # We launch it with cwd=ZKLLM_ROOT, so write the temp file there (not CWD of python process).
    rms_inv_path = Path(cwd) / "rms_inv_temp.bin"
    fileio_utils.save_int(rms_inv, 1 << 16, str(rms_inv_path))
    _run(f"{bin_rmsnorm} input {input_file} {SEQ_LEN} {HIDDEN} {WORKDIR} {prefix} {rn1_out}",
         "rms1", cwd)
    rms_inv_path.unlink(missing_ok=True)

    # STEP 2a: QKV linear
    if on_phase: on_phase("attn_linear")
    _run(f"{bin_selfattn} linear {rn1_out} {SEQ_LEN} {HIDDEN} {WORKDIR} {prefix} {attn_out} {N_HEADS}",
         "attn-linear", cwd)

    # STEP 2b: python-side RoPE reference (bookkeeping only, GAP-1)
    if on_phase: on_phase("attn_rope_ref")
    Q_int = np.fromfile(Path(cwd) / "temp_Q.bin", dtype=np.int32).reshape(SEQ_LEN, HIDDEN)
    K_int = np.fromfile(Path(cwd) / "temp_K.bin", dtype=np.int32).reshape(SEQ_LEN, HIDDEN)
    V_int = np.fromfile(Path(cwd) / "temp_V.bin", dtype=np.int32).reshape(SEQ_LEN, HIDDEN)
    Q = fileio_utils.to_float(torch.tensor(Q_int, device="cuda"), VALUE_LOGSF)
    K = fileio_utils.to_float(torch.tensor(K_int, device="cuda"), VALUE_LOGSF)
    V = fileio_utils.to_float(torch.tensor(V_int, device="cuda"), VALUE_LOGSF)
    Q = Q.view(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(0, 1).contiguous()
    K = K.view(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(0, 1).contiguous()
    V = V.view(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(0, 1).contiguous()

    inv_freq = 1.0 / (_ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device="cuda") / HEAD_DIM))
    freqs = torch.outer(torch.arange(SEQ_LEN, dtype=torch.float64, device="cuda"), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(Q.dtype), emb.sin().to(Q.dtype)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    Q = Q * cos + rotate_half(Q) * sin
    K = K * cos + rotate_half(K) * sin

    A_ = Q @ K.transpose(-2, -1)
    A = fileio_utils.to_int64(A_, VALUE_LOGSF)
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device="cuda"), diagonal=1)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(HEAD_DIM) * torch.log(
        (torch.exp(fileio_utils.to_float(A, ACCU_LOGSF) / math.sqrt(HEAD_DIM))
         * ~mask).sum(dim=-1, keepdim=True).clamp(min=1e-9))
    A -= fileio_utils.to_int64(shift, ACCU_LOGSF)
    attn_w = (torch.exp(fileio_utils.to_float(A, ACCU_LOGSF, torch.float64)
                        / math.sqrt(HEAD_DIM)) * ~mask)
    attn_pre = fileio_utils.fromto_int64(attn_w @ V.to(attn_w.dtype), VALUE_LOGSF)
    attn_pre = attn_pre.transpose(0, 1).contiguous().view(SEQ_LEN, HIDDEN)
    fileio_utils.save_int(attn_pre, 1 << VALUE_LOGSF, "temp_attn_out.bin")

    # CUDA attn proof (no RoPE — GAP-1)
    if on_phase: on_phase("attn_proof")
    _run(f"{bin_selfattn} attn {rn1_out} {SEQ_LEN} {HIDDEN} {WORKDIR} {prefix} {attn_out} {N_HEADS}",
         "attn-attn", cwd)
    os.system(f"rm -f {cwd}/temp*.bin")

    # STEP 3: skip #1
    if on_phase: on_phase("skip1")
    _run(f"{bin_skip} {input_file} {attn_out} {skip1}", "skip1", cwd)

    # STEP 4: post-attention RMSNorm
    if on_phase: on_phase("rmsnorm2")
    X2 = (torch.tensor(np.fromfile(skip1, dtype=np.int32).reshape(SEQ_LEN, HIDDEN),
                       device="cuda", dtype=torch.float64) / (1 << LOG_SF))
    rms_inv2 = 1.0 / torch.sqrt((X2 ** 2).mean(dim=-1) + _cfg.rms_norm_eps)
    rms_inv2_path = Path(cwd) / "rms_inv_temp.bin"
    fileio_utils.save_int(rms_inv2, 1 << 16, str(rms_inv2_path))
    _run(f"{bin_rmsnorm} post_attention {skip1} {SEQ_LEN} {HIDDEN} {WORKDIR} {prefix} {rn2_out}",
         "rms2", cwd)
    rms_inv2_path.unlink(missing_ok=True)

    # STEP 5: FFN
    if on_phase: on_phase("ffn")
    xs = torch.arange(-(1 << 7), (1 << 7), step=1.0 / (1 << 12))
    ys = xs * torch.sigmoid(xs)
    swiglu_path = Path(cwd) / "swiglu-table.bin"
    fileio_utils.save_int(ys, 1 << 16, str(swiglu_path))
    _run(f"{bin_ffn} {rn2_out} {SEQ_LEN} {HIDDEN} {INTER} {WORKDIR} {prefix} {ffn_out}",
         "ffn", cwd)
    if swiglu_path.exists(): swiglu_path.unlink()

    # STEP 6: skip #2
    if on_phase: on_phase("skip2")
    _run(f"{bin_skip} {skip1} {ffn_out} {output_file}", "skip2", cwd)

    return time.time() - t0