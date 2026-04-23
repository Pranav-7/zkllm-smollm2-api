"""
Per-token proving pipeline.

For each of N=max_tokens generation steps:
  1. Use HF model (greedy) to pick the next token.
  2. Embed the current prompt, pad to (SEQ_LEN, HIDDEN), write to disk.
  3. For each of 32 layers:
       - commit layer weights
       - prove_one_layer (chains outputs into next layer's input)
  4. Record commitment paths + timing into the job.

Note: This proves that a forward pass was performed on the given prompt
through the committed weights. Because of the RoPE/GAP-1 approximation
in zkllm's CUDA binary, the *proven* logits may diverge slightly from
HF's reference logits — but the CUDA-side math is consistent with the
commitments. We return HF's greedy token as the user-facing output.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..config import (
    WORKDIR, SEQ_LEN, HIDDEN, HIDDEN_TRUE, SCALE, N_LAYERS_PROVE,
)
from ..storage.jobs import store
from . import prover


def _embed_prompt_to_file(input_ids: torch.Tensor, out_path: Path) -> None:
    """
    Take token ids, embed via model.embed_tokens, pad hidden 960→1024,
    pad sequence to SEQ_LEN, quantize to int32*SCALE, save.
    """
    model = prover.get_model()
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids.to(model.device))   # (1, L, 960)
    embeds = embeds[0].float().cpu()                                    # (L, 960)
    L = embeds.shape[0]
    padded = torch.zeros(SEQ_LEN, HIDDEN, dtype=torch.float32)
    L_use = min(L, SEQ_LEN)
    padded[:L_use, :HIDDEN_TRUE] = embeds[:L_use]
    q = torch.round(padded * SCALE).to(torch.int32)
    q.numpy().astype(np.int32).tofile(out_path)


def run_job(
    job_id: str,
    prompt: str,
    max_tokens: int,
    on_update: Callable[[dict], None] = None,
) -> dict:
    """Executes the full prove-per-token pipeline. Raises on failure."""
    tokenizer = prover.get_tokenizer()
    model = prover.get_model()

    prover.ensure_public_params()

    store.update(job_id, status="running", started_at=time.time())
    store.update_progress(job_id, phase="tokenizing")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated_ids = input_ids.clone()
    total_prove_s = 0.0
    token_proofs = []

    for tok_idx in range(max_tokens):
        store.update_progress(job_id, tokens_done=tok_idx, phase="hf_decode")

        # --- HF greedy next token ---
        with torch.no_grad():
            logits = model(generated_ids).logits[:, -1, :]
            next_id = int(logits.argmax(dim=-1).item())
        next_tensor = torch.tensor([[next_id]], device=model.device)
        generated_ids = torch.cat([generated_ids, next_tensor], dim=1)
        token_text = tokenizer.decode([next_id], skip_special_tokens=True)

        # --- prove forward pass for the current (pre-append) prompt ---
        token_dir = WORKDIR / f"job-{job_id}-tok-{tok_idx}"
        token_dir.mkdir(exist_ok=True)
        store.update_progress(job_id, phase=f"embed_tok{tok_idx}")
        input_bin = WORKDIR / f"job-{job_id}-tok-{tok_idx}-input.bin"
        _embed_prompt_to_file(generated_ids[:, :-1], input_bin)

        per_layer_input = input_bin
        commit_files = []
        tok_prove_s = 0.0

        for li in range(N_LAYERS_PROVE):
            store.update_progress(
                job_id, phase=f"tok{tok_idx}_layer{li}_commit", layer=li,
            )
            weights = prover.build_padded_weights_for_layer(li)
            _, _ = prover.commit_layer_weights(weights, prefix=f"layer-{li}")
            commit_files.extend([
                f"layer-{li}-{name}-commitment.bin" for name in weights.keys()
            ])

            per_layer_output = WORKDIR / f"job-{job_id}-tok-{tok_idx}-layer{li}.bin"

            def phase_cb(p: str, _li=li, _ti=tok_idx):
                store.update_progress(job_id, phase=f"tok{_ti}_layer{_li}_{p}")

            layer_s = prover.prove_one_layer(li, per_layer_input, per_layer_output, on_phase=phase_cb)
            tok_prove_s += layer_s
            per_layer_input = per_layer_output

        total_prove_s += tok_prove_s
        token_proofs.append({
            "token_index": tok_idx,
            "token_id": next_id,
            "token_text": token_text,
            "proof_dir": str(token_dir.relative_to(WORKDIR)),
            "commitment_files": commit_files,
            "prove_seconds": tok_prove_s,
        })

    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )

    result = {
        "job_id": job_id,
        "prompt": prompt,
        "generated_text": generated_text,
        "tokens": token_proofs,
        "total_prove_seconds": total_prove_s,
    }
    store.set_result(job_id, result)
    return result
