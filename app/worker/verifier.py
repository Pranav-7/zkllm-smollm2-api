"""
Commitment-check verifier.

Honest bounds: this does NOT run the full zkLLM SNARK verification
(the upstream zkllm-ccs2024 repo couples prover+verifier in the same
binary, not separable). Instead we verify:

  1. The SHA-256 of each committed .bin file saved during /generate.
  2. We re-run ./commit-param on the same int-quantized weights and
     byte-compare the recomputed commitment against the stored one.

If step 2 matches, the weights on disk have not been tampered with and
the Pedersen commitment (BLS12-381) binds them to the same group element.
This is a real cryptographic check on the commitment layer of zkLLM.
"""
from __future__ import annotations
import hashlib
import subprocess
import time
from pathlib import Path
from typing import List, Dict

from ..config import ZKLLM_ROOT, WORKDIR
from ..schemas import LayerVerification


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_job_commitments(job_meta: dict) -> tuple[bool, int, int, List[LayerVerification], float, str]:
    """
    job_meta['result']['tokens'][i]['commitment_files'] must list commitment
    file paths (relative to WORKDIR) saved at generation time, along with
    the int .bin files used to produce them.
    """
    t0 = time.time()
    details: List[LayerVerification] = []
    mismatches = 0
    commit_bin = str(ZKLLM_ROOT / "commit-param")

    result = job_meta.get("result") or {}
    tokens = result.get("tokens", [])

    # We verify the commitments produced during the first token's run;
    # weights are identical across tokens (same model), so checking them
    # once is enough. If you later support swapping models mid-job, iterate
    # per-token here.
    if not tokens:
        return False, 0, 0, [], time.time() - t0, "no tokens in job result"

    first = tokens[0]
    commit_files: List[str] = first.get("commitment_files", [])

    # Each commitment file is named "{prefix}-{weight_name}-commitment.bin"
    # Its matching int-bin is "{prefix}-{weight_name}-int.bin"
    # Its pp file is "{weight_name}-pp.bin"
    for cf_rel in commit_files:
        cf = WORKDIR / Path(cf_rel).name
        if not cf.exists():
            details.append(LayerVerification(
                layer=-1, weight_name=cf.name,
                expected_commitment_sha256="<missing>",
                recomputed_commitment_sha256="<missing>",
                match=False))
            mismatches += 1
            continue

        expected = _sha256(cf)

        # Derive matching int.bin and pp.bin paths
        stem = cf.name.replace("-commitment.bin", "")   # "layer-0-model.layers.0.self_attn.q_proj.weight"
        # Split "layer-{N}-" prefix off to get weight name
        # prefix pattern: "layer-{int}-"
        try:
            _, layer_str, rest = stem.split("-", 2)
            layer_idx = int(layer_str)
            weight_name = rest
        except ValueError:
            details.append(LayerVerification(
                layer=-1, weight_name=cf.name,
                expected_commitment_sha256=expected,
                recomputed_commitment_sha256="<unparseable-name>",
                match=False))
            mismatches += 1
            continue

        int_path    = WORKDIR / f"layer-{layer_idx}-{weight_name}-int.bin"
        pp_path     = WORKDIR / f"{weight_name}-pp.bin"
        recomp_path = WORKDIR / f"verify-{cf.name}"

        if not int_path.exists() or not pp_path.exists():
            details.append(LayerVerification(
                layer=layer_idx, weight_name=weight_name,
                expected_commitment_sha256=expected,
                recomputed_commitment_sha256="<source-files-missing>",
                match=False))
            mismatches += 1
            continue

        # Re-run commit-param. We need the (M, N) shape; read int file size
        # and derive — int32, so count = filesize / 4. For 1-D weights N=1.
        n_elements = int_path.stat().st_size // 4
        # We don't store shape in a sidecar. Heuristic: try ndim-based inference.
        # All 2D padded weights in this model have known shapes from config:
        # (HIDDEN, HIDDEN), (INTER, HIDDEN), (HIDDEN, INTER), or (HIDDEN,) for 1D.
        # Rather than guess, we shell out using the SAME logic as commit_layer_weights
        # by reading WORKDIR for a shape sidecar produced at commit time.
        shape_path = WORKDIR / f"layer-{layer_idx}-{weight_name}-shape.txt"
        if not shape_path.exists():
            # Fall back to element-count only heuristic - mark as weak check
            details.append(LayerVerification(
                layer=layer_idx, weight_name=weight_name,
                expected_commitment_sha256=expected,
                recomputed_commitment_sha256="<no-shape-sidecar>",
                match=False))
            mismatches += 1
            continue

        with open(shape_path) as f:
            shape_line = f.read().strip()
            parts = shape_line.split()
            if len(parts) == 2:
                M, N = int(parts[0]), int(parts[1])
            else:
                M, N = int(parts[0]), 1

        r = subprocess.run(
            f"{commit_bin} {pp_path} {int_path} {recomp_path} {M} {N}",
            shell=True, capture_output=True, text=True, cwd=str(ZKLLM_ROOT),
        )
        if r.returncode != 0 or not recomp_path.exists():
            details.append(LayerVerification(
                layer=layer_idx, weight_name=weight_name,
                expected_commitment_sha256=expected,
                recomputed_commitment_sha256=f"<commit-param-failed: {r.stderr[:100]}>",
                match=False))
            mismatches += 1
            continue

        recomputed = _sha256(recomp_path)
        recomp_path.unlink(missing_ok=True)

        match = (recomputed == expected)
        details.append(LayerVerification(
            layer=layer_idx, weight_name=weight_name,
            expected_commitment_sha256=expected,
            recomputed_commitment_sha256=recomputed,
            match=match))
        if not match:
            mismatches += 1

    verified = (mismatches == 0 and len(details) > 0)
    note = ("Commitment re-check only; full SNARK verification not exposed "
            "by upstream zkllm-ccs2024 (prover+verifier fused in prover binary).")
    return verified, len(details), mismatches, details, time.time() - t0, note
