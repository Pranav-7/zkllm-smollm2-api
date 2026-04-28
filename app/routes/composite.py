"""Composite proof endpoint: zk proof + TEE attestation + verification result."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..schemas import CompositeProofResponse
from ..storage.jobs import store
from ..tee import fixtures as tee_fixtures


router = APIRouter(tags=["composite"])


def _build_composite(source_id: str) -> dict:
    """Assemble the composite proof from the source generate job + any matching verify job."""
    src = store.get(source_id)
    if not src:
        raise HTTPException(404, f"job {source_id} not found")
    if src.get("kind") != "generate":
        raise HTTPException(
            400, f"job {source_id} is a {src.get('kind')!r} job, not a generate job"
        )
    if src["status"] != "done":
        raise HTTPException(409, f"generate job status is {src['status']}, not done")

    src_result = src.get("result") or {}

    # zk proof block (always present once /generate is done)
    zk_proof = {
        "tokens": src_result.get("tokens", []),
        "total_prove_seconds": src_result.get("total_prove_seconds", 0.0),
    }

    # zk verification (only if /verify has been run and finished)
    verify_job = store.find_done_verify_for_source(source_id)
    zk_verification = None
    if verify_job and verify_job.get("result"):
        vr = verify_job["result"]
        zk_verification = {
            "verified": vr.get("verified", False),
            "total_layers_checked": vr.get("total_layers_checked", 0),
            "mismatches": vr.get("mismatches", 0),
            "verify_seconds": vr.get("verify_seconds", 0.0),
        }

    # TEE proof — same fixture that was attached during /generate (deterministic by job_id).
    # Falls back to live extraction if the generate result predates fixture configuration.
    tee_proof = src_result.get("tee_attestation") or tee_fixtures.get_tee_block_for_job(source_id)
    tee_verification = tee_fixtures.build_tee_verification(tee_proof)

    composite_verified = None
    if zk_verification is not None:
        composite_verified = (
            bool(zk_verification.get("verified")) and bool(tee_verification.get("verified"))
        )

    return {
        "job_id": source_id,
        "prompt": src_result.get("prompt", src.get("prompt", "")),
        "generated_text": src_result.get("generated_text", ""),
        "zk_proof": zk_proof,
        "zk_verification": zk_verification,
        "tee_proof": tee_proof,
        "tee_verification": tee_verification,
        "composite_verified": composite_verified,
    }


@router.get("/composite/{job_id}", response_model=CompositeProofResponse)
def composite(job_id: str):
    """Return the composite proof (zk + TEE) for a completed generate job.

    If /verify has been run for this job_id, the composite includes the
    zk_verification block and composite_verified flag. Otherwise both are null.
    """
    return _build_composite(job_id)


@router.get("/composite/{job_id}/download")
def composite_download(job_id: str):
    """Same payload as /composite/{job_id}, served as a downloadable file."""
    payload = _build_composite(job_id)
    headers = {
        "Content-Disposition": f'attachment; filename="composite_proof_{job_id}.json"'
    }
    return JSONResponse(content=payload, headers=headers)