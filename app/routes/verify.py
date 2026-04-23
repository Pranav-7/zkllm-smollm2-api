"""Verification endpoint."""
from fastapi import APIRouter, HTTPException

from ..schemas import VerifyRequest, VerifyResponse
from ..storage.jobs import store
from ..worker.verifier import verify_job_commitments


router = APIRouter(tags=["verify"])


@router.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    j = store.get(req.job_id)
    if not j:
        raise HTTPException(404, f"job {req.job_id} not found")
    if j["status"] != "done":
        raise HTTPException(409, f"cannot verify: job status is {j['status']}")

    verified, total, mismatches, details, elapsed, note = verify_job_commitments(j)
    return VerifyResponse(
        job_id=req.job_id,
        verified=verified,
        total_layers_checked=total,
        mismatches=mismatches,
        details=details,
        verify_seconds=elapsed,
        note=note,
    )
