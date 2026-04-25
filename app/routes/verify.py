"""Verification endpoints (async).

Verification is GPU-bound and may take 5-10 minutes for a 32-layer model.
That exceeds the idle timeout of upstream proxies (e.g. Cloudflare ~100s
in front of RunPod), so we follow the same async pattern as /generate:

  POST /verify                  -> {verify_job_id, status: "pending", ...}
  GET  /verify/status/{vjid}    -> live progress (weights_checked / total)
  GET  /verify/result/{vjid}    -> final VerifyResponse when status == "done"
"""
from fastapi import APIRouter, HTTPException

from ..schemas import (
    VerifyRequest, VerifyResponse, VerifyEnqueueResponse, StatusResponse,
)
from ..storage.jobs import store
from ..worker import worker as worker_mod


router = APIRouter(tags=["verify"])

# Rough estimate: ~1.5s per commitment on H200, 288 commitments per generate
# job (32 layers × 9 weights). Adjust after measuring on real workloads.
_SECS_PER_COMMITMENT_EST = 2


@router.post("/verify", response_model=VerifyEnqueueResponse)
def verify_submit(req: VerifyRequest):
    """Enqueue a verify job for an existing completed generate job."""
    src = store.get(req.job_id)
    if not src:
        raise HTTPException(404, f"source job {req.job_id} not found")
    if src.get("kind", "generate") != "generate":
        raise HTTPException(
            400,
            f"source job {req.job_id} kind is {src.get('kind')!r}; "
            "only 'generate' jobs can be verified",
        )
    if src["status"] != "done":
        raise HTTPException(
            409, f"cannot verify: source job status is {src['status']} (need 'done')"
        )

    # Estimate verify time from the source job's commitment count
    n_commits = 288  # default
    try:
        tokens = (src.get("result") or {}).get("tokens") or []
        if tokens:
            n_commits = len(tokens[0].get("commitment_files", [])) or n_commits
    except Exception:
        pass

    verify_job_id = store.create_verify(source_job_id=req.job_id)
    worker_mod.enqueue(verify_job_id)

    return VerifyEnqueueResponse(
        verify_job_id=verify_job_id,
        source_job_id=req.job_id,
        status="pending",
        message=f"Queued. Poll /verify/status/{verify_job_id}.",
        estimated_seconds=n_commits * _SECS_PER_COMMITMENT_EST,
    )


@router.get("/verify/status/{verify_job_id}", response_model=StatusResponse)
def verify_status(verify_job_id: str):
    j = store.get(verify_job_id)
    if not j:
        raise HTTPException(404, f"verify job {verify_job_id} not found")
    if j.get("kind") != "verify":
        raise HTTPException(
            400,
            f"job {verify_job_id} is a {j.get('kind')!r} job, not a verify job; "
            f"use /status/{verify_job_id} instead",
        )
    return StatusResponse(
        job_id=j["job_id"],
        status=j["status"],
        progress=j["progress"],
        started_at=j["started_at"],
        updated_at=j["updated_at"],
        error=j["error"],
    )


@router.get("/verify/result/{verify_job_id}", response_model=VerifyResponse)
def verify_result(verify_job_id: str):
    j = store.get(verify_job_id)
    if not j:
        raise HTTPException(404, f"verify job {verify_job_id} not found")
    if j.get("kind") != "verify":
        raise HTTPException(
            400, f"job {verify_job_id} is not a verify job"
        )
    if j["status"] != "done":
        raise HTTPException(
            409, f"verify job status is {j['status']}, not done"
        )
    if not j.get("result"):
        raise HTTPException(500, "verify job marked done but result is empty")
    return VerifyResponse(**j["result"])