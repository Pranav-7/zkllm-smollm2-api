"""Generation endpoints."""
from fastapi import APIRouter, HTTPException

from ..schemas import (
    GenerateRequest, GenerateResponse, StatusResponse, ResultResponse,
)
from ..storage.jobs import store
from ..worker import worker as worker_mod
from ..config import N_LAYERS_PROVE


router = APIRouter(tags=["generate"])

# Rough estimate: assume ~30s/layer on H200 (conservative; tune after first run)
_SECS_PER_LAYER_EST = 30


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.deterministic:
        raise HTTPException(400, "deterministic=false is not supported; "
                                 "proofs require greedy decoding.")
    job_id = store.create(req.prompt, req.max_tokens)
    worker_mod.enqueue(job_id)
    est = req.max_tokens * N_LAYERS_PROVE * _SECS_PER_LAYER_EST
    return GenerateResponse(
        job_id=job_id,
        status="pending",
        message=f"Queued. Poll /status/{job_id}.",
        estimated_seconds=est,
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str):
    j = store.get(job_id)
    if not j:
        raise HTTPException(404, f"job {job_id} not found")
    return StatusResponse(
        job_id=j["job_id"],
        status=j["status"],
        progress=j["progress"],
        started_at=j["started_at"],
        updated_at=j["updated_at"],
        error=j["error"],
    )


@router.get("/result/{job_id}", response_model=ResultResponse)
def result(job_id: str):
    j = store.get(job_id)
    if not j:
        raise HTTPException(404, f"job {job_id} not found")
    if j["status"] != "done":
        raise HTTPException(409, f"job status is {j['status']}, not done")
    return ResultResponse(**j["result"])
