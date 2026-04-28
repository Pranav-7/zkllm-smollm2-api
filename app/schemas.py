"""Pydantic models for API request/response shapes."""
from __future__ import annotations
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from .config import MAX_TOKENS_HARD_CAP, DEFAULT_MAX_TOKENS


JobStatus = Literal["pending", "running", "done", "failed"]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=MAX_TOKENS_HARD_CAP)
    # temperature kept for parity with OpenAI-style APIs but we force greedy
    # decoding because the proof is tied to a deterministic forward pass.
    deterministic: bool = True


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    estimated_seconds: int


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: dict  # {"tokens_done": int, "tokens_total": int, "layer": int, "phase": str}
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    error: Optional[str] = None


class TokenProof(BaseModel):
    token_index: int
    token_id: int
    token_text: str
    proof_dir: str                       # path relative to WORKDIR
    commitment_files: List[str]
    prove_seconds: float


class ResultResponse(BaseModel):
    job_id: str
    prompt: str
    generated_text: str
    tokens: List[TokenProof]
    total_prove_seconds: float
    tee_attestation: Optional[Dict[str, Any]] = None


class VerifyRequest(BaseModel):
    job_id: str


class VerifyEnqueueResponse(BaseModel):
    """Returned by POST /verify (async). Poll /verify/status/{verify_job_id}."""
    verify_job_id: str
    source_job_id: str
    status: JobStatus
    message: str
    estimated_seconds: int


class LayerVerification(BaseModel):
    layer: int
    weight_name: str
    expected_commitment_sha256: str
    recomputed_commitment_sha256: str
    match: bool


class VerifyResponse(BaseModel):
    job_id: str
    verified: bool
    total_layers_checked: int
    mismatches: int
    details: List[LayerVerification]
    verify_seconds: float
    note: str
    tee_attestation: Optional[Dict[str, Any]] = None


class CompositeProofResponse(BaseModel):
    """Combined zk + TEE proof for a single inference job."""
    job_id: str
    prompt: str
    generated_text: str
    zk_proof: Dict[str, Any]                            # {tokens, total_prove_seconds}
    zk_verification: Optional[Dict[str, Any]] = None    # null until /verify completes
    tee_proof: Optional[Dict[str, Any]] = None          # cleaned tee_attestation block
    tee_verification: Dict[str, Any]
    composite_verified: Optional[bool] = None           # null until zk verification done