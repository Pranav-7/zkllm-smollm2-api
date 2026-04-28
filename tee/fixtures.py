"""Load TEE attestation report fixtures and extract the cryptographic-provenance subset.

Three full attestation report JSONs live in app/tee/fixtures/. At runtime we:
  1. Pick one fixture deterministically by job_id (so /generate and /verify
     always show the same attestation for the same inference).
  2. Extract only hardware/cryptographic-provenance fields. Wrapper metadata
     is dropped at extraction time.

Robust to missing fixture files: API stays healthy, tee_attestation is omitted.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILENAMES = ["proof_1.json", "proof_2.json", "proof_3.json"]

_RAW_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_raw_fixtures() -> List[Dict[str, Any]]:
    """Load all fixture files. Missing/malformed files are skipped with a warning."""
    global _RAW_CACHE
    if _RAW_CACHE is not None:
        return _RAW_CACHE

    loaded: List[Dict[str, Any]] = []
    for name in FIXTURE_FILENAMES:
        path = FIXTURES_DIR / name
        if not path.exists():
            logger.warning("tee fixture missing: %s", path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("tee fixture unreadable %s: %s", path, e)
            continue

    _RAW_CACHE = loaded
    if not loaded:
        logger.warning("no tee fixtures loaded; tee_attestation will be omitted")
    else:
        logger.info("loaded %d tee fixture(s) from %s", len(loaded), FIXTURES_DIR)
    return loaded


def _pick_index_for_job(job_id: str, n: int) -> int:
    """Deterministic 0..n-1 from job_id. Same job_id always picks same fixture."""
    return hashlib.sha256(job_id.encode("utf-8")).digest()[0] % n


def _extract_tcb_info(tcb_in: Dict[str, Any]) -> Dict[str, Any]:
    """Pull tcb_info subset; drop the verbose event_log array."""
    keep = (
        "mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3",
        "mr_aggregated", "os_image_hash", "compose_hash",
        "device_id", "app_compose",
    )
    return {k: tcb_in[k] for k in keep if k in tcb_in}


def _extract_info(info_in: Dict[str, Any]) -> Dict[str, Any]:
    """Pull info subset; tcb_info has its own filter for event_log removal."""
    out: Dict[str, Any] = {}
    for k in (
        "app_id", "instance_id", "app_cert", "app_name",
        "device_id", "mr_aggregated", "os_image_hash", "compose_hash",
        "key_provider_info", "vm_config",
    ):
        if k in info_in:
            out[k] = info_in[k]
    if isinstance(info_in.get("tcb_info"), dict):
        out["tcb_info"] = _extract_tcb_info(info_in["tcb_info"])
    return out


def _extract_attestation_report(report_in: Dict[str, Any]) -> Dict[str, Any]:
    """Pull attestation_report subset; drops event_log, all_attestations, duplicate quote."""
    out: Dict[str, Any] = {}
    for k in ("signing_address", "signing_algo", "request_nonce",
              "intel_quote", "nvidia_payload", "vm_config"):
        if k in report_in:
            out[k] = report_in[k]
    if isinstance(report_in.get("info"), dict):
        out["info"] = _extract_info(report_in["info"])
    return out


def _extract_keep_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Build the cleaned tee_attestation block from a full raw report."""
    out: Dict[str, Any] = {}
    for k in ("signing_address", "attestation_nonce",
              "intel_tdx_present", "nvidia_gpu_present"):
        if k in raw:
            out[k] = raw[k]
    if isinstance(raw.get("signature"), dict):
        out["signature"] = raw["signature"]
    if isinstance(raw.get("attestation_report"), dict):
        out["attestation_report"] = _extract_attestation_report(raw["attestation_report"])
    return out


def get_tee_block_for_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Cleaned tee_attestation block for a given job_id, or None if no fixtures available."""
    fixtures = _load_raw_fixtures()
    if not fixtures:
        return None
    idx = _pick_index_for_job(job_id, len(fixtures))
    try:
        return _extract_keep_fields(fixtures[idx])
    except Exception as e:
        logger.exception("failed to extract tee block from fixture %d: %s", idx, e)
        return None


def build_tee_verification(tee_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Small verification summary derived from a tee block."""
    if not tee_block:
        return {
            "intel_tdx_present": False,
            "nvidia_gpu_present": False,
            "signature_present": False,
            "verified": False,
        }
    sig = tee_block.get("signature") or {}
    intel = bool(tee_block.get("intel_tdx_present"))
    nvidia = bool(tee_block.get("nvidia_gpu_present"))
    sig_present = bool(sig.get("signature")) and bool(sig.get("signing_address"))
    return {
        "intel_tdx_present": intel,
        "nvidia_gpu_present": nvidia,
        "signature_present": sig_present,
        "verified": intel and nvidia and sig_present,
    }