"""Thread-safe in-memory job registry. Jobs are lost on restart (by design).

Supports two job kinds:
  - "generate": prove forward passes for N tokens, produces commitment files.
  - "verify":   re-check commitments of a previously-completed generate job.
Both share the same FIFO worker queue.
"""
import threading
import time
import uuid
from typing import Optional, Dict, Any


class JobStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, prompt: str, max_tokens: int) -> str:
        """Create a generate job (kind='generate')."""
        job_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": "generate",
                "status": "pending",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "progress": {"tokens_done": 0, "tokens_total": max_tokens,
                             "layer": 0, "phase": "queued"},
                "started_at": None,
                "updated_at": time.time(),
                "error": None,
                "result": None,
            }
        return job_id

    def create_verify(self, source_job_id: str) -> str:
        """Create a verify job that will re-check commitments of source_job_id."""
        job_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": "verify",
                "status": "pending",
                "source_job_id": source_job_id,
                "progress": {"weights_checked": 0, "weights_total": 0,
                             "phase": "queued"},
                "started_at": None,
                "updated_at": time.time(),
                "error": None,
                "result": None,
            }
        return job_id

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(fields)
            self._jobs[job_id]["updated_at"] = time.time()

    def update_progress(self, job_id: str, **progress_fields) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]["progress"].update(progress_fields)
            self._jobs[job_id]["updated_at"] = time.time()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def set_result(self, job_id: str, result: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["result"] = result
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["updated_at"] = time.time()

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["updated_at"] = time.time()

    def find_done_verify_for_source(self, source_job_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recently-updated 'done' verify job for source_job_id, or None."""
        with self._lock:
            candidates = [
                j for j in self._jobs.values()
                if j.get("kind") == "verify"
                and j.get("source_job_id") == source_job_id
                and j.get("status") == "done"
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda j: j.get("updated_at", 0) or 0, reverse=True)
            return dict(candidates[0])


# Module-level singleton
store = JobStore()