"""Thread-safe in-memory job registry. Jobs are lost on restart (by design)."""
import threading
import time
import uuid
from typing import Optional, Dict, Any


class JobStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, prompt: str, max_tokens: int) -> str:
        job_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
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


# Module-level singleton
store = JobStore()
