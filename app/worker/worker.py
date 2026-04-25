"""
Single-worker thread. One GPU, one job at a time, strict FIFO.

We use a queue.Queue so FastAPI request handlers can enqueue without
blocking. The worker thread owns the GPU for the lifetime of each job.

Two job kinds share the same queue:
  - "generate": prove forward passes for N tokens
  - "verify":   re-check commitments of a completed generate job
"""
from __future__ import annotations
import queue
import threading
import time
import traceback

from ..storage.jobs import store
from . import pipeline


_q: "queue.Queue[str]" = queue.Queue()
_worker_thread: threading.Thread | None = None
_started = False
_start_lock = threading.Lock()


def enqueue(job_id: str) -> None:
    _q.put(job_id)


def _run_verify_job(job_id: str, job: dict) -> None:
    """Run a verify job: re-check commitments of the source generate job."""
    # Lazy import to avoid loading verifier at module import time
    from . import verifier
    from ..schemas import VerifyResponse

    source_id = job.get("source_job_id")
    if not source_id:
        raise RuntimeError("verify job missing source_job_id")

    source = store.get(source_id)
    if source is None:
        raise RuntimeError(f"source job {source_id} not found")
    if source.get("status") != "done":
        raise RuntimeError(
            f"source job {source_id} status is {source.get('status')}, not 'done'"
        )

    store.update(job_id, started_at=time.time())
    store.update_progress(job_id, phase="verifying")

    def _on_progress(**kw):
        store.update_progress(job_id, **kw)

    verified, total, mismatches, details, elapsed, note = (
        verifier.verify_job_commitments(source, on_progress=_on_progress)
    )

    # Build response payload that matches VerifyResponse schema
    result = VerifyResponse(
        job_id=source_id,                 # the source generate job
        verified=verified,
        total_layers_checked=total,
        mismatches=mismatches,
        details=details,
        verify_seconds=elapsed,
        note=note,
    ).model_dump()
    store.set_result(job_id, result)


def _loop():
    while True:
        job_id = _q.get()
        job = store.get(job_id)
        if job is None:
            _q.task_done()
            continue
        try:
            kind = job.get("kind", "generate")
            if kind == "generate":
                pipeline.run_job(job_id, job["prompt"], job["max_tokens"])
            elif kind == "verify":
                _run_verify_job(job_id, job)
            else:
                raise RuntimeError(f"unknown job kind: {kind!r}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] job {job_id} failed: {e}\n{tb}", flush=True)
            store.set_failed(job_id, f"{type(e).__name__}: {e}")
        finally:
            _q.task_done()


def start_worker() -> None:
    global _worker_thread, _started
    with _start_lock:
        if _started:
            return
        _worker_thread = threading.Thread(target=_loop, daemon=True, name="zkllm-worker")
        _worker_thread.start()
        _started = True
        print("[worker] started", flush=True)