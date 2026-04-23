"""
Single-worker thread. One GPU, one job at a time, strict FIFO.

We use a queue.Queue so FastAPI request handlers can enqueue without
blocking. The worker thread owns the GPU for the lifetime of each job.
"""
from __future__ import annotations
import queue
import threading
import traceback

from ..storage.jobs import store
from . import pipeline


_q: "queue.Queue[str]" = queue.Queue()
_worker_thread: threading.Thread | None = None
_started = False
_start_lock = threading.Lock()


def enqueue(job_id: str) -> None:
    _q.put(job_id)


def _loop():
    while True:
        job_id = _q.get()
        job = store.get(job_id)
        if job is None:
            _q.task_done()
            continue
        try:
            pipeline.run_job(job_id, job["prompt"], job["max_tokens"])
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
