"""FastAPI entrypoint."""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import API_TITLE, API_VERSION
from .routes import generate as generate_routes
from .routes import verify as verify_routes
from .worker import worker as worker_mod


app = FastAPI(title=API_TITLE, version=API_VERSION)
app.include_router(generate_routes.router)
app.include_router(verify_routes.router)


@app.on_event("startup")
def _startup():
    worker_mod.start_worker()


@app.get("/")
def root():
    return {"service": API_TITLE, "version": API_VERSION,
            "docs": "/docs",
            "endpoints": [
                "POST /generate",
                "GET  /status/{job_id}",
                "GET  /result/{job_id}",
                "POST /verify",
                "GET  /verify/status/{verify_job_id}",
                "GET  /verify/result/{verify_job_id}",
            ]}


@app.get("/health")
def health():
    import torch
    return JSONResponse({
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    })