"""Central config. All magic numbers from the notebook live here."""
import os
from pathlib import Path

# --- model ---
MODEL_CARD = os.getenv("MODEL_CARD", "HuggingFaceTB/SmolLM2-360M")

# --- zkllm paths ---
ZKLLM_ROOT = Path(os.getenv("ZKLLM_ROOT", "/app/zkllm"))          # where binaries live
WORKDIR    = Path(os.getenv("ZKLLM_WORKDIR", "/data/workdir"))     # proof artifacts
JOB_DIR    = Path(os.getenv("JOB_DIR", "/data/jobs"))              # per-job state
WORKDIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)

# --- padded dims (from notebook Cell 7, fixed for SmolLM2-360M) ---
HIDDEN_TRUE   = 960
N_HEADS_TRUE  = 15
KV_HEADS_TRUE = 5
HEAD_DIM      = 64
INTER_TRUE    = 2560
N_GROUPS      = N_HEADS_TRUE // KV_HEADS_TRUE     # 3

HIDDEN   = 1024
N_HEADS  = 16
INTER    = 4096

SEQ_LEN  = 512
LOG_SF   = 16
SCALE    = 1 << LOG_SF
LOG_OFF  = 5

# --- generation limits ---
MAX_TOKENS_HARD_CAP = int(os.getenv("MAX_TOKENS_HARD_CAP", "10"))
DEFAULT_MAX_TOKENS  = int(os.getenv("DEFAULT_MAX_TOKENS", "1"))
N_LAYERS_PROVE      = int(os.getenv("N_LAYERS_PROVE", "32"))        # full proof

# --- API ---
API_TITLE   = "zkLLM SmolLM2-360M Prover"
API_VERSION = "0.1.0"
