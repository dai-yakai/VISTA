"""Project configuration constants. All paths are absolute."""
import os

# --- paths ---
NETARENA_DIR = "/home/ubuntu/NetArena"
NETARENA_MALT_DIR = os.path.join(NETARENA_DIR, "app-malt")
NETARENA_SRC_DIR = os.path.join(NETARENA_DIR, "src")

VISTA_ROOT = "/home/ubuntu/vista"
DATA_DIR = os.path.join(VISTA_ROOT, "data")
MODELS_DIR = os.path.join(VISTA_ROOT, "models")
OUTPUTS_DIR = os.path.join(VISTA_ROOT, "outputs")
LOGS_DIR = os.path.join(VISTA_ROOT, "logs")
CACHE_DIR = os.path.join(VISTA_ROOT, "cache")

# --- raw data ---
MALT_RAW_DATA = os.path.join(NETARENA_MALT_DIR, "data", "sampled_500_benchmark_malt.jsonl")

# --- splits ---
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE  = os.path.join(DATA_DIR, "test.jsonl")
SFT_FILE   = os.path.join(DATA_DIR, "sft_train.jsonl")  # ShareGPT-style for trl

# --- models ---
BASE_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-Coder-7B-Instruct")
EMBED_MODEL = os.path.join(MODELS_DIR, "bge-base-en-v1.5")
SFT_OUTPUT_DIR = os.path.join(MODELS_DIR, "sft-qwen-coder-7b-malt")

# --- splits config ---
TEST_PER_LABEL = 100   # 100 per label × 10 labels = 1000 test
TRAIN_PER_LABEL = 400  # 400 per label × 10 labels = 4000 train
SEED = 42

# --- vLLM defaults ---
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_MAX_MODEL_LEN = 4096
VLLM_DTYPE = "bfloat16"

# --- generation defaults ---
GEN_MAX_TOKENS = 1024
GEN_TEMPERATURE_GREEDY = 0.0
GEN_TEMPERATURE_SAMPLE = 0.7
GEN_TOP_P = 0.95

# --- VISTA hyperparameters ---
VISTA_NUM_HYPOTHESES = 5
VISTA_MAX_REPAIR_ITER = 3
VISTA_RETRIEVAL_K = 4

# Number of test queries per evaluation pass. None means use full TEST_FILE.
EVAL_LIMIT = None

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
