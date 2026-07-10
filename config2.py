import os
import random
import numpy as np
import torch
from pathlib import Path

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

def seed_everything(seed: int = SEED):
    """Lock every source of randomness so results are perfectly repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_generator(seed: int = SEED):
    """Deterministic generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    """Seed each DataLoader worker (called via worker_init_fn)."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"datapath")
DATA_DIR    = BASE_DIR / "gen_data"
TRAIN_DIR   = DATA_DIR / "train"
TEST_DIR    = DATA_DIR / "test2"          
OUTPUT_DIR  = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Classes  ───────────────────
CLASS_NAMES  = ["class 1", "class 2", "class 3", "class 4"]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ─── Data / model shape ───────────────────────────────────────────────────────
IN_CHANNELS = 1            
IMAGE_SIZE  = 128          
VAL_SPLIT   = 0.2          

# ─── Shared training budget ───────────────────────────────────────────────────
BATCH_SIZE        = 32      
GRAD_ACCUM_STEPS  = 2       
EPOCHS            = 40     
PATIENCE          = 8       
NUM_WORKERS       = 2       


VERSIONS = [
    {
        "name": "ver1", "aug_level": "geometric",
        "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.5,
        "label_smoothing": 0.0, "use_mixup": False, "mixup_alpha": 0.0,
    },
    {
        "name": "ver2", "aug_level": "robust",
        "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.5,
        "label_smoothing": 0.05, "use_mixup": False, "mixup_alpha": 0.0,
    },
    {
        "name": "ver3", "aug_level": "robust_plus",
        "lr": 7e-4, "weight_decay": 5e-4, "dropout": 0.5,
        "label_smoothing": 0.1, "use_mixup": True, "mixup_alpha": 0.2,
    },
]

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Output file paths ────────────────────────────────────────────────────────
NORM_STATS_PATH   = OUTPUT_DIR / "norm_stats.json"     
HISTORY_DIR       = OUTPUT_DIR                          
BEST_MODEL_PATH   = OUTPUT_DIR / "report2.pth"          
BEST_MODEL_PATH2  = BASE_DIR  / "report2.pth"           
UTIL_SAVE_PATH    = BASE_DIR  / "util2.txt"

def version_model_path(name: str) -> Path:
    return OUTPUT_DIR / f"report2_{name}.pth"

def version_history_path(name: str) -> Path:
    return OUTPUT_DIR / f"history_{name}.csv"