"""Configuration and constants for QuantPilot."""

import os
from pathlib import Path

# Base directories
QUANTPILOT_HOME = Path(os.getenv("QUANTPILOT_HOME", Path.home() / ".quantpilot"))
MODELS_DIR = QUANTPILOT_HOME / "models"
CACHE_DIR = QUANTPILOT_HOME / "cache"
PREBUILT_DIR = MODELS_DIR / "prebuilt"
CUSTOM_DIR = MODELS_DIR / "custom"
REGISTRY_FILE = MODELS_DIR / "registry.json"

# Training defaults
DEFAULT_LOOKBACK = 60  # days for factor calculation
DEFAULT_FORWARD = 20   # days for return prediction
DEFAULT_N_SPLITS = 5   # purged walk-forward folds
DEFAULT_EMBARGO = 5    # days between train/test

# LightGBM defaults
LGBM_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

# Evaluation thresholds
MIN_IC = 0.03
MIN_IR = 0.5
MIN_SHARPE = 1.0
MAX_DD = -0.20

# Pre-trained sector definitions
SECTORS = {
    "cpo": {
        "name": "CPO光模块",
        "source": "eastmoney",
        "concept_id": "BK1195",
        "min_stocks": 15,
    },
    "pcb": {
        "name": "PCB电路板",
        "source": "eastmoney",
        "concept_id": "BK0738",
        "min_stocks": 15,
    },
    "ai": {
        "name": "AI应用",
        "source": "eastmoney",
        "concept_id": "BK1131",
        "min_stocks": 15,
    },
    "embodied": {
        "name": "具身智能",
        "source": "eastmoney",
        "concept_id": "BK2097",
        "min_stocks": 15,
    },
    "consumer": {
        "name": "消费白马",
        "source": "custom",
        "concept_id": None,
        "min_stocks": 15,
    },
    "new_energy": {
        "name": "新能源",
        "source": "eastmoney",
        "concept_id": "BK0493",
        "min_stocks": 15,
    },
}


def ensure_dirs():
    """Create all required directories."""
    for d in [MODELS_DIR, CACHE_DIR, PREBUILT_DIR, CUSTOM_DIR]:
        d.mkdir(parents=True, exist_ok=True)
