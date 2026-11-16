"""Project-wide paths and audio/feature constants."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Repository root (…/coughcare). This file lives at src/coughcare/config.py.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = RAW_DIR / "public_dataset"
METADATA_CSV = DATASET_DIR / "metadata_compiled.csv"
ZIP_PATH = RAW_DIR / "public_dataset.zip"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = ROOT / "assets"
SCREENSHOTS_DIR = ROOT / "screenshots"

MODEL_PATH = MODELS_DIR / "coughcare_lr.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MANIFEST_PATH = PROCESSED_DIR / "train_manifest.csv"
OOF_PATH = REPORTS_DIR / "oof_predictions.csv"

# Audio
SAMPLE_RATE = 16_000
COUGH_WINDOW_SECONDS = 1.5
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 15.0

# Features
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
N_MFCC = 20
FMAX = 8_000

# Dataset / experiment
RANDOM_STATE = 42
TARGET_PER_CLASS = 500
CANDIDATES_PER_CLASS = 1_400
N_SPLITS = 5
BOOTSTRAP_ROUNDS = 1_000

# Files and labels
AUDIO_EXTENSIONS = (".webm", ".ogg", ".wav", ".mp3", ".m4a", ".flac")
CLASS_NAMES = {0: "normal", 1: "atypical"}
POSITIVE_LABEL = 1

# Deployment / demo
DEMO_MODE_ENV = "COUGHCARE_DEMO"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single concise log format for scripts and the app."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs() -> None:
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        ASSETS_DIR,
        SCREENSHOTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def demo_mode() -> bool:
    """True when the app was explicitly started in labelled demo mode."""
    return os.environ.get(DEMO_MODE_ENV, "").strip().lower() in {"1", "true", "yes"}
