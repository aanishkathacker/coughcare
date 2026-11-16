"""COUGHVID metadata handling, label derivation and balanced sampling."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import AudioError, probe_duration
from .config import (
    CANDIDATES_PER_CLASS,
    DATASET_DIR,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    RANDOM_STATE,
    TARGET_PER_CLASS,
)

log = logging.getLogger(__name__)

HEALTHY_TOKENS = {"healthy", "health", "no condition", "none", "normal", "no"}
COVID_TOKENS = {"covid", "covid-19", "covid19", "positive"}
UNKNOWN_TOKENS = {"", "nan", "none", "unknown", "not sure", "prefer not to say", "n/a", "na"}


def load_metadata(path: str | Path = DATASET_DIR / "metadata_compiled.csv") -> pd.DataFrame:
    """Load and normalise the COUGHVID metadata CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "uuid" not in df.columns:
        df = df.rename(columns={df.columns[0]: "uuid"})
    df["uuid"] = df["uuid"].astype(str).str.strip()
    df = df[df["uuid"].str.len() > 0].drop_duplicates("uuid").reset_index(drop=True)
    if "cough_detected" in df.columns:
        df["cough_detected"] = pd.to_numeric(df["cough_detected"], errors="coerce")
    log.info("Loaded metadata: %d recordings, %d columns", len(df), df.shape[1])
    return df


def normalise_status(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip().lower()
    if text in COVID_TOKENS or text.startswith("covid"):
        return "covid"
    if text in HEALTHY_TOKENS:
        return "healthy"
    if "symptom" in text:
        return "symptomatic"
    if text in UNKNOWN_TOKENS:
        return "unknown"
    return text


def _parse_expert_labels(value: object) -> list[dict]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, dict):
        return [value]
    text = str(value).strip()
    if not text or text.lower() in UNKNOWN_TOKENS:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _diagnosis_vote(diagnosis: object, quality: object = "") -> int | None:
    """Map one expert diagnosis to 0/1, ignoring COVID-specific labels."""
    text = "" if diagnosis is None else str(diagnosis).strip().lower()
    if text in UNKNOWN_TOKENS or not text:
        return None
    if text.startswith("covid"):
        return None
    quality_text = "" if quality is None else str(quality).strip().lower()
    if quality_text and quality_text not in {"ok", "good", "acceptable"}:
        return None
    if text in {"healthy", "healthy_cough", "no condition", "none", "normal"}:
        return 0
    return 1


def expert_signal(row: pd.Series) -> int | None:
    """Return 0 (healthy), 1 (pathology noted) or None from expert labels.

    Supports both COUGHVID metadata layouts: JSON columns
    (``expert_labels_1``) and the flattened v1 columns
    (``diagnosis_1`` / ``quality_1``). COVID-19 expert diagnoses are ignored
    so this project never uses COVID status as a target.
    """
    votes: list[int] = []

    for column in [c for c in row.index if c.startswith("expert_labels")]:
        for label in _parse_expert_labels(row.get(column)):
            vote = _diagnosis_vote(label.get("diagnosis"), label.get("quality", ""))
            if vote is not None:
                votes.append(vote)

    for index in range(1, 6):
        diagnosis_col = f"diagnosis_{index}"
        if diagnosis_col in row.index:
            vote = _diagnosis_vote(row.get(diagnosis_col), row.get(f"quality_{index}", ""))
            if vote is not None:
                votes.append(vote)

    if not votes:
        return None
    return int(np.mean(votes) >= 0.5)


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``status_norm``, ``expert_label`` and the combined ``label`` column.

    Label policy (documented in the README):
    * expert consensus (when available and of acceptable quality) takes priority
    * otherwise self-reported ``symptomatic`` -> atypical (1)
    * otherwise self-reported ``healthy`` -> normal (0)
    * self-reported ``COVID-19`` rows are excluded entirely (no disease target)
    * expert diagnoses naming COVID-19 are ignored, not mapped to a target
    * anything else is excluded

    Expert labels exist for only a small, stratified subset of COUGHVID, so
    most rows are labelled from the self-report. This is a noisy proxy for
    "atypical cough" and is reported as such.
    """
    out = df.copy()
    if "status" not in out.columns:
        raise ValueError("COUGHVID metadata has no 'status' column; cannot derive labels.")
    out["status_norm"] = out["status"].map(normalise_status)
    out["expert_label"] = out.apply(expert_signal, axis=1)

    labels: list[float] = []
    for status, expert in zip(out["status_norm"], out["expert_label"]):
        if status == "covid":
            labels.append(np.nan)
        elif expert is not None and not pd.isna(expert):
            labels.append(float(int(expert)))
        elif status == "symptomatic":
            labels.append(1.0)
        elif status == "healthy":
            labels.append(0.0)
        else:
            labels.append(np.nan)
    out["label"] = labels
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    return out


def assign_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Attach a grouping key for leakage-safe splitting.

    COUGHVID's public metadata does not expose a persistent participant ID in
    most releases. If one exists it is used; otherwise we fall back to the
    recording UUID and record that limitation explicitly.
    """
    out = df.copy()
    for candidate in ("user_id", "participant_id", "subject_id", "device_id"):
        if candidate in out.columns:
            values = out[candidate].astype(str).str.strip()
            if values.replace("", np.nan).notna().mean() > 0.5:
                out["group"] = values
                log.info("Using '%s' for user-level grouping", candidate)
                return out, True
    out["group"] = out["uuid"]
    return out, False


def _find_audio(dataset_dir: Path, uuid: str) -> Path | None:
    for ext in (".webm", ".ogg", ".wav", ".mp3", ".m4a", ".flac"):
        candidate = dataset_dir / f"{uuid}{ext}"
        if candidate.exists():
            return candidate
    return None


def select_samples(
    df: pd.DataFrame,
    dataset_dir: str | Path = DATASET_DIR,
    target_per_class: int = TARGET_PER_CLASS,
    candidates_per_class: int = CANDIDATES_PER_CLASS,
    min_duration: float = MIN_DURATION_SECONDS,
    max_duration: float = MAX_DURATION_SECONDS,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """Pick a balanced subset of valid coughs, skipping too-short clips.

    Only headers are probed to read duration, keeping this fast. Returns the
    manifest DataFrame and a summary dict for the metrics report.
    """
    dataset_dir = Path(dataset_dir)
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if "cough_detected" not in df.columns:
        raise ValueError("Metadata has no 'cough_detected' column.")
    before = len(df)
    df = df[df["cough_detected"] >= 0.8]
    log.info("cough_detected >= 0.8: %d -> %d rows", before, len(df))

    accepted: list[dict] = []
    stats = {"skipped_missing": 0, "skipped_short": 0, "skipped_long": 0, "skipped_decode": 0}
    for label in (0, 1):
        pool = df[df["label"] == label]
        if pool.empty:
            raise ValueError(f"No samples available for class {label}.")
        pool = pool.sample(frac=1.0, random_state=seed + label)
        picked = 0
        for _, row in pool.iterrows():
            if picked >= target_per_class or picked >= candidates_per_class:
                break
            audio_path = _find_audio(dataset_dir, row["uuid"])
            if audio_path is None:
                stats["skipped_missing"] += 1
                continue
            try:
                duration = probe_duration(audio_path)
            except AudioError:
                stats["skipped_decode"] += 1
                continue
            if duration < min_duration:
                stats["skipped_short"] += 1
                continue
            if duration > max_duration:
                stats["skipped_long"] += 1
                continue
            accepted.append(
                {
                    "uuid": row["uuid"],
                    "path": str(audio_path.relative_to(dataset_dir)),
                    "label": int(label),
                    "group": str(row.get("group", row["uuid"])),
                    "duration_s": round(float(duration), 3),
                    "status": row.get("status_norm", "unknown"),
                    "cough_detected": round(float(row["cough_detected"]), 4),
                }
            )
            picked += 1
        log.info("Class %d: selected %d clips", label, picked)

    manifest = pd.DataFrame(accepted)
    if manifest.empty:
        raise ValueError("No usable clips were found. Check the dataset directory.")
    counts = manifest["label"].value_counts().to_dict()
    summary = {
        "valid_cough_threshold": 0.8,
        "min_duration_s": min_duration,
        "max_duration_s": max_duration,
        "target_per_class": target_per_class,
        "selected_total": int(len(manifest)),
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "skipped": stats,
        "excluded_covid_status": True,
    }
    return manifest.reset_index(drop=True), summary
