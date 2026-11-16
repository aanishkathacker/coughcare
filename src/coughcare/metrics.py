"""Evaluation metrics and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Confusion-matrix based metrics for the positive (atypical) class."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float((recall_score(y_true, y_pred, pos_label=1, zero_division=0) + specificity) / 2),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "prevalence": float(np.mean(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    rounds: int = 1_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap 95% CIs over held-out predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc")
    samples: dict[str, list[float]] = {key: [] for key in keys}
    n = len(y_true)

    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics = binary_metrics(y_true[idx], y_prob[idx], threshold)
        for key in keys:
            samples[key].append(metrics[key])

    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([v for v in samples[key] if np.isfinite(v)])
        if values.size == 0:
            out[key] = {"lower": float("nan"), "upper": float("nan")}
        else:
            out[key] = {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
    out["_meta"] = {"rounds": int(rounds), "valid_rounds": int(len(samples["accuracy"]))}
    return out


def baseline_metrics(y_true: np.ndarray) -> dict:
    """Majority-class and chance baselines for context."""
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(np.mean(y_true))
    majority = int(prevalence > 0.5)  # 0 when normals are the majority (or tie)
    majority_pred = np.full_like(y_true, majority)
    cm = confusion_matrix(y_true, majority_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = cm
    return {
        "majority_class": {
            "predicted_class": "normal" if majority == 0 else "atypical",
            "accuracy": float(max(prevalence, 1 - prevalence)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "roc_auc": 0.5,
            "pr_auc": prevalence,
        },
        "chance": {"accuracy": 0.5, "sensitivity": 0.5, "specificity": 0.5, "roc_auc": 0.5, "pr_auc": prevalence},
    }
