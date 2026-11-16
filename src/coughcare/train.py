"""Training pipeline: balanced COUGHVID subset, grouped CV, honest metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coughcare.audio import AudioError, load_audio
from coughcare.config import (  # noqa: E402
    BOOTSTRAP_ROUNDS,
    DATASET_DIR,
    FIGURES_DIR,
    MANIFEST_PATH,
    METADATA_CSV,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    N_SPLITS,
    OOF_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    configure_logging,
    ensure_dirs,
)
from coughcare.data import derive_labels, assign_groups, load_metadata, select_samples  # noqa: E402
from coughcare.features import extract_features, feature_names  # noqa: E402
from coughcare.metrics import baseline_metrics, binary_metrics, bootstrap_ci  # noqa: E402

log = logging.getLogger("train")

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def prepare_manifest(rebuild: bool = False) -> tuple[pd.DataFrame, dict]:
    if MANIFEST_PATH.exists() and not rebuild:
        manifest = pd.read_csv(MANIFEST_PATH)
        summary_path = PROCESSED_DIR / "selection_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        log.info("Loaded cached manifest: %d clips", len(manifest))
        return manifest, summary

    log.info("Building balanced manifest from metadata…")
    metadata = load_metadata(METADATA_CSV)
    labeled = derive_labels(metadata)
    log.info(
        "Label counts before filtering: %s",
        labeled["label"].value_counts(dropna=False).to_dict(),
    )
    grouped, has_user_id = assign_groups(labeled)
    manifest, summary = select_samples(grouped, dataset_dir=DATASET_DIR)
    summary["user_level_grouping_available"] = bool(has_user_id)
    summary["grouping_key"] = "user id" if has_user_id else "recording uuid (fallback)"
    summary["expert_labels_rows"] = int(labeled["expert_label"].notna().sum())

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    (PROCESSED_DIR / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Manifest saved: %s", MANIFEST_PATH)
    return manifest, summary


def build_features(manifest: pd.DataFrame, rebuild: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = PROCESSED_DIR / "features.npz"
    uuids = manifest["uuid"].astype(str).to_numpy()
    if cache.exists() and not rebuild:
        cached = np.load(cache, allow_pickle=False)
        if set(cached["uuids"].tolist()) == set(uuids.tolist()):
            log.info("Loaded cached features: %s", cache)
            return cached["X"], cached["y"], cached["groups"]

    X: list[np.ndarray] = []
    failed: list[str] = []
    started = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), 1):
        path = DATASET_DIR / row.path
        try:
            y_audio, sr = load_audio(path)
            X.append(extract_features(y_audio, sr))
        except (AudioError, ValueError) as exc:
            log.warning("Skipping %s: %s", row.uuid, exc)
            failed.append(row.uuid)
        if i % 100 == 0:
            log.info("Features %d / %d (%.1f%%)", i, len(manifest), 100 * i / len(manifest))

    if failed:
        keep = ~manifest["uuid"].isin(failed)
        manifest = manifest[keep].reset_index(drop=True)
        manifest.to_csv(MANIFEST_PATH, index=False)
        log.warning("Dropped %d undecodable clips; manifest updated.", len(failed))

    X_arr = np.vstack(X).astype(np.float32)
    y_arr = manifest["label"].astype(int).to_numpy()
    groups = manifest["group"].astype(str).to_numpy()
    np.savez_compressed(
        cache,
        X=X_arr,
        y=y_arr,
        groups=np.asarray(groups, dtype="U128"),
        uuids=np.asarray(manifest["uuid"], dtype="U64"),
    )
    log.info("Features built in %.1fs: %s", time.time() - started, X_arr.shape)
    return X_arr, y_arr, groups


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    min_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, min_groups))
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5_000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def cross_validate(X, y, groups, n_splits: int, seed: int) -> tuple[np.ndarray, list[dict], list[dict]]:
    oof = np.full(len(y), np.nan)
    per_fold: list[dict] = []
    selected_c: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(grouped_folds(y, groups, n_splits, seed).split(X, y, groups), 1):
        log.info("Outer fold %d: train=%d test=%d", fold, len(train_idx), len(test_idx))
        search = GridSearchCV(
            make_pipeline(),
            param_grid={"clf__C": C_GRID},
            scoring="roc_auc",
            cv=grouped_folds(y[train_idx], groups[train_idx], 3, seed + fold),
            n_jobs=-1,
            refit=True,
        )
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        prob = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = prob
        metrics = binary_metrics(y[test_idx], prob)
        metrics["fold"] = fold
        metrics["best_C"] = float(search.best_params_["clf__C"])
        per_fold.append(metrics)
        selected_c.append(float(search.best_params_["clf__C"]))
        log.info(
            "Fold %d: AUC=%.3f acc=%.3f sens=%.3f spec=%.3f C=%s",
            fold,
            metrics["roc_auc"],
            metrics["accuracy"],
            metrics["sensitivity"],
            metrics["specificity"],
            search.best_params_["clf__C"],
        )

    if np.isnan(oof).any():
        raise RuntimeError("Some samples were not covered by the CV splits.")
    return oof, per_fold, selected_c


def fit_final(X, y, groups, selected_c: list[float]) -> tuple[Pipeline, float]:
    """Fit the final model with the C chosen most often in outer folds.

    Tuning C on all data after nested CV can pick an unstable, overfit value
    (the inner folds are small). The modal outer-fold choice is more stable and
    lets the reported out-of-fold metrics correspond to the deployed model
    family.
    """
    from collections import Counter

    best_c = float(Counter(selected_c).most_common(1)[0][0])
    log.info("Final C=%s (modal outer-fold choice %s); fitting on all %d samples.", best_c, selected_c, len(y))
    model = make_pipeline().set_params(clf__C=best_c)
    model.fit(X, y)
    return model, best_c


def save_figures(y: np.ndarray, oof: np.ndarray, metrics: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    accent = "#0E6F5C"
    ink = "#1D2A26"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#B9B1A2", "text.color": ink, "axes.labelcolor": ink})

    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=160)
    ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Atypical"]).plot(
        ax=ax, cmap="BuGn", colorbar=False, values_format="d"
    )
    ax.set_title("Held-out confusion matrix (threshold 0.50)", color=ink)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", transparent=False, facecolor="#F7F3EA")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=160)
    fig.patch.set_facecolor("#F7F3EA")
    for ax in axes:
        ax.set_facecolor("#F7F3EA")

    fpr, tpr, _ = roc_curve(y, oof)
    axes[0].plot(fpr, tpr, color=accent, linewidth=2, label="CoughCare")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#8B8578", linewidth=1, label="Chance")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUC {metrics['roc_auc']:.3f})")
    axes[0].legend(loc="lower right", frameon=False)

    precision, recall, _ = precision_recall_curve(y, oof)
    axes[1].plot(recall, precision, color=accent, linewidth=2, label="CoughCare")
    axes[1].axhline(float(np.mean(y)), linestyle="--", color="#8B8578", linewidth=1, label="Prevalence")
    axes[1].set_xlabel("Recall (sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"Precision–recall (AP {metrics['pr_auc']:.3f})")
    axes[1].legend(loc="upper right", frameon=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=160)
    fig.patch.set_facecolor("#F7F3EA")
    for ax in axes:
        ax.set_facecolor("#F7F3EA")

    ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Atypical"]).plot(
        ax=axes[0], cmap="BuGn", colorbar=False, values_format="d"
    )
    axes[0].set_title("Held-out predictions (threshold 0.50)")

    fpr, tpr, _ = roc_curve(y, oof)
    axes[1].plot(fpr, tpr, color=accent, linewidth=2, label="CoughCare")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#8B8578", linewidth=1, label="Chance")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC curve (AUC {metrics['roc_auc']:.3f})")
    axes[1].legend(loc="lower right", frameon=False)

    precision, recall, _ = precision_recall_curve(y, oof)
    axes[2].plot(recall, precision, color=accent, linewidth=2, label="CoughCare")
    axes[2].axhline(float(np.mean(y)), linestyle="--", color="#8B8578", linewidth=1, label="Prevalence")
    axes[2].set_xlabel("Recall (sensitivity)")
    axes[2].set_ylabel("Precision")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title(f"Precision–recall (AP {metrics['pr_auc']:.3f})")
    axes[2].legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evaluation_panel.png", facecolor="#F7F3EA")
    plt.close(fig)
    log.info("Figures written to %s", FIGURES_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-data", action="store_true", help="Re-select samples from metadata.")
    parser.add_argument("--rebuild-features", action="store_true", help="Ignore cached features.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ROUNDS)
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    started = time.time()

    manifest, summary = prepare_manifest(rebuild=args.rebuild_data)
    X, y, groups = build_features(manifest, rebuild=args.rebuild_features)
    log.info("Feature matrix: %s | class counts: %s", X.shape, np.bincount(y).tolist())

    oof, per_fold, selected_c = cross_validate(X, y, groups, N_SPLITS, RANDOM_STATE)
    primary = binary_metrics(y, oof)
    ci = bootstrap_ci(y, oof, rounds=args.bootstrap, seed=RANDOM_STATE)
    baselines = baseline_metrics(y)

    final_model, best_c = fit_final(X, y, groups, selected_c)

    folds_auc = np.array([f["roc_auc"] for f in per_fold])
    metrics = {
        "mode": "real",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "COUGHVID (Zenodo 4048312)",
            "license": "CC BY 4.0",
            "selection": summary,
            "class_balance": {
                "normal": int(np.sum(y == 0)),
                "atypical": int(np.sum(y == 1)),
                "normal_fraction": float(np.mean(y == 0)),
            },
            "label_definition": (
                "Normal = self-reported healthy with no expert pathology; atypical = "
                "self-reported symptomatic or expert pathology. COVID-19 self-reports "
                "were excluded. Expert labels used only when present."
            ),
            "excluded_covid_rows": True,
        },
        "features": {
            "count": int(X.shape[1]),
            "type": "log-mel, MFCC + delta, spectral statistics",
            "window_seconds": 1.5,
        },
        "split": {
            "type": "StratifiedGroupKFold",
            "n_splits": int(N_SPLITS),
            "grouping_key": summary.get("grouping_key", "recording uuid (fallback)"),
            "user_level_grouping_available": bool(summary.get("user_level_grouping_available", False)),
            "note": (
                "COUGHVID's public metadata has no participant identifier in this release, "
                "so recording UUID is used as the group. Participant-level leakage cannot "
                "be fully ruled out; treat metrics as an upper bound."
            ),
        },
        "model": {
            "name": "StandardScaler + LogisticRegression",
            "class_weight": "balanced",
            "selected_C": best_c,
            "outer_fold_C": selected_c,
            "threshold": 0.5,
            "n_features": int(X.shape[1]),
        },
        "primary_metrics": primary,
        "bootstrap_ci_95": ci,
        "per_fold": per_fold,
        "fold_auc_mean": float(folds_auc.mean()),
        "fold_auc_std": float(folds_auc.std(ddof=1)),
        "baselines": baselines,
        "limitations": [
            "Research prototype only; it does not diagnose disease.",
            "Healthy vs symptomatic is a noisy, self-reported label; no clinical confirmation.",
            "Crowdsourced phone recordings vary in microphone, room and cough technique.",
            "The balanced 1,000-clip subset is small and not representative of Delhi.",
            "No participant identifier in the public metadata; grouping uses recording UUID.",
            "Metrics are cross-validated on COUGHVID, not on external or local Delhi data.",
        ],
        "runtime_seconds": round(time.time() - started, 1),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_model,
        "feature_names": feature_names(),
        "threshold": 0.5,
        "label_names": {0: "normal", 1: "atypical"},
        "mode": "real",
        "trained_at": metrics["generated_at"],
        "dataset": "COUGHVID (Zenodo 4048312)",
        "metrics": {
            "roc_auc": primary["roc_auc"],
            "pr_auc": primary["pr_auc"],
            "accuracy": primary["accuracy"],
            "sensitivity": primary["sensitivity"],
            "specificity": primary["specificity"],
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"uuid": manifest["uuid"], "y_true": y, "y_prob": oof}).to_csv(OOF_PATH, index=False)
    save_figures(y, oof, primary)

    log.info("=" * 62)
    log.info("Held-out metrics (5-fold grouped CV)")
    log.info("Accuracy %.3f | Sensitivity %.3f | Specificity %.3f", primary["accuracy"], primary["sensitivity"], primary["specificity"])
    log.info("ROC-AUC %.3f | PR-AUC %.3f | F1 %.3f", primary["roc_auc"], primary["pr_auc"], primary["f1"])
    log.info("95%% CI accuracy: %.3f–%.3f | AUC: %.3f–%.3f", ci["accuracy"]["lower"], ci["accuracy"]["upper"], ci["roc_auc"]["lower"], ci["roc_auc"]["upper"])
    log.info("Majority baseline accuracy: %.3f", baselines["majority_class"]["accuracy"])
    log.info("Model saved: %s", MODEL_PATH)
    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
