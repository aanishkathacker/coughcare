from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf

from coughcare.data import derive_labels, normalise_status, select_samples
from coughcare.metrics import baseline_metrics, binary_metrics, bootstrap_ci


def test_status_normalisation():
    assert normalise_status("healthy") == "healthy"
    assert normalise_status("Symptomatic") == "symptomatic"
    assert normalise_status("COVID-19") == "covid"
    assert normalise_status(None) == "unknown"


def test_derive_labels_combines_expert_and_self_report():
    df = pd.DataFrame(
        {
            "uuid": ["a", "b", "c", "d"],
            "status": ["healthy", "symptomatic", "COVID-19", "healthy"],
            "expert_labels_1": [
                None,
                None,
                None,
                json.dumps({"quality": "ok", "diagnosis": "asthma"}),
            ],
        }
    )
    out = derive_labels(df)
    assert out.loc[0, "label"] == 0
    assert out.loc[1, "label"] == 1
    assert pd.isna(out.loc[2, "label"])  # COVID rows excluded
    assert out.loc[3, "label"] == 1  # expert pathology -> atypical


def test_expert_covid_label_ignored():
    df = pd.DataFrame(
        {
            "uuid": ["a"],
            "status": ["healthy"],
            "expert_labels_1": [json.dumps({"quality": "ok", "diagnosis": "COVID-19"})],
        }
    )
    out = derive_labels(df)
    assert out.loc[0, "label"] == 0


def test_flattened_v1_expert_columns():
    df = pd.DataFrame(
        {
            "uuid": ["a", "b", "c"],
            "status": ["healthy", "healthy", "symptomatic"],
            "diagnosis_1": ["lower_infection", "healthy_cough", "COVID-19"],
            "quality_1": ["ok", "ok", "ok"],
        }
    )
    out = derive_labels(df)
    assert out.loc[0, "label"] == 1  # expert pathology overrides self-report
    assert out.loc[1, "label"] == 0
    assert out.loc[2, "label"] == 1  # self-report symptomatic, COVID diagnosis ignored


def test_select_samples_filters_and_balances(tmp_path):
    sr = 16_000
    rows = []
    for i in range(8):
        uid = f"clip{i}"
        seconds = 0.5 if i in (0, 4) else 2.0
        y = np.zeros(int(seconds * sr), dtype=np.float32)
        y[sr // 2 : sr // 2 + sr // 4] = 0.5
        sf.write(tmp_path / f"{uid}.wav", y, sr)
        rows.append(
            {
                "uuid": uid,
                "label": 0 if i < 4 else 1,
                "group": f"user{i % 2}",
                "cough_detected": 0.95 if i != 1 else 0.5,
            }
        )
    manifest, summary = select_samples(
        pd.DataFrame(rows), dataset_dir=tmp_path, target_per_class=2, candidates_per_class=10
    )
    assert set(manifest["label"]) == {0, 1}
    assert len(manifest) == 4  # clip0/4 too short, clip1 filtered by cough_detected
    assert (manifest["duration_s"] >= 1.0).all()
    assert summary["selected_total"] == 4
    assert summary["class_counts"] == {"0": 2, "1": 2}


def test_binary_metrics_perfect_and_confusion():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.15, 100), 0, 1)
    m = binary_metrics(y, p)
    ci = bootstrap_ci(y, p, rounds=200, seed=1)
    assert ci["accuracy"]["lower"] <= m["accuracy"] <= ci["accuracy"]["upper"]
    assert ci["roc_auc"]["lower"] <= m["roc_auc"] <= ci["roc_auc"]["upper"]


def test_majority_baseline():
    y = np.array([0] * 8 + [1] * 2)
    baseline = baseline_metrics(y)
    assert baseline["majority_class"]["predicted_class"] == "normal"
    assert baseline["majority_class"]["accuracy"] == 0.8
