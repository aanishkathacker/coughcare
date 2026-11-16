"""Inference: load a clip, extract features, return a cautious prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .audio import AudioError, AudioQuality, assess_quality, extract_cough_window, load_audio, to_wav_bytes, waveform_peaks
from .config import MODEL_PATH
from .features import extract_features

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: int
    label_name: str
    probability_atypical: float
    confidence: float
    threshold: float
    quality: AudioQuality
    duration_s: float
    window_duration_s: float
    peaks: list[float]
    playback_wav: bytes
    mode: str = "real"
    model_note: str = ""

    @property
    def is_atypical(self) -> bool:
        return self.label == 1


def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python scripts/train_model.py` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Model file is not a valid CoughCare bundle.")
    return bundle


def _heuristic_probability(y: np.ndarray, sr: int) -> float:
    """Transparent fallback used only in clearly-labelled demo mode.

    This is deliberately simple and is *not* trained on COUGHVID. It exists so
    the UI can be demonstrated if the dataset/model is unavailable.
    """
    import librosa

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) / (sr / 2)
    score = 0.35 + 1.4 * flatness + 0.6 * zcr + 0.25 * centroid
    return float(1 / (1 + np.exp(-4 * (score - 0.55))))


def analyze_audio(
    source: str | Path | bytes,
    bundle: dict | None = None,
    demo: bool = False,
) -> Prediction:
    """Run the full clip -> features -> probability pipeline.

    Raises :class:`AudioError` when the clip is silent, unreadable or too short
    to analyse. Quality warnings are attached to the returned prediction.
    """
    y, sr = load_audio(source)
    quality = assess_quality(y, sr)
    if not quality.is_usable:
        message = "; ".join(issue.split(":", 1)[-1].strip() for issue in quality.issues)
        raise AudioError(message)

    window = extract_cough_window(y, sr)
    window_duration = float(len(window) / sr)
    peaks = waveform_peaks(window, 360)
    playback = to_wav_bytes(window, sr)

    if demo:
        prob = _heuristic_probability(window, sr)
        mode = "demo"
        note = "Demo heuristic — not trained on COUGHVID. Results are illustrative only."
        threshold = 0.5
    else:
        bundle = bundle or load_bundle()
        features = extract_features(window, sr)
        pipeline = bundle["pipeline"]
        expected = int(getattr(pipeline, "n_features_in_", features.size))
        if features.size != expected:
            raise RuntimeError(
                f"Feature mismatch: model expects {expected}, got {features.size}. Retrain the model."
            )
        prob = float(pipeline.predict_proba(features.reshape(1, -1))[0, 1])
        threshold = float(bundle.get("threshold", 0.5))
        mode = str(bundle.get("mode", "real"))
        note = "Logistic regression trained on COUGHVID."

    label = int(prob >= threshold)
    confidence = prob if label == 1 else 1 - prob
    log.info(
        "Prediction %s (p=%.3f, confidence=%.3f, duration=%.2fs, warnings=%d)",
        "atypical" if label else "normal",
        prob,
        confidence,
        quality.duration_s,
        len(quality.warnings),
    )
    return Prediction(
        label=label,
        label_name="atypical" if label == 1 else "normal",
        probability_atypical=prob,
        confidence=confidence,
        threshold=threshold,
        quality=quality,
        duration_s=quality.duration_s,
        window_duration_s=window_duration,
        peaks=peaks,
        playback_wav=playback,
        mode=mode,
        model_note=note,
    )
