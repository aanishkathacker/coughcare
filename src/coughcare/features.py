"""Feature extraction: compact MFCC + log-mel + spectral statistics."""

from __future__ import annotations

import logging

import librosa
import numpy as np

from .config import (
    COUGH_WINDOW_SECONDS,
    FMAX,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
)
from .audio import extract_cough_window

log = logging.getLogger(__name__)


def _stats(matrix: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Summary statistics per row (band/coefficient) of a 2-D feature matrix."""
    values = np.concatenate(
        [
            matrix.mean(axis=1),
            matrix.std(axis=1),
            np.percentile(matrix, 10, axis=1),
            np.percentile(matrix, 90, axis=1),
        ]
    ).astype(np.float32)
    names = [f"{prefix}_{stat}_{i}" for stat in ("mean", "std", "p10", "p90") for i in range(matrix.shape[0])]
    return values, names


def feature_names() -> list[str]:
    """Names in the exact order produced by :func:`extract_features`."""
    names: list[str] = []
    names += [f"logmel_{stat}_{i}" for stat in ("mean", "std", "p10", "p90") for i in range(N_MELS)]
    names += [f"mfcc_{stat}_{i}" for stat in ("mean", "std", "p10", "p90") for i in range(N_MFCC)]
    names += [f"deltamfcc_{stat}_{i}" for stat in ("mean", "std") for i in range(N_MFCC)]
    names += [
        "centroid_mean", "centroid_std",
        "bandwidth_mean", "bandwidth_std",
        "rolloff_mean", "rolloff_std",
        "flatness_mean", "flatness_std",
        "zcr_mean", "zcr_std",
        "rms_mean", "rms_std",
    ]
    names += [f"contrast_{i}" for i in range(7)]
    return names


def extract_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_s: float = COUGH_WINDOW_SECONDS,
) -> np.ndarray:
    """Extract a fixed-length feature vector from the loudest cough window."""
    if y.size == 0:
        raise ValueError("Cannot extract features from empty audio.")

    y = np.asarray(y, dtype=np.float32)
    y = extract_cough_window(y, sr, window_s=window_s)

    # Loudness-normalise so microphone gain does not dominate the features.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak

    # Guarantee enough samples for at least one frame.
    min_samples = N_FFT
    if y.size < min_samples:
        y = np.pad(y, (0, min_samples - y.size))
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if np.max(np.abs(y)) < 1e-7:
        raise ValueError("Audio is silent after preprocessing.")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=min(FMAX, sr // 2),
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    mfcc = librosa.feature.mfcc(
        S=logmel,
        n_mfcc=N_MFCC,
    )
    delta = librosa.feature.delta(mfcc, width=5, mode="nearest")

    contrast = librosa.feature.spectral_contrast(
        S=np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)),
        sr=sr,
        n_bands=6,
        fmin=100.0,
    )

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)

    parts: list[np.ndarray] = []
    parts.append(_stats(logmel, "logmel")[0])
    parts.append(_stats(mfcc, "mfcc")[0])
    parts.append(np.concatenate([delta.mean(axis=1), delta.std(axis=1)]).astype(np.float32))
    parts.append(
        np.array(
            [
                centroid.mean(), centroid.std(),
                bandwidth.mean(), bandwidth.std(),
                rolloff.mean(), rolloff.std(),
                flatness.mean(), flatness.std(),
                zcr.mean(), zcr.std(),
                rms.mean(), rms.std(),
            ],
            dtype=np.float32,
        )
    )
    parts.append(contrast.mean(axis=1).astype(np.float32))
    vector = np.concatenate(parts).astype(np.float32)

    expected = len(feature_names())
    if vector.size != expected:
        raise RuntimeError(f"Feature length mismatch: {vector.size} != {expected}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
