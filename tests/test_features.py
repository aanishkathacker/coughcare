from __future__ import annotations

import numpy as np
import pytest

from coughcare.features import extract_features, feature_names


def cough_like(seconds=1.5, sr=16_000, seed=0):
    rng = np.random.default_rng(seed)
    y = 0.01 * rng.standard_normal(int(seconds * sr)).astype(np.float32)
    burst = min(int(0.25 * sr), len(y) // 2)
    start = min(int(0.5 * sr), max(0, len(y) - burst))
    t = np.arange(burst) / sr
    envelope = np.exp(-8 * t)
    y[start : start + burst] += (0.7 * envelope * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
    return y, sr


def test_feature_vector_shape_and_finiteness():
    y, sr = cough_like()
    vector = extract_features(y, sr)
    assert vector.shape == (len(feature_names()),)
    assert np.isfinite(vector).all()
    assert vector.dtype == np.float32


def test_features_are_deterministic():
    y, sr = cough_like(seed=7)
    a = extract_features(y, sr)
    b = extract_features(y, sr)
    np.testing.assert_array_equal(a, b)


def test_short_clip_is_padded_not_crashing():
    y, sr = cough_like(seconds=0.1)
    vector = extract_features(y, sr)
    assert vector.shape == (len(feature_names()),)
    assert np.isfinite(vector).all()


def test_silence_raises():
    with pytest.raises(ValueError):
        extract_features(np.zeros(1600, dtype=np.float32))
