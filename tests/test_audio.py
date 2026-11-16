from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from coughcare.audio import (
    AudioError,
    assess_quality,
    extract_cough_window,
    load_audio,
    probe_duration,
    to_wav_bytes,
    waveform_peaks,
)


def make_wav(path, seconds=2.0, sr=16_000, freq=220.0, amplitude=0.4):
    t = np.arange(int(seconds * sr)) / sr
    y = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, y, sr)
    return path


def test_load_audio_from_path_and_bytes(tmp_path):
    path = make_wav(tmp_path / "tone.wav")
    y, sr = load_audio(path)
    assert sr == 16_000
    assert abs(len(y) / sr - 2.0) < 0.02
    assert y.dtype == np.float32

    y2, sr2 = load_audio(path.read_bytes())
    assert sr2 == sr
    np.testing.assert_allclose(y, y2, atol=1e-5)


def test_probe_duration(tmp_path):
    path = make_wav(tmp_path / "tone.wav", seconds=3.0)
    assert abs(probe_duration(path) - 3.0) < 0.05


def test_quality_warns_on_short_and_quiet():
    sr = 16_000
    short = np.zeros(int(0.2 * sr), dtype=np.float32)
    quality = assess_quality(short, sr)
    assert not quality.is_usable
    assert quality.issues

    quiet = (1e-5 * np.ones(int(2 * sr))).astype(np.float32)
    quality = assess_quality(quiet, sr)
    assert not quality.is_usable


def test_quality_ok_for_clean_clip():
    sr = 16_000
    t = np.arange(2 * sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    quality = assess_quality(y, sr)
    assert quality.is_usable
    assert quality.duration_s == 2.0


def test_extract_window_returns_fixed_length():
    sr = 16_000
    y = np.zeros(5 * sr, dtype=np.float32)
    y[3 * sr : 3 * sr + int(0.4 * sr)] = 0.8
    window = extract_cough_window(y, sr, window_s=1.5)
    assert len(window) == int(1.5 * sr)
    # The loud cough should be inside the selected window.
    assert np.max(np.abs(window)) >= 0.79


def test_waveform_peaks_normalised():
    y = np.linspace(-1, 1, 1000, dtype=np.float32)
    peaks = waveform_peaks(y, n_points=64)
    assert len(peaks) == 64
    assert max(peaks) == pytest.approx(1.0, abs=1e-3)
    assert min(peaks) >= 0.0


def test_silent_audio_rejected():
    with pytest.raises(AudioError):
        load_audio(b"")


def test_to_wav_roundtrip():
    y = np.zeros(16_000, dtype=np.float32)
    y[8_000:9_000] = 0.5
    data = to_wav_bytes(y)
    assert data[:4] == b"RIFF"
    y2, sr = load_audio(data)
    assert sr == 16_000
    assert len(y2) == 16_000
