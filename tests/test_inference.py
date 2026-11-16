from __future__ import annotations

import numpy as np
import pytest

from coughcare.config import MODEL_PATH
from coughcare.inference import analyze_audio

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not present")


def test_real_model_prediction_contract():
    from coughcare.config import ASSETS_DIR

    pred = analyze_audio(ASSETS_DIR / "samples" / "atypical.webm")
    assert pred.mode == "real"
    assert 0.0 <= pred.probability_atypical <= 1.0
    assert 0.5 <= pred.confidence <= 1.0
    assert pred.label in (0, 1)
    assert pred.label_name in {"normal", "atypical"}
    assert len(pred.peaks) == 360
    assert max(pred.peaks) > 0.1
    assert pred.window_duration_s == pytest.approx(1.5, abs=0.01)
    assert pred.playback_wav[:4] == b"RIFF"


def test_sample_clips_decode_and_predict():
    from coughcare.config import ASSETS_DIR

    for name in ("normal", "atypical"):
        pred = analyze_audio(ASSETS_DIR / "samples" / f"{name}.webm")
        assert np.isfinite(pred.probability_atypical)
        assert pred.quality.is_usable
