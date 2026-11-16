from __future__ import annotations

import base64

from coughcare.components import waveform_player


def test_waveform_player_embeds_audio_and_peaks():
    wav = b"RIFF" + b"\x00" * 100
    peaks = [0.0, 0.5, 1.0, 0.25]
    html = waveform_player(wav, peaks, 1.5)
    assert "data:audio/wav;base64," + base64.b64encode(wav).decode() in html
    assert "[0.0, 0.5, 1.0, 0.25]" in html
    assert "requestAnimationFrame" in html
    assert "const DURATION = 1.5" in html
    assert "<canvas" in html


def test_waveform_player_escapes_no_user_text():
    html = waveform_player(b"RIFF", [0.1], 1.0)
    # Only numeric data and static markup are interpolated.
    assert "<script" in html
    assert "</html>" in html
