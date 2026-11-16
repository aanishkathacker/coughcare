from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert any("CoughCare" in str(m.value) for m in at.markdown)
    assert len(at.radio) >= 1
    assert at.radio[0].value == "Check a cough"
