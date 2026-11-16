"""Capture reproducible screenshots of the running app with Playwright + Chrome.

Usage:
    streamlit run app.py --server.headless true &
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "screenshots"
SHOTS.mkdir(exist_ok=True)
BASE = "http://localhost:8501"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

errors: list[str] = []


def prepare(page: Page, url: str, text: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector(f"text={text}", timeout=90_000)
    page.evaluate("() => document.fonts.ready")
    page.wait_for_timeout(800)


def snap(page: Page, name: str, full_page: bool = False) -> None:
    path = SHOTS / name
    page.screenshot(path=str(path), full_page=full_page)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--force-color-profile=srgb"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", lambda msg: errors.append(f"console: {msg.text}") if msg.type == "error" else None)

        # 1. Home, desktop
        prepare(page, BASE, "Listen first.")
        snap(page, "01_home_desktop.png")

        # 2. Atypical sample result, desktop (tall viewport so the whole flow fits)
        page.set_viewport_size({"width": 1440, "height": 2600})
        prepare(page, BASE + "?sample=atypical", "Monitor this cough")
        page.wait_for_timeout(1500)
        snap(page, "02_result_atypical_desktop.png")

        # 3. Normal sample result, desktop
        prepare(page, BASE + "?sample=normal", "Normal cough")
        page.wait_for_timeout(1500)
        snap(page, "03_result_normal_desktop.png")

        # 4. Stats page, desktop
        page.set_viewport_size({"width": 1440, "height": 3200})
        prepare(page, BASE + "?page=stats", "The numbers,")
        snap(page, "04_stats_desktop.png")

        # 5. Mobile home (390 x 844, iPhone-ish)
        mobile = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
            is_mobile=True,
            has_touch=True,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            ),
        )
        mpage = mobile.new_page()
        mpage.on("pageerror", lambda exc: errors.append(f"mobile pageerror: {exc}"))
        prepare(mpage, BASE, "Listen first.")
        snap(mpage, "05_home_mobile.png")

        # 6. Mobile result, scrolled to the result card
        prepare(mpage, BASE + "?sample=atypical", "Monitor this cough")
        mpage.wait_for_timeout(1500)
        mpage.locator(".cc-result").scroll_into_view_if_needed()
        mpage.wait_for_timeout(500)
        snap(mpage, "06_result_atypical_mobile.png")

        # 7. Real upload path (not the sample shortcut), to validate decoding of WebM.
        page.set_viewport_size({"width": 1440, "height": 2600})
        prepare(page, BASE, "Upload a clip")
        page.get_by_role("tab", name="Upload a clip").click()
        page.locator('input[type="file"]').set_input_files(str(ROOT / "assets" / "samples" / "upload_validation.webm"))
        page.wait_for_selector("text=Monitor this cough", timeout=90_000)
        page.wait_for_timeout(1500)
        snap(page, "07_upload_result_desktop.png")

        # 8. Mobile stats, for layout sanity
        prepare(mpage, BASE + "?page=stats", "The numbers,")
        snap(mpage, "08_stats_mobile.png")

        browser.close()

    if errors:
        print("\nBrowser errors:")
        for error in errors:
            print(" -", error)
        return 1
    print("\nNo browser console/page errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
