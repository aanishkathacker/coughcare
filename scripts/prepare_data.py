"""Prepare the balanced manifest without training, for inspection/tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coughcare.config import PROCESSED_DIR, configure_logging  # noqa: E402
from coughcare.train import prepare_manifest  # noqa: E402


def main() -> int:
    configure_logging()
    manifest, summary = prepare_manifest(rebuild=True)
    print(manifest.head(10).to_string(index=False))
    print("\nClass counts:", manifest["label"].value_counts().to_dict())
    print("Status counts:", manifest["status"].value_counts().to_dict())
    print("\nSelection summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nManifest: {PROCESSED_DIR / 'train_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
