"""Script: download, verify and unpack the COUGHVID public dataset."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coughcare.config import (  # noqa: E402
    DATASET_DIR,
    METADATA_CSV,
    RAW_DIR,
    ZIP_PATH,
    configure_logging,
)

log = logging.getLogger("download")

ZENODO_URL = "https://zenodo.org/records/4048312/files/public_dataset.zip"
EXPECTED_MD5 = "5c30a8b00c8bb7783a2c15a48cb8ea9e"
EXPECTED_SIZE = 951_442_487


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - Zenodo publishes an MD5 checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0
    if existing == expected_size:
        log.info("Zip already downloaded (%s bytes), skipping download.", f"{existing:,}")
        return

    headers = {}
    mode = "wb"
    if 0 < existing < expected_size:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        log.info("Resuming download at %s bytes.", f"{existing:,}")
    else:
        existing = 0

    request = urllib.request.Request(url, headers=headers)
    log.info("Downloading %s", url)
    start = time.time()
    with urllib.request.urlopen(request, timeout=60) as response, target.open(mode) as out:
        total = expected_size
        copied = existing
        last_log = 0.0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            now = time.time()
            if now - last_log > 5:
                pct = 100 * copied / total
                speed = (copied - existing) / max(now - start, 1) / 1024 / 1024
                log.info("%5.1f%% | %s / %s MB | %.1f MB/s", pct, f"{copied/1e6:,.0f}", f"{total/1e6:,.0f}", speed)
                last_log = now
    log.info("Download finished in %.1fs", time.time() - start)


def verify(path: Path) -> None:
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"Size mismatch: expected {EXPECTED_SIZE:,}, got {size:,}")
    log.info("Size OK (%s bytes). Verifying MD5…", f"{size:,}")
    digest = md5sum(path)
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {EXPECTED_MD5}, got {digest}")
    log.info("MD5 OK: %s", digest)


def unzip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt member in zip: {bad}")
        members = archive.namelist()
        log.info("Zip integrity OK. %d members.", len(members))
        # The archive contains a top-level public_dataset/ folder. Extract
        # directly after stripping it so paths stay predictable.
        for index, member in enumerate(members, 1):
            parts = Path(member).parts
            relative = Path(*parts[1:]) if len(parts) > 1 and parts[0] == "public_dataset" else Path(member)
            if not relative.parts or relative.name == "":
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as out:
                out.write(source.read())
            if index % 2000 == 0:
                log.info("Extracted %d / %d", index, len(members))
    log.info("Extracted to %s", destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=ZENODO_URL)
    parser.add_argument("--zip", type=Path, default=ZIP_PATH)
    parser.add_argument("--dest", type=Path, default=DATASET_DIR)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    configure_logging()
    args.zip.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download(args.url, args.zip, EXPECTED_SIZE)
    if not args.zip.exists():
        log.error("Zip not found at %s", args.zip)
        return 1
    if not args.skip_verify:
        verify(args.zip)
    if not METADATA_CSV.exists() or not any(args.dest.glob("*.webm")):
        unzip(args.zip, args.dest)
    else:
        log.info("Dataset already extracted at %s", args.dest)

    if not METADATA_CSV.exists():
        log.error("metadata_compiled.csv missing after extraction.")
        return 1
    log.info("Ready: %s", METADATA_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
