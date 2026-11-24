"""Download and safely extract the selected public Kaggle ISL character dataset."""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
ARCHIVE_PATH = DATA_ROOT / "raw" / "prathumarikeri_indian_sign_language_isl_v1.zip"
DESTINATION = DATA_ROOT / "isl_alphabet"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/prathumarikeri/indian-sign-language-isl"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHUNK_SIZE = 1024 * 1024


def download() -> None:
    if ARCHIVE_PATH.exists():
        raise FileExistsError(f"Archive already exists; move it manually if you want a fresh download: {ARCHIVE_PATH}")
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "Phase1-ISL-Student-Project/0.1"},
    )
    temporary = ARCHIVE_PATH.with_suffix(".zip.part")
    total_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            if response.headers.get_content_type() != "application/zip":
                sample = response.read(300).decode("utf-8", errors="replace")
                raise RuntimeError(f"Kaggle did not return a ZIP archive. Response starts with: {sample!r}")
            expected = int(response.headers.get("Content-Length") or 0)
            print(f"Downloading public Kaggle archive ({expected / (1024 * 1024):.1f} MB)...")
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                output.write(block)
                total_bytes += len(block)
                if total_bytes // (50 * CHUNK_SIZE) != (total_bytes - len(block)) // (50 * CHUNK_SIZE):
                    print(f"  {total_bytes / (1024 * 1024):.0f} MB downloaded")
        if total_bytes < 1024:
            raise RuntimeError("The Kaggle download was unexpectedly small.")
        temporary.replace(ARCHIVE_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def safe_extract_images() -> dict[str, int]:
    if DESTINATION.exists() and any(DESTINATION.rglob("*")):
        raise FileExistsError(f"Destination already contains files; no files were overwritten: {DESTINATION}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    destination_resolved = DESTINATION.resolve()
    counts: Counter[str] = Counter()
    try:
        with zipfile.ZipFile(ARCHIVE_PATH) as archive:
            members = archive.infolist()
            image_members = [member for member in members if PurePosixPath(member.filename).suffix.casefold() in IMAGE_SUFFIXES]
            if len(image_members) < 100:
                raise RuntimeError(f"Expected a labeled image dataset, found only {len(image_members)} image files.")
            for member in image_members:
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or any(part in {"..", "."} for part in relative.parts):
                    raise RuntimeError(f"Unsafe archive path rejected: {member.filename!r}")
                target = DESTINATION.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(destination_resolved):
                    raise RuntimeError(f"Archive path escaped the dataset folder: {member.filename!r}")
                if len(relative.parts) < 2:
                    raise RuntimeError(f"Image has no class folder: {member.filename!r}")
                # In this Kaggle archive, the folder immediately above each image is the label.
                label = relative.parts[-2]
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    while True:
                        block = source.read(CHUNK_SIZE)
                        if not block:
                            break
                        output.write(block)
                counts[label] += 1
    except Exception:
        # Keep any partial extraction visible for inspection; do not delete user data.
        raise
    if len(counts) < 2:
        raise RuntimeError(f"Expected multiple sign-label folders; found {sorted(counts)}")
    return dict(sorted(counts.items()))


def main() -> int:
    try:
        if not ARCHIVE_PATH.exists():
            download()
        counts = safe_extract_images()
        print(f"\nExtracted {sum(counts.values())} images into {DESTINATION}")
        print(f"Found {len(counts)} class folders:")
        for label, count in counts.items():
            print(f"  {label}: {count}")
        print("\nDataset: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl")
        print("License shown on Kaggle: CC BY-SA 4.0. Review the page and upstream credits before redistribution.")
        return 0
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
