"""Discover and convert class-folder image datasets into hand landmarks."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "training", "valid", "validation", "val", "test", "testing"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str
    relative_path: str


@dataclass
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    paths: list[str]
    skipped_no_hand: int
    skipped_unreadable: int


class Detector(Protocol):
    def process_bgr(self, frame: np.ndarray): ...


def _label_from_parts(parts: tuple[str, ...]) -> str | None:
    directories = parts[:-1]
    for index, part in enumerate(directories):
        if part.casefold() in SPLIT_NAMES and index + 1 < len(directories):
            return directories[index + 1]
    if not directories:
        return None
    if directories[-1].casefold() in SPLIT_NAMES and len(directories) > 1:
        return directories[-2]
    return directories[-1]


def discover_images(root: str | Path, max_per_class: int | None = None, seed: int = 42) -> list[ImageSample]:
    """Read common class-folder layouts, including train/<class>/image layouts.

    The dataset stays outside this repository. Supported examples are
    ``root/A/image.jpg`` and ``root/train/A/image.jpg``. For best evaluation,
    provide a separate group CSV to train.py when signer/session IDs are known.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root_path}")

    by_label: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        label = _label_from_parts(path.relative_to(root_path).parts)
        if label:
            by_label[label].append(path)

    if not by_label:
        raise ValueError(
            "No labeled images found. Put images in one folder per sign, for example "
            "data/isl_alphabet/A/*.jpg and data/isl_alphabet/B/*.jpg."
        )

    rng = random.Random(seed)
    samples: list[ImageSample] = []
    for label in sorted(by_label):
        paths = by_label[label]
        if max_per_class is not None and len(paths) > max_per_class:
            paths = sorted(rng.sample(paths, max_per_class))
        for path in paths:
            samples.append(
                ImageSample(
                    path=path,
                    label=label,
                    relative_path=path.relative_to(root_path).as_posix(),
                )
            )
    return samples


def load_group_map(csv_path: str | Path, samples: list[ImageSample]) -> np.ndarray:
    """Load a CSV with exact dataset-relative ``path`` and signer/session ``group`` columns."""

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Group mapping CSV not found: {path}")
    group_by_path: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"path", "group"}.issubset(reader.fieldnames):
            raise ValueError("Group CSV must contain exactly named columns 'path' and 'group'.")
        for row in reader:
            key = Path(row["path"].replace("\\", "/")).as_posix().lstrip("./").casefold()
            group = row["group"].strip()
            if key and group:
                group_by_path[key] = group

    missing = [sample.relative_path for sample in samples if sample.relative_path.casefold() not in group_by_path]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"Group CSV is missing {len(missing)} image path(s), including: {preview}")
    return np.asarray([group_by_path[sample.relative_path.casefold()] for sample in samples])


def extract_features(
    samples: list[ImageSample],
    detector: Detector,
    *,
    progress_every: int = 200,
) -> FeatureDataset:
    """Extract hand landmarks from images and count unusable samples."""

    features: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    no_hand = 0
    unreadable = 0
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        image = cv2.imread(str(sample.path))
        if image is None:
            unreadable += 1
            continue
        detection = detector.process_bgr(image)
        if detection.features is None:
            no_hand += 1
        else:
            features.append(detection.features)
            labels.append(sample.label)
            paths.append(sample.relative_path)
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"Landmarks: {index}/{total} images processed", flush=True)

    if not features:
        raise ValueError(
            "No hand landmarks were detected. Check that the images show visible hands and that "
            "MediaPipe can read this image format."
        )
    return FeatureDataset(
        features=np.vstack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=str),
        paths=paths,
        skipped_no_hand=no_hand,
        skipped_unreadable=unreadable,
    )


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return dict(sorted(Counter(labels.tolist()).items()))
