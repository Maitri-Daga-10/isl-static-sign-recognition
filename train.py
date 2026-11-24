"""Train and evaluate the static ISL landmark classifier from class folders."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isl_recognition.dataset import class_counts, discover_images, extract_features, load_group_map
from isl_recognition.features import FEATURE_COUNT, FEATURE_VERSION
from isl_recognition.mediapipe_backend import HandLandmarkDetector
from isl_recognition.model import build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "isl_alphabet",
                        help="Dataset root with one folder per sign label.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts",
                        help="Folder for the trained model, metrics, and confusion matrix.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Optional deterministic cap for a quick baseline (e.g. 100).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Held-out fraction for evaluation (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-detection-confidence", type=float, default=0.3,
                        help="MediaPipe minimum hand-detection confidence (default: 0.3).")
    parser.add_argument("--groups-csv", type=Path, default=None,
                        help="Optional CSV with dataset-relative path,group columns for signer/session-held-out evaluation.")
    return parser.parse_args()


def _split_indices(labels: np.ndarray, groups: np.ndarray | None, test_size: float, seed: int):
    indices = np.arange(len(labels))
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(indices, labels, groups))
        split_name = "group-held-out"
    else:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=labels
        )
        split_name = "stratified-image-split"
    train_labels = set(labels[train_idx])
    test_labels = set(labels[test_idx])
    all_labels = set(labels)
    if train_labels != all_labels or test_labels != all_labels:
        missing_train = sorted(all_labels - train_labels)
        missing_test = sorted(all_labels - test_labels)
        raise ValueError(
            "The selected split does not contain every class in both train and evaluation sets. "
            f"Missing from train: {missing_train}; missing from evaluation: {missing_test}. "
            "Add more examples/signers or adjust --test-size."
        )
    return train_idx, test_idx, split_name


def _save_confusion_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    width = min(max(8, len(labels) * 0.45), 24)
    height = min(max(6, len(labels) * 0.40), 22)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted label", ylabel="True label",
        title="Held-out confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> dict:
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.5.")
    if args.max_per_class is not None and args.max_per_class < 2:
        raise ValueError("--max-per-class must be at least 2.")
    if not 0.0 < args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence must be greater than 0 and at most 1.")

    samples = discover_images(args.data, max_per_class=args.max_per_class, seed=args.seed)
    preflight_counts = {}
    for sample in samples:
        preflight_counts[sample.label] = preflight_counts.get(sample.label, 0) + 1
    if len(preflight_counts) < 2:
        raise ValueError("At least two sign-label folders are required to train a classifier.")

    print(f"Dataset: {args.data.resolve()}", flush=True)
    print(f"Selected {len(samples)} images across {len(preflight_counts)} sign labels.", flush=True)
    with HandLandmarkDetector(
        static_image_mode=True,
        max_hands=2,
        min_detection_confidence=args.min_detection_confidence,
    ) as detector:
        dataset = extract_features(samples, detector)

    counts = class_counts(dataset.labels)
    too_small = {label: count for label, count in counts.items() if count < 2}
    if too_small:
        raise ValueError(f"Labels need at least two images with detected hands: {too_small}")
    print(f"Detected hands: {len(dataset.labels)}; no-hand images: {dataset.skipped_no_hand}; "
          f"unreadable images: {dataset.skipped_unreadable}.", flush=True)
    print("Detected samples by label:", flush=True)
    for label, count in counts.items():
        print(f"  {label}: {count}", flush=True)

    groups = load_group_map(args.groups_csv, samples) if args.groups_csv else None
    if groups is not None and len(groups) != len(dataset.labels):
        # No-hand and unreadable files are removed from features. Align groups to the retained paths.
        group_by_path = {sample.relative_path.casefold(): group for sample, group in zip(samples, groups)}
        groups = np.asarray([group_by_path[path.casefold()] for path in dataset.paths])

    train_idx, test_idx, split_name = _split_indices(
        dataset.labels, groups, args.test_size, args.seed
    )
    model = build_classifier(args.seed)
    model.fit(dataset.features[train_idx], dataset.labels[train_idx])
    predictions = model.predict(dataset.features[test_idx])
    target_labels = sorted(set(dataset.labels.tolist()))
    matrix = confusion_matrix(dataset.labels[test_idx], predictions, labels=target_labels)
    report = classification_report(
        dataset.labels[test_idx], predictions, labels=target_labels, output_dict=True,
        zero_division=0,
    )
    try:
        dataset_root_for_report = args.data.expanduser().resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        # Avoid embedding a machine-specific absolute path in a shareable report.
        dataset_root_for_report = args.data.expanduser().resolve().name

    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "static isolated ISL signs represented by the selected dataset labels",
        "dataset_root": dataset_root_for_report,
        "dataset_images_selected": len(samples),
        "max_per_class": args.max_per_class,
        "images_by_label_before_landmarks": dict(sorted(preflight_counts.items())),
        "detected_samples": int(len(dataset.labels)),
        "skipped_no_hand": dataset.skipped_no_hand,
        "skipped_unreadable": dataset.skipped_unreadable,
        "class_counts_after_detection": counts,
        "landmark_detection_rate_by_label": {
            label: round(counts.get(label, 0) / selected, 4)
            for label, selected in sorted(preflight_counts.items())
        },
        "classes": target_labels,
        "feature_version": FEATURE_VERSION,
        "feature_count": FEATURE_COUNT,
        "min_detection_confidence": args.min_detection_confidence,
        "split": split_name,
        "test_size": args.test_size,
        "train_examples": int(len(train_idx)),
        "evaluation_examples": int(len(test_idx)),
        "accuracy": float(accuracy_score(dataset.labels[test_idx], predictions)),
        "macro_f1": float(f1_score(dataset.labels[test_idx], predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(dataset.labels[test_idx], predictions, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_labels": target_labels,
        "confusion_matrix": matrix.tolist(),
        "limitations": [
            "A stratified image split can overestimate performance when images share a signer or recording session.",
            "This model recognizes still hand configurations only; it does not translate continuous ISL.",
            "Results apply only to labels and recording conditions represented in this dataset.",
        ],
    }
    if groups is None:
        metrics["evaluation_warning"] = (
            "No signer/session groups were provided; use a group-held-out evaluation before making generalization claims."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "isl_static_svc.joblib"
    metrics_path = args.output / "metrics.json"
    matrix_path = args.output / "confusion_matrix.png"
    joblib.dump(
        {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "feature_count": FEATURE_COUNT,
            "min_detection_confidence": args.min_detection_confidence,
            "classes": target_labels,
            "trained_utc": metrics["created_utc"],
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_confusion_matrix(matrix_path, target_labels, matrix)

    print("\nHeld-out evaluation")
    print(f"  split: {split_name}")
    print(f"  accuracy: {metrics['accuracy']:.3f}")
    print(f"  macro F1: {metrics['macro_f1']:.3f}")
    print(f"  weighted F1: {metrics['weighted_f1']:.3f}")
    if groups is None:
        print(f"  note: {metrics['evaluation_warning']}")
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {matrix_path}")
    return metrics


if __name__ == "__main__":
    try:
        train(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
