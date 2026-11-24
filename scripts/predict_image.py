"""Run one offline prediction and save the hand skeleton/bounding-box overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from inference import _load_bundle
from isl_recognition.features import FEATURE_COUNT
from isl_recognition.mediapipe_backend import HandLandmarkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Image file to classify.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts" / "isl_static_svc.joblib")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "preview.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {args.image}")
    bundle = _load_bundle(args.model)
    with HandLandmarkDetector(
        static_image_mode=True,
        max_hands=2,
        min_detection_confidence=bundle.get("min_detection_confidence", 0.5),
    ) as detector:
        detection = detector.process_bgr(image)
        if detection.features is None:
            print("No hand was detected; no class prediction was made.")
            return 1
        vector = detection.features.reshape(1, FEATURE_COUNT)
        predicted = str(bundle["model"].predict(vector)[0])
        display = detector.draw_overlay(image.copy(), detection.result)

    font_scale = min(0.75, max(0.34, display.shape[1] / 800))
    bar_height = max(28, int(display.shape[0] * 0.22))
    cv2.rectangle(display, (0, 0), (display.shape[1], bar_height), (20, 25, 32), -1)
    cv2.putText(display, f"ISL: {predicted}", (8, int(bar_height * 0.73)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245),
                max(1, int(font_scale * 2)), cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), display):
        raise OSError(f"Could not save preview image: {args.output}")
    print(f"Predicted label: {predicted}")
    print(f"Saved overlay: {args.output.resolve()}")
    print("The input image's folder name is not used as a model feature.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
