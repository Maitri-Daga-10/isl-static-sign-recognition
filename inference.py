"""Live webcam inference with a bounding box, hand skeleton, and stable output."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isl_recognition.features import FEATURE_COUNT, FEATURE_VERSION
from isl_recognition.mediapipe_backend import HandLandmarkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts" / "isl_static_svc.joblib")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index (default: 0).")
    parser.add_argument("--smooth-frames", type=int, default=7,
                        help="Majority-vote window for stable display (default: 7).")
    parser.add_argument("--min-votes", type=int, default=4,
                        help="Minimum matching predictions in the window before showing a sign.")
    parser.add_argument("--min-detection-confidence", type=float, default=None,
                        help="Override the MediaPipe threshold saved with the model.")
    parser.add_argument("--show-margin", action="store_true",
                        help="Show the SVC decision margin; it is not a calibrated probability.")
    parser.add_argument("--flip-input", action="store_true",
                        help="Horizontally flip frames before detection if the training images are mirrored.")
    return parser.parse_args()


def _load_bundle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}\nTrain it first with: python train.py --data <dataset-folder>"
        )
    bundle = joblib.load(path)
    if bundle.get("feature_version") != FEATURE_VERSION or bundle.get("feature_count") != FEATURE_COUNT:
        raise ValueError("This model uses a different landmark schema. Retrain it with this project version.")
    return bundle


def run(args: argparse.Namespace) -> None:
    if args.smooth_frames < 1 or not 1 <= args.min_votes <= args.smooth_frames:
        raise ValueError("Use --smooth-frames >= 1 and --min-votes between 1 and smooth-frames.")
    if args.min_detection_confidence is not None and not 0.0 < args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence must be greater than 0 and at most 1.")
    bundle = _load_bundle(args.model)
    model = bundle["model"]
    detection_confidence = (
        args.min_detection_confidence
        if args.min_detection_confidence is not None
        else bundle.get("min_detection_confidence", 0.5)
    )
    predictions: deque[str] = deque(maxlen=args.smooth_frames)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Close other camera apps or try --camera 1."
        )

    print("Camera running. Keep one static sign steady; press Q or Esc to stop.")
    try:
        with HandLandmarkDetector(
            static_image_mode=False,
            max_hands=2,
            min_detection_confidence=detection_confidence,
        ) as detector:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("The webcam stopped returning frames.")
                if args.flip_input:
                    frame = cv2.flip(frame, 1)

                detection = detector.process_bgr(frame)
                display = detector.draw_overlay(frame.copy(), detection.result)
                if detection.features is None:
                    predictions.clear()
                    label_text = "No hand detected"
                    margin_text = ""
                else:
                    vector = detection.features.reshape(1, FEATURE_COUNT)
                    predicted = str(model.predict(vector)[0])
                    predictions.append(predicted)
                    vote, vote_count = Counter(predictions).most_common(1)[0]
                    label_text = vote if vote_count >= args.min_votes else "Hold sign steady..."
                    margin_text = ""
                    if args.show_margin and hasattr(model, "decision_function"):
                        scores = np.asarray(model.decision_function(vector)).reshape(-1)
                        margin_text = f"SVC margin: {float(scores.max()):.2f} (not probability)"

                cv2.rectangle(display, (0, 0), (display.shape[1], 86 if margin_text else 58), (20, 25, 32), -1)
                cv2.putText(display, f"ISL static sign: {label_text}", (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2, cv2.LINE_AA)
                if margin_text:
                    cv2.putText(display, margin_text, (16, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (190, 205, 220), 1, cv2.LINE_AA)
                cv2.putText(display, "Q / Esc: quit", (16, display.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.imshow("Phase 1 - ISL Static Sign Recognition", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
