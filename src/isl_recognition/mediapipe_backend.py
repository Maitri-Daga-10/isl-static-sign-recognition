"""MediaPipe hand detection, feature extraction, and OpenCV overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .features import FEATURE_COUNT, hands_to_features


@dataclass
class Detection:
    """One frame's detected landmarks, extracted vector, and MediaPipe result."""

    features: np.ndarray | None
    result: Any


class HandLandmarkDetector:
    """Reusable MediaPipe Hands instance for training or live webcam frames."""

    def __init__(
        self,
        *,
        static_image_mode: bool,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Activate the project environment and run "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_bgr(self, frame: np.ndarray) -> Detection:
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty image frame.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        raw_hands: list[np.ndarray] = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                raw_hands.append(
                    np.asarray(
                        [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
                        dtype=np.float32,
                    )
                )
        features = hands_to_features(raw_hands) if raw_hands else None
        if features is not None and features.shape != (FEATURE_COUNT,):
            raise RuntimeError("Internal feature shape mismatch.")
        return Detection(features=features, result=result)

    def draw_overlay(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """Draw MediaPipe skeletons and a box around each detected hand."""

        if not result.multi_hand_landmarks:
            return frame
        height, width = frame.shape[:2]
        for hand_landmarks in result.multi_hand_landmarks:
            self._mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp.solutions.hands.HAND_CONNECTIONS,
                self._mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                self._mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
            xs = [point.x for point in hand_landmarks.landmark]
            ys = [point.y for point in hand_landmarks.landmark]
            x1 = max(0, int(min(xs) * width) - 12)
            y1 = max(0, int(min(ys) * height) - 12)
            x2 = min(width - 1, int(max(xs) * width) + 12)
            y2 = min(height - 1, int(max(ys) * height) + 12)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 120), 2)
        return frame

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
