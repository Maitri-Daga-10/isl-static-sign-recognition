"""Stable, translation/scale-normalized features from up to two hands."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
MAX_HANDS = 2
PRESENCE_FEATURES = MAX_HANDS
PAIR_OFFSET_FEATURES = 3
PAIR_PRESENT_FEATURES = 1
FEATURE_COUNT = (
    MAX_HANDS * FEATURES_PER_HAND
    + PRESENCE_FEATURES
    + PAIR_OFFSET_FEATURES
    + PAIR_PRESENT_FEATURES
)
FEATURE_VERSION = "wrist_scale_two_hand_v1_132"


def normalize_hand(points: Sequence[Sequence[float]]) -> tuple[np.ndarray, float]:
    """Return flattened wrist-relative XYZ coordinates and the image-space scale.

    MediaPipe coordinates are image-relative. Removing the wrist position and
    dividing by the farthest XY landmark distance reduces translation and hand
    size differences while retaining hand orientation. The scale is returned
    separately so a two-hand sign can retain the distance between wrists.
    """

    coordinates = np.asarray(points, dtype=np.float32)
    expected = (LANDMARKS_PER_HAND, COORDS_PER_LANDMARK)
    if coordinates.shape != expected:
        raise ValueError(f"Expected hand landmarks with shape {expected}; got {coordinates.shape}.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Hand landmarks must contain only finite values.")

    relative = coordinates - coordinates[0]
    scale = float(np.linalg.norm(relative[:, :2], axis=1).max())
    if scale <= 1e-6:
        raise ValueError("Hand landmarks have no measurable size; cannot normalize them.")
    return (relative / scale).astype(np.float32).reshape(-1), scale


def hands_to_features(hands: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """Build one fixed-size feature vector from zero, one, or two detected hands.

    Hands are ordered by wrist X position in the source frame. Each hand is
    independently normalized; the normalized left-to-right wrist offset is
    appended when both hands are visible. A missing hand is represented by a
    zero block and an explicit presence bit.
    """

    if len(hands) > MAX_HANDS:
        hands = hands[:MAX_HANDS]

    parsed: list[tuple[np.ndarray, float, np.ndarray]] = []
    for hand in hands:
        points = np.asarray(hand, dtype=np.float32)
        normalized, scale = normalize_hand(points)
        parsed.append((normalized, scale, points[0]))
    parsed.sort(key=lambda item: float(item[2][0]))

    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    presence_start = MAX_HANDS * FEATURES_PER_HAND
    pair_offset_start = presence_start + PRESENCE_FEATURES
    pair_present_index = pair_offset_start + PAIR_OFFSET_FEATURES

    for index, (normalized, _scale, _wrist) in enumerate(parsed):
        start = index * FEATURES_PER_HAND
        features[start : start + FEATURES_PER_HAND] = normalized
        features[presence_start + index] = 1.0

    if len(parsed) == 2:
        left_wrist = parsed[0][2]
        right_wrist = parsed[1][2]
        pair_scale = max(parsed[0][1], parsed[1][1], 1e-6)
        features[pair_offset_start : pair_offset_start + 3] = (
            (right_wrist - left_wrist) / pair_scale
        )
        features[pair_present_index] = 1.0

    return features
