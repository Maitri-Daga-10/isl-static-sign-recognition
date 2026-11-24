import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isl_recognition.features import FEATURE_COUNT, FEATURES_PER_HAND, hands_to_features, normalize_hand


def make_hand(wrist_x: float = 0.5, wrist_y: float = 0.5, scale: float = 1.0) -> np.ndarray:
    points = []
    for index in range(21):
        x = wrist_x + scale * ((index % 5) * 0.08)
        y = wrist_y + scale * ((index // 5) * 0.09)
        z = scale * ((index % 3) * 0.01)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


class LandmarkFeatureTests(unittest.TestCase):
    def test_normalization_is_invariant_to_translation_and_uniform_size(self):
        hand = make_hand()
        moved_and_scaled = hand * 2.5 + np.asarray([0.17, 0.21, 0.03], dtype=np.float32)
        first, _ = normalize_hand(hand)
        second, _ = normalize_hand(moved_and_scaled)
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_one_hand_has_fixed_shape_and_presence_bit(self):
        vector = hands_to_features([make_hand()])
        self.assertEqual(vector.shape, (FEATURE_COUNT,))
        self.assertEqual(float(vector[2 * FEATURES_PER_HAND]), 1.0)
        self.assertEqual(float(vector[2 * FEATURES_PER_HAND + 1]), 0.0)
        self.assertEqual(float(vector[-1]), 0.0)

    def test_two_hands_keep_relative_wrist_offset(self):
        left = make_hand(wrist_x=0.3)
        right = make_hand(wrist_x=0.7)
        vector = hands_to_features([right, left])
        self.assertEqual(float(vector[2 * FEATURES_PER_HAND]), 1.0)
        self.assertEqual(float(vector[2 * FEATURES_PER_HAND + 1]), 1.0)
        self.assertGreater(float(vector[-4]), 0.0)
        self.assertEqual(float(vector[-1]), 1.0)

    def test_no_hands_returns_zero_vector(self):
        vector = hands_to_features([])
        self.assertEqual(vector.shape, (FEATURE_COUNT,))
        self.assertTrue(np.all(vector == 0))


if __name__ == "__main__":
    unittest.main()
