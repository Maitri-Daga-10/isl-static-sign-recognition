import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isl_recognition.model import build_classifier


class ClassifierTests(unittest.TestCase):
    def test_scaled_linear_svc_fits_and_predicts_separable_landmark_vectors(self):
        rng = np.random.default_rng(7)
        class_a = rng.normal(loc=-1.0, scale=0.1, size=(12, 132)).astype(np.float32)
        class_b = rng.normal(loc=1.0, scale=0.1, size=(12, 132)).astype(np.float32)
        features = np.vstack([class_a, class_b])
        labels = np.asarray(["A"] * len(class_a) + ["B"] * len(class_b))
        model = build_classifier(seed=7)
        model.fit(features, labels)
        np.testing.assert_array_equal(model.predict(features), labels)


if __name__ == "__main__":
    unittest.main()
