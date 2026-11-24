import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isl_recognition.dataset import discover_images, load_group_map


class DatasetFolderTests(unittest.TestCase):
    def test_discovers_class_directories_and_nested_train_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "A").mkdir()
            (root / "A" / "a1.jpg").touch()
            (root / "train" / "B").mkdir(parents=True)
            (root / "train" / "B" / "b1.png").touch()
            (root / "test" / "B").mkdir(parents=True)
            (root / "test" / "B" / "b2.png").touch()

            samples = discover_images(root)
            self.assertEqual({sample.label for sample in samples}, {"A", "B"})
            self.assertEqual(len(samples), 3)

    def test_max_per_class_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "A").mkdir()
            for index in range(12):
                (root / "A" / f"{index:02}.jpg").touch()
            first = discover_images(root, max_per_class=4, seed=3)
            second = discover_images(root, max_per_class=4, seed=3)
            self.assertEqual([s.relative_path for s in first], [s.relative_path for s in second])
            self.assertEqual(len(first), 4)

    def test_group_csv_maps_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "A").mkdir()
            (root / "A" / "one.jpg").touch()
            samples = discover_images(root)
            groups_csv = root / "groups.csv"
            groups_csv.write_text("path,group\nA/one.jpg,signer-1\n", encoding="utf-8")
            groups = load_group_map(groups_csv, samples)
            self.assertEqual(groups.tolist(), ["signer-1"])


if __name__ == "__main__":
    unittest.main()
