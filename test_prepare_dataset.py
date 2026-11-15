"""Offline checks for coordinates, split leakage, and export safety."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from prepare_dataset import FIELD_CROP, convert_robot, export_dataset, split_frames


class DatasetTests(unittest.TestCase):
    def test_crop_and_normalization(self):
        robot = {"visible": True, "team": "868", "box": [-10, 100, 1920, 720]}
        box, reason = convert_robot(robot, FIELD_CROP, "alliance")
        self.assertIsNone(reason)
        self.assertEqual(box["class_id"], 1)
        self.assertEqual(box["xywh"], [.5, .5, 1, 1])
        box, _ = convert_robot({**robot, "box": [0, 110, 192, 171]}, FIELD_CROP, "robot")
        for actual, expected in zip(box["xywh"], [.05, .05, .1, .1]):
            self.assertAlmostEqual(actual, expected)

    def test_hidden_outside_and_invalid_boxes(self):
        self.assertEqual(convert_robot({"visible": False}, FIELD_CROP, "robot"), (None, "hidden"))
        self.assertEqual(convert_robot({"visible": True, "box": [0, 0, 10, 10]}, FIELD_CROP, "robot"),
                         (None, "outside_crop"))
        for box in ([1, 1, 0, 10], [1, float("nan"), 10, 10], [0, 1, 2]):
            with self.assertRaises(ValueError):
                convert_robot({"visible": True, "box": box}, FIELD_CROP, "robot")

    def test_chronological_split_with_gap(self):
        files = [Path(f"f_{i:04}.json") for i in range(123)]
        splits = split_frames(files, 10)
        groups = {s: [i for i, f in enumerate(files) if splits[f.stem] == s]
                  for s in ("train", "val", "test", "excluded")}
        self.assertEqual(len(groups["excluded"]), 20)
        self.assertEqual(min(groups["val"]) - max(groups["train"]), 11)
        self.assertEqual(min(groups["test"]) - max(groups["val"]), 11)
        with self.assertRaises(ValueError):
            split_frames(files[:10], 10)

    def test_export_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames, labels, output = root / "frames", root / "labels", root / "dataset"
            frames.mkdir()
            labels.mkdir()
            for i in range(20):
                stem = f"f_{i:04}"
                Image.new("RGB", (1920, 1080), "gray").save(frames / (stem + ".jpg"))
                (labels / (stem + ".json")).write_text(json.dumps({"frame": stem + ".jpg", "robots": [
                    {"team": "868", "visible": True, "box": [0, 110, 192, 171]},
                    {"team": "2910", "visible": False, "box": [1, 1, 20, 20]}]}))
            manifest = export_dataset(frames, labels, output, gap=2)
            self.assertEqual(sum(s["images"] for s in manifest["summary"].values()), 16)
            with Image.open(output / "images/train/f_0000.jpg") as image:
                self.assertEqual(image.size, (1920, 610))
            values = (output / "labels/train/f_0000.txt").read_text().split()
            self.assertEqual(len(values), 5)
            self.assertEqual([float(v) for v in values[1:]], [.05, .05, .1, .1])
            for split in ("train", "val", "test"):
                self.assertTrue((output / f"preview_{split}.jpg").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_dataset(frames, labels, output, gap=2)
            (frames / "f_0019.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "Missing image"):
                export_dataset(frames, labels, root / "missing", gap=2)
            self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
