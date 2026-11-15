import unittest

from label_landmarks import build_meta, sample_times
from pathlib import Path


class LandmarkLabelerTests(unittest.TestCase):
    def test_sampling_stays_in_window(self):
        self.assertEqual(sample_times(5.5, 5, 2), [5.5, 7.5, 9.5])

    def test_invalid_sampling_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_times(0, 0, 1)

    def test_plan_has_floor_landmarks_and_frames(self):
        meta = build_meta(Path("match.mp4"), 1920, 1080, [1, 3])
        self.assertEqual(meta["image_size"], [1920, 1080])
        self.assertEqual(len(meta["landmarks"]), 8)
        self.assertEqual(meta["frames"][1]["image"], "frames/frame_0001.jpg")
