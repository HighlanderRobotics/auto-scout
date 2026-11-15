import json
import tempfile
import unittest
from pathlib import Path

from auto_paths import connected, field_frame, field_position, load_calibration, score_reference, youtube_id


class AutoPathTests(unittest.TestCase):
    def test_render_json_track_colors_and_observations(self):
        tracks=[{"id":1,"color":[255,100,0],"points":[{"t":0,"x":10,"y":10},{"t":.1,"x":12,"y":11}]}]
        image=field_frame(tracks,.1,20,True)
        self.assertEqual(image.size,(960,540))
        self.assertEqual(image.mode,"RGB")

    def test_youtube_urls_reject_unrelated_hosts(self):
        for url in ("https://youtu.be/EjF9we707DA", "https://www.youtube.com/watch?v=EjF9we707DA&t=5", "https://youtube.com/shorts/EjF9we707DA"):
            self.assertEqual(youtube_id(url), "EjF9we707DA")
        for url in ("https://youtube.com.example.org/watch?v=EjF9we707DA", "https://example.org/video", "https://youtube.com/watch?v=bad", "file:///tmp/a.mp4"):
            self.assertIsNone(youtube_id(url))

    def test_homography_maps_floor_contact_point_not_box_center(self):
        config = {"image_size": [200, 100], "crop": [0, 0, 200, 100],
                  "image_points": [[0, 0], [200, 0], [200, 100], [0, 100]],
                  "field_points": [[0, 0], [100, 0], [100, 100], [0, 100]]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera.json"
            path.write_text(json.dumps(config))
            _, matrix = load_calibration(path)
            self.assertEqual(field_position([80, 10, 120, 75], matrix), [50, 75])
            self.assertIsNone(field_position([80, 80, 120, 110], matrix))
            config["image_points"] = [[0, 0], [10, 0], [20, 0], [30, 0]]
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_calibration(path)

    def test_tracks_do_not_bridge_occlusions_or_large_jumps(self):
        a = {"t": 0, "x": 10, "y": 10}
        self.assertTrue(connected(a, {"t": .1, "x": 12, "y": 11}))
        self.assertFalse(connected(a, {"t": 2, "x": 12, "y": 11}))
        self.assertFalse(connected(a, {"t": .1, "x": 90, "y": 11}))
        self.assertFalse(connected(a, a))

    def test_reference_score_counts_missing_and_extra_boxes_at_matching_times(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "f_0000.json"
            p.write_text(json.dumps({"time_s": 5.5, "robots": [
                {"visible": True, "box": [0, 0, 10, 10]}, {"visible": True, "box": [20, 0, 30, 10]}]}))
            frames = [{"video_s": 5.5, "robots": [{"box": [0, 0, 10, 10]}, {"box": [40, 0, 50, 10]}]},
                      {"video_s": 5.6, "robots": []}]
            result = score_reference(frames, folder)
            self.assertEqual(result["frames_compared"], 1)
            self.assertEqual(result["0.5"]["true_positive"], 1)
            self.assertEqual(result["0.5"]["false_positive"], 1)
            self.assertEqual(result["0.5"]["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
