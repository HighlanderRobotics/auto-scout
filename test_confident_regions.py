import unittest

from prepare_confident_regions import overlap, select_region


def robot(box, confidence=.8, visible=True):
    return {"box": box, "confidence": confidence, "visible": visible}


class ConfidentRegionTests(unittest.TestCase):
    def test_uncertain_visible_and_hidden_estimates_stay_outside_crop(self):
        for visible in (True, False):
            data = {"usable": True, "robots": [robot([100, 100, 200, 200]),
                    robot([1000, 100, 1500, 200], confidence=.2, visible=visible)]}
            region = select_region(data)
            self.assertIsNotNone(region)
            self.assertEqual(len(region["boxes"]), 1)
            self.assertEqual(overlap(data["robots"][1]["box"], region["crop"]), 0)

    def test_uncertain_interleaved_robot_prevents_using_incomplete_region(self):
        data = {"usable": True, "robots": [robot([900, 100, 1000, 200]),
                robot([800, 0, 1100, 610], confidence=.2)]}
        self.assertIsNone(select_region(data))

    def test_fully_confident_frame_keeps_all_boxes(self):
        data = {"usable": True, "robots": [robot([50, 50, 100, 100]), robot([1700, 50, 1800, 100])]}
        selected = select_region(data)
        self.assertEqual(selected["crop"], [0, 0, 1920, 610])
        self.assertEqual(len(selected["boxes"]), 2)

    def test_duplicate_boxes_and_non_finite_estimates_are_rejected(self):
        self.assertIsNone(select_region({"usable": True, "robots": [robot([100, 100, 200, 200]), robot([100, 100, 200, 200])]}))
        self.assertIsNone(select_region({"usable": True, "robots": [robot([100, 100, 200, 200]), robot([float("nan"), 0, 10, 10], visible=False)]}))

    def test_low_confidence_is_not_promoted_to_positive(self):
        self.assertIsNone(select_region({"usable": True, "robots": [robot([100, 100, 200, 200], .49)]}))


if __name__ == "__main__":
    unittest.main()
