import unittest

from evaluate_detector import count_matches


def box(coords, cls=0):
    return {"box": coords, "class_id": cls}


class DetectionEvaluationTests(unittest.TestCase):
    def test_missing_and_extra_detections_are_counted(self):
        refs = [box([0, 0, 10, 10]), box([20, 20, 30, 30])]
        predictions = [box([0, 0, 10, 10]), box([50, 50, 60, 60])]
        self.assertEqual(count_matches(predictions, refs, .5),
                         {"true_positive": 1, "false_positive": 1, "false_negative": 1})

    def test_duplicates_cannot_match_one_robot_twice(self):
        ref = box([0, 0, 10, 10])
        self.assertEqual(count_matches([ref, ref], [ref], .5),
                         {"true_positive": 1, "false_positive": 1, "false_negative": 0})

    def test_matching_reassigns_to_maximize_robot_count(self):
        refs = [box([0, 0, 10, 10]), box([5, 0, 15, 10])]
        predictions = [box([0, 0, 15, 10]), box([0, 0, 10, 10])]
        self.assertEqual(count_matches(predictions, refs, .5)["true_positive"], 2)

    def test_empty_predictions_and_wrong_classes(self):
        ref = box([0, 0, 10, 10])
        self.assertEqual(count_matches([], [ref], .5)["false_negative"], 1)
        self.assertEqual(count_matches([box(ref["box"], 1)], [ref], .5)["true_positive"], 0)


if __name__ == "__main__":
    unittest.main()
