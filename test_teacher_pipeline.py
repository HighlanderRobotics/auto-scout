"""Offline tests only: mocked Claude responses, budget accounting, dataset filters."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
from run_teacher_training import training_boxes
from teacher_labels import MODEL, Teacher, InvalidTeacherLabel, normalize_labels


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.temp.name)
        Path("track/clip_frames").mkdir(parents=True)
        Path("track/final").mkdir(parents=True)
        Image.new("RGB", (1920, 1080)).save("track/clip_frames/f_0000.jpg")
        Path("track/final/f_0000.json").write_text(json.dumps({"robots": []}))
        Image.new("RGB", (1920, 610)).save("frame.jpg")
        self.key_patch = patch("teacher_labels.load_key", return_value="offline-test-placeholder")
        self.key_patch.start()
        self.client = MagicMock()
        self.client_patch = patch("anthropic.Anthropic", return_value=self.client)
        self.client_patch.start()
        self.instances = []

    def tearDown(self):
        for teacher in self.instances:
            teacher.lock.close()
        self.client_patch.stop()
        self.key_patch.stop()
        os.chdir(self.previous)
        self.temp.cleanup()

    def teacher(self):
        instance = Teacher("billing")
        self.instances.append(instance)
        return instance

    def response(self, box=None):
        usage = MagicMock()
        usage.model_dump.return_value = {"input_tokens": 1000, "output_tokens": 100}
        return SimpleNamespace(model=MODEL, stop_reason="end_turn", id="offline-response",
                               usage=usage, content=[SimpleNamespace(type="text", text=json.dumps({
                                   "usable": True, "robots": [{"box": box or [10, 10, 100, 100], "confidence": .9}], "notes": ""}))])

    def test_usage_accounted_and_resume_does_not_call_again(self):
        teacher = self.teacher()
        self.client.messages.create.return_value = self.response()
        teacher.label("frame.jpg", "out.json")
        teacher.label("frame.jpg", "out.json")
        self.assertEqual(self.client.messages.create.call_count, 1)
        self.assertAlmostEqual(teacher.ledger["spent_usd"], .00015)
        self.assertAlmostEqual(teacher.ledger["reserved_usd"], 0)
        self.assertEqual(self.client.messages.create.call_args.kwargs["model"], MODEL)

    def test_budget_stops_before_request(self):
        teacher = self.teacher()
        teacher.ledger["spent_usd"] = 3.995
        with self.assertRaisesRegex(RuntimeError, "Budget exhausted"):
            teacher.label("frame.jpg", "out.json")
        self.client.messages.create.assert_not_called()

    def test_failed_request_keeps_reservation(self):
        teacher = self.teacher()
        self.client.messages.create.side_effect = RuntimeError("offline failure")
        with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
            teacher.label("frame.jpg", "out.json")
        self.assertAlmostEqual(teacher.ledger["reserved_usd"], .01)
        self.assertEqual(teacher.ledger["attempts"][0]["status"], "failed_or_uncertain")
        self.assertEqual(self.client.messages.create.call_count, 1)

    def test_invalid_label_is_billed_but_not_exported(self):
        teacher = self.teacher()
        self.client.messages.create.return_value = self.response([10, 10, 5, 5])
        with self.assertRaises(InvalidTeacherLabel):
            teacher.label("frame.jpg", "out.json")
        self.assertGreater(teacher.ledger["spent_usd"], 0)
        self.assertFalse(Path("out.json").exists())
        self.assertTrue(Path("raw/out.json").exists())
        self.assertEqual(self.client.messages.create.call_count, 1)

    def test_hidden_robot_with_empty_box_uses_previous_estimate(self):
        teacher = self.teacher()
        teams = {str(i): "blue" if i < 3 else "red" for i in range(6)}
        robots = [{"team": str(i), "visible": i != 0, "box": [] if i == 0 else [10, 10, 30, 30],
                   "confidence": .8, "named_by": "previous_frames", "bumper_text": ""} for i in range(6)]
        response = self.response()
        response.content[0].text = json.dumps({"robots": robots, "notes": ""})
        self.client.messages.create.return_value = response
        prior = [{"team": "0", "box": [100, 100, 200, 200]}]
        result = teacher.label("frame.jpg", "out.json", teams=teams, prior=prior)
        self.assertFalse(result["robots"][0]["visible"])
        self.assertEqual(result["robots"][0]["box"], prior[0]["box"])


class PseudoLabelTests(unittest.TestCase):
    def data(self):
        return {"usable": True, "robots": [{"team": str(i), "visible": True, "confidence": .9,
                                            "box": [i * 100, 20, i * 100 + 50, 70]} for i in range(6)]}

    def test_filters_incomplete_low_confidence_and_duplicate_frames(self):
        data = self.data()
        self.assertEqual(len(training_boxes(data)[0]), 6)
        data["robots"][0]["visible"] = False
        self.assertEqual(training_boxes(data)[1], "hidden_or_incomplete")
        data = self.data()
        data["robots"][0]["confidence"] = .3
        self.assertEqual(training_boxes(data)[1], "low_confidence")
        data = self.data()
        data["robots"][1]["box"] = data["robots"][0]["box"]
        self.assertEqual(training_boxes(data)[1], "duplicate_or_ambiguous")

    def test_plan_holds_out_matches_and_never_selects_sonnet(self):
        plan = json.loads(Path("configs/haiku_teacher_v2.json").read_text())
        self.assertEqual(plan["model"], MODEL)
        train = {s["match"] for s in plan["sources"] if s["split"] == "train"}
        val = {s["match"] for s in plan["sources"] if s["split"] == "val"}
        self.assertFalse(train & val)
        self.assertNotIn("2026cmptx_f1m1", train | val)

    def test_tiny_boundary_overshoot_is_clipped(self):
        data = {"usable": True, "robots": [{"box": [-1, 2, 101, 20], "confidence": .9}]}
        normalized = normalize_labels(data, None, None, 100, 50, 200, 100)
        self.assertEqual(normalized["robots"][0]["box"], [0, 4, 200, 40])
        self.assertEqual(len(normalized["box_adjustments"]), 1)

    def test_bad_geometry_is_quarantined_not_silently_corrected(self):
        for box in ([1000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 50, 500], [], [0, 0, float("nan"), 10]):
            with self.subTest(box=box), self.assertRaises(InvalidTeacherLabel):
                normalize_labels({"usable": True, "robots": [{"box": box, "confidence": .9}]}, None, None, 100, 50, 100, 50)

    def test_box_fully_outside_after_clipping_is_rejected(self):
        with self.assertRaises(InvalidTeacherLabel):
            normalize_labels({"usable": True, "robots": [{"box": [-2, 2, -1, 10], "confidence": .9}]}, None, None, 100, 50, 100, 50)


if __name__ == "__main__":
    unittest.main()
