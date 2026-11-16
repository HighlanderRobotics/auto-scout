"""Offline tests for the independently frozen side-camera evaluation policy."""
import json
import unittest
from pathlib import Path

from evaluate_multicamera import excluded_prediction


class SideCameraEvaluationTests(unittest.TestCase):
    def test_only_predictions_mostly_inside_ignore_are_excluded(self):
        ignored=[[10,10,30,30]]
        self.assertTrue(excluded_prediction([11,11,29,29],ignored,[]))
        self.assertFalse(excluded_prediction([0,0,100,100],ignored,[]))

    def test_scored_reference_cannot_be_hidden_by_ignore_region(self):
        ref=[dict(class_id=0,box=[10,10,30,30])]
        self.assertFalse(excluded_prediction([10,10,30,30],[[0,0,50,50]],ref))

    def test_temporal_holdout_never_overlaps_training(self):
        labels=json.loads(Path('configs/multicamera_holdout_v1.json').read_text())
        plan=json.loads(Path('configs/temporal_multicamera_v1.json').read_text())
        end=plan['start_s']+plan['duration_s']
        self.assertTrue(all(r['video_s']>=end+10 for r in labels['frames']))
        self.assertEqual(len(labels['frames']),12)
        self.assertEqual(sum(len(r['boxes']) for r in labels['frames']),16)


if __name__=='__main__':unittest.main()
