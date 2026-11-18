import unittest
import json,tempfile,contextlib,io
from pathlib import Path
from types import SimpleNamespace
from hybrid_shooting import parse_counter,score_increments
from shooting_review import reconcile

class ShootingTests(unittest.TestCase):
    def test_missing_score_context_stops_before_any_paid_client(self):
        from hybrid_shooting import teach
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifest.json').write_text(json.dumps({'examples':[{'anchor_s':12}]}));(root/'scoreboard.json').write_text(json.dumps({'rows':[{'video_s':43},{'video_s':142}]}))
            with self.assertRaisesRegex(ValueError,'before paid teaching'):teach(SimpleNamespace(out=root))

    def test_human_corrections_train_action_and_count_heads(self):
        from hybrid_shooting import train
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'labels').mkdir();examples=[];events=[]
            for i in range(6):
                eid=f'e_{i}';examples.append(dict(id=eid,start_s=i,end_s=i+1,features=[float(i),float(i%2)],split='train'))
                (root/'labels'/f'{eid}.json').write_text(json.dumps(dict(action='idle',confidence=.9,review_status='unreviewed_teacher')))
                events.append(dict(id=eid,start_s=i,end_s=i+1,action='shooting' if i%2 else 'idle',attempts=i if i%2 else 0,reviewed=True))
            (root/'manifest.json').write_text(json.dumps(dict(source_video_sha256='synthetic-fixture',config={'video':'unused.mp4'},examples=examples)))
            reviews=root/'reviews.json';reviews.write_text(json.dumps(dict(source_video_sha256='synthetic-fixture',events=events)))
            output=root/'model'
            with contextlib.redirect_stdout(io.StringIO()):train(SimpleNamespace(out=root,reviews=reviews,model_out=output))
            report=json.loads((output/'training_result.json').read_text())
            self.assertEqual(report['human_training_examples'],6)
            self.assertEqual(report['training_shooting_positive_examples'],3)
            self.assertTrue((output/'attempt_rate_model.npz').exists())

    def test_total_points_and_bad_ocr_are_not_fuel(self):
        self.assertIsNone(parse_counter([dict(text='575',confidence=1)]))
        self.assertEqual(parse_counter([dict(text='55 / 240',confidence=1)]),[55,240])
        self.assertIsNone(parse_counter([dict(text='55/240',confidence=1),dict(text='65/240',confidence=1)]))

    def test_gaps_and_negative_counter_changes_are_not_fabricated(self):
        rows=[dict(video_s=t,blue=v,red=None) for t,v in [(1,[10,240]),(2,None),(3,[20,240]),(4,[19,240]),(5,[22,240]),(6,[24,240])]]
        increments=score_increments(rows)
        self.assertEqual([(r['start_s'],r['fuel_delta']) for r in increments],[(5,2)])

    def test_overlap_is_unassigned_and_no_default_totals_are_invented(self):
        inc=[dict(start_s=12,end_s=13,alliance='red',fuel_delta=5)]
        teams={'9470':'red','5104':'red'}
        one=dict(id='a',team='9470',start_s=11,end_s=13,action='shooting',reviewed=True,counters_reviewed=True,exclusive_source=True)
        self.assertEqual(reconcile(inc,[],teams)['unassigned_observed_fuel']['red'],5)
        assigned=reconcile(inc,[one],teams);self.assertEqual(assigned['attributed_fuel_subtotals']['9470'],5)
        self.assertIsNone(assigned['per_robot_total_made']['9470'])
        two=dict(one,id='b',team='5104');shared=reconcile(inc,[one,two],teams)
        self.assertEqual(shared['unassigned_observed_fuel']['red'],5)
        self.assertEqual(sum(shared['attributed_fuel_subtotals'].values()),0)
        conflict=reconcile(inc,[dict(one,attempts=3)],teams)
        self.assertEqual(conflict['unassigned_observed_fuel']['red'],5)

if __name__=='__main__':unittest.main()
