"""Offline checks for paid-run limits, cached/uncertain requests and label filtering."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from temporal_multicamera import Budget, HAIKU, OPUS, accepted_boxes, schema_for, training_region


class MultiCameraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = MagicMock()
        usage = MagicMock()
        usage.model_dump.return_value = {"input_tokens":1000,"output_tokens":100}
        self.client.messages.create.return_value = SimpleNamespace(model=HAIKU, usage=usage,
            content=[SimpleNamespace(type="text",text='{}')],stop_reason="end_turn",id="offline")
        self.budget = Budget(self.root, "test", self.client)

    def tearDown(self):
        self.budget.lock.close()
        self.temp.cleanup()

    def call(self, model=HAIKU, name="first"):
        return self.budget.request(model, [], {}, self.root/(name+'.json'), name)

    def test_usage_and_cache_are_shared_without_duplicate_charge(self):
        self.call()
        self.call()
        self.assertEqual(self.client.messages.create.call_count,1)
        self.assertAlmostEqual(self.budget.data['spent_usd'],.00015)
        self.assertAlmostEqual(self.budget.data['reserved_usd'],0)

    def test_total_cap_reserves_before_call(self):
        self.budget.data['spent_usd']=4.895
        self.assertIsNone(self.call())
        self.client.messages.create.assert_not_called()

    def test_opus_cannot_use_more_than_its_subbudget(self):
        self.budget.data['attempts']=[dict(model=OPUS,status='billed',cost_usd=1.4,fingerprint='old')]
        self.assertIsNone(self.call(OPUS))
        self.client.messages.create.assert_not_called()

    def test_haiku_leaves_reserved_expert_allocation(self):
        self.budget.data['attempts']=[dict(model=HAIKU,status='billed',cost_usd=3.395,fingerprint='old')]
        self.assertIsNone(self.call())

    def test_uncertain_failure_is_not_retried_after_restart(self):
        self.client.messages.create.side_effect=TimeoutError('offline')
        with self.assertRaisesRegex(RuntimeError,'reservation retained'):
            self.call()
        self.assertAlmostEqual(self.budget.data['reserved_usd'],.01)
        self.budget.lock.close()
        self.budget=Budget(self.root,'test',self.client)
        with self.assertRaisesRegex(RuntimeError,'no paid retry'):
            self.call()
        self.assertEqual(self.client.messages.create.call_count,1)

    def test_cached_context_change_is_rejected(self):
        self.call()
        with self.assertRaisesRegex(ValueError,'context changed'):
            self.budget.request(HAIKU,[],{},self.root/'first.json','different')

    def test_hidden_estimates_never_enter_detector_labels(self):
        data={'usable':True,'complete':True,'robots':[
            dict(visible=True,box=[10,10,30,30],box_confidence=.9),
            dict(visible=False,box=[50,50,90,90],box_confidence=1.)]}
        self.assertEqual(accepted_boxes(data),[[10,10,30,30]])
        data['robots'][0]['box_confidence']=.7
        self.assertIsNone(accepted_boxes(data))
        data['robots'][0]['box_confidence']=.9
        data['complete']=False
        self.assertIsNone(accepted_boxes(data))

    def test_match_split_and_schema(self):
        plan=json.loads(Path('configs/temporal_multicamera_v1.json').read_text())
        train={s['match'] for s in plan['sources'] if s['split']=='train'}
        val={s['match'] for s in plan['sources'] if s['split']=='val'}
        self.assertFalse(train & val)
        self.assertNotIn('2026cmptx_f1m1',train|val)
        self.assertEqual(schema_for({'123':'red'})['properties']['robots']['items']['properties']['team']['enum'],['123'])

    def test_confident_regions_exclude_uncertain_estimates(self):
        data={'usable':True,'complete':False,'record':{'crop':[0,0,650,275]},'robots':[
            dict(visible=True,box=[250,90,360,160],box_confidence=.75),
            dict(visible=True,box=[25,60,128,125],box_confidence=.3),
            dict(visible=False,box=[500,80,550,160],box_confidence=.9)]}
        selected=training_region(data)
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(selected['crop'][0],128)
        self.assertLessEqual(selected['crop'][2],500)
        self.assertEqual(selected['boxes'],[[250,90,360,160]])

    def test_hidden_estimate_coincident_with_positive_remains_uncertain(self):
        data={'usable':True,'complete':True,'record':{'crop':[0,0,650,275]},'robots':[
            dict(visible=True,box=[250,90,360,160],box_confidence=.9),
            dict(visible=False,box=[250,90,360,160],box_confidence=.9)]}
        self.assertIsNone(training_region(data))


if __name__=='__main__':
    unittest.main()
