import unittest
import numpy as np
from local_memory import VisualMemory
from train_memory_appearance import learn_weights


def observation(x,raw_id=1,alliance='blue',vector=None):
    return dict(box=[x,20,x+20,40],confidence=.9,raw_id=raw_id,
                alliance=alliance,appearance=np.array([1.,0.] if vector is None else vector))


class MemoryTests(unittest.TestCase):
    def test_identity_survives_a_missing_frame_and_raw_id_change(self):
        memory=VisualMemory();first=memory.update([observation(0)],0)[0][0]['id']
        memory.update([observation(2)],.1);memory.update([], .2)
        later=memory.update([observation(6,raw_id=99)],.3)[0][0]['id']
        self.assertEqual(first,later)

    def test_two_robots_cannot_share_one_state_in_one_frame(self):
        memory=VisualMemory();memory.update([observation(0),observation(80,2,vector=[0.,1.])],0)
        result=memory.update([observation(4),observation(76,2,vector=[0.,1.])],.1)
        self.assertEqual(len({state['id'] for state,_ in result}),2)

    def test_conflicting_appearance_does_not_reuse_id(self):
        memory=VisualMemory();first=memory.update([observation(0)],0)[0][0]['id']
        second=memory.update([observation(0,vector=[0.,1.])],.1)[0][0]['id']
        self.assertNotEqual(first,second)

    def test_identity_is_not_resurrected_after_expiration(self):
        memory=VisualMemory();first=memory.update([observation(0)],0)[0][0]['id']
        later=memory.update([observation(0)],3)[0][0]['id']
        self.assertNotEqual(first,later)

    def test_same_bumper_color_is_insufficient_after_longer_loss(self):
        memory=VisualMemory();first=memory.update([observation(0)],0)[0][0]['id']
        candidate=observation(0,raw_id=9,vector=[.84,.5426])
        later=memory.update([candidate],1)[0][0]['id']
        self.assertNotEqual(first,later)

    def test_hidden_state_is_not_a_detector_observation(self):
        memory=VisualMemory();memory.update([observation(0)],0)
        self.assertEqual(memory.update([], .2),[])
        self.assertEqual(len(memory.states),1)

    def test_known_later_anchor_preserves_early_outside_projection_observation(self):
        memory=VisualMemory(known_raw_ids={1});row=observation(0);row['field']=None
        self.assertEqual(len(memory.update([row],0)),1)
        untrusted=VisualMemory();self.assertEqual(untrusted.update([row],0),[])

    def test_metric_training_is_finite_with_constant_features(self):
        weights=learn_weights(np.ones((4,3)),np.array(['a','a','b','b']))
        self.assertTrue(np.isfinite(weights).all())


if __name__=='__main__':unittest.main()
