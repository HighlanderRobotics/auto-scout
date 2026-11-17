import unittest
import numpy as np
from identity_student import decide,fit_head,check_training_row


class IdentityStudentTests(unittest.TestCase):
    def test_reviewed_evaluation_frame_cannot_enter_training(self):
        row=dict(match='2026cancmp_qm3',video_s=94,review_status='agent_reviewed_source_crop',label='3970',current_roster={'3970':'blue'},full_frame_box=[1,2,10,20])
        with self.assertRaisesRegex(ValueError,'outside allowed'):check_training_row(row,{'2026cancmp_qm3':[45,65]})
        row['video_s']=50;check_training_row(row,{'2026cancmp_qm3':[45,65]})

    def test_unknown_absent_and_duplicate_claims_stay_unresolved(self):
        classes=np.array(['3970','7137','unknown','9470'])
        scores=np.array([[.9,.1,0,0],[.85,.1,0,0],[0,0,.9,.1],[.7,0,0,.95],[.1,.9,0,0]])
        self.assertEqual(decide(scores,classes,{'3970':'blue','7137':'blue'},.5,.15),{4:'7137'})

    def test_training_head_can_separate_reviewed_features(self):
        x=np.array([[1.,0.],[.9,.1],[0,1.],[.1,.9]])
        weights,classes=fit_head(x,['3970','3970','unknown','unknown'],.1)
        predicted=classes[(np.column_stack([x,np.ones(4)])@weights).argmax(axis=1)]
        self.assertEqual(predicted.tolist(),['3970','3970','unknown','unknown'])


if __name__=='__main__':unittest.main()
