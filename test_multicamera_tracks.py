import unittest
from multicamera_tracks import resolve_identities, track_key


class MultiCameraIdentityTests(unittest.TestCase):
    def frame(self,t,cameras):
        return dict(video_s=t,cameras={key:[dict(id=i) for i in ids] for key,ids in cameras.items()})

    def reads(self,key,team):
        return [dict(key=key,team=team,video_s=t,confidence=.9) for t in [0.,.4]]

    def test_same_numeric_id_in_different_cameras_is_not_same_track(self):
        self.assertNotEqual(track_key('main',1),track_key('side',1))
        frames=[self.frame(t,{'main':[1],'side':[1]}) for t in [0.,.4]]
        assignments,_=resolve_identities(frames,self.reads('main:1','123'),{'123'})
        self.assertEqual(assignments,{'main:1':'123'})

    def test_same_team_in_two_views_is_valid(self):
        frames=[self.frame(t,{'main':[1],'side':[1]}) for t in [0.,.4]]
        assignments,_=resolve_identities(frames,self.reads('main:1','123')+self.reads('side:1','123'),{'123'})
        self.assertEqual(len(assignments),2)

    def test_two_simultaneous_same_camera_tracks_cannot_share_team(self):
        frames=[self.frame(t,{'main':[1,2]}) for t in [0.,.4]]
        assignments,audit=resolve_identities(frames,self.reads('main:1','123')+self.reads('main:2','123'),{'123'})
        self.assertEqual(assignments,{})
        self.assertTrue(audit)

    def test_conflicting_or_single_read_does_not_force_name(self):
        frames=[self.frame(t,{'main':[1]}) for t in [0.,.4]]
        reads=self.reads('main:1','123')+self.reads('main:1','456')
        self.assertEqual(resolve_identities(frames,reads,{'123','456'})[0],{})
        self.assertEqual(resolve_identities(frames,reads[:1],{'123'})[0],{})


if __name__=='__main__':unittest.main()
