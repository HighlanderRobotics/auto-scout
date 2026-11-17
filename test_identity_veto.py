import unittest
from veto_identity_conflicts import veto


class IdentityVetoTests(unittest.TestCase):
    def test_read_is_current_crop_only_and_never_creates_identity(self):
        data=dict(source={'teams':{'3970':'blue','7137':'blue'}},frames=[dict(video_s=1,robots=[dict(team='3970',box=[0,0,10,10])]),dict(video_s=2,robots=[dict(team='3970',box=[0,0,10,10]),dict(team=None,box=[20,0,30,10])])])
        manifest={'a.jpg':dict(video_s=1,camera='main',box=[0,0,10,10]),'b.jpg':dict(video_s=2,camera='main',box=[20,0,30,10])}
        result,changes=veto(data,manifest,[dict(file='a.jpg',team='7137',confidence=1),dict(file='b.jpg',team='7137',confidence=1)])
        self.assertIsNone(result['frames'][0]['robots'][0]['team'])
        self.assertEqual(result['frames'][1]['robots'][0]['team'],'3970')
        self.assertIsNone(result['frames'][1]['robots'][1]['team'])
        self.assertEqual(len(changes),1)
        self.assertEqual(data['frames'][0]['robots'][0]['team'],'3970')


if __name__=='__main__':unittest.main()
