import unittest
from path_stitching import stitch_fragments,fill_short_gaps


def fragment(identifier,times,x):
    return {'id':identifier,'points':[{'t':t,'x':x,'y':10} for t in times]}


class StitchTests(unittest.TestCase):
    def test_backwards_propagation_through_unnamed_chain(self):
        tracks=[fragment(1,[0,.1],10),fragment(2,[.3,.4],10),fragment(3,[.6,.7],10)]
        chains,_=stitch_fragments(tracks,{1:'red',2:'red',3:'red'},{3:'254'})
        self.assertIn([1,2,3],chains)

    def test_ambiguous_successors_stay_separate(self):
        tracks=[fragment(1,[0,.1],10),fragment(2,[.3,.4],10),fragment(3,[.3,.4],11)]
        chains,_=stitch_fragments(tracks,{1:'red',2:'red',3:'red'},{1:'254'})
        self.assertEqual(len(chains),3)

    def test_conflicting_identity_cannot_bridge_through_unknown(self):
        tracks=[fragment(1,[0,.1],10),fragment(2,[.3,.4],10),fragment(3,[.6,.7],10)]
        chains,_=stitch_fragments(tracks,{1:'red',2:'red',3:'red'},{1:'254',3:'971'})
        self.assertFalse(any(1 in c and 3 in c for c in chains))

    def test_opposite_color_and_quarantined_tracks_cannot_join(self):
        tracks=[fragment(1,[0,.1],10),fragment(2,[.3,.4],10)]
        self.assertEqual(len(stitch_fragments(tracks,{1:'blue',2:'red'},{})[0]),2)
        self.assertEqual(len(stitch_fragments(tracks,{1:'red',2:'red'},{},blocked={2})[0]),2)

    def test_estimation_is_bounded_and_preserves_original_points(self):
        points=[{'t':0,'x':10,'y':10,'source_id':1},{'t':1,'x':20,'y':10,'source_id':2},{'t':5,'x':30,'y':10,'source_id':3}]
        out=fill_short_gaps(points,10)
        self.assertEqual(out[0],points[0]);self.assertEqual(out[-1],points[-1])
        self.assertEqual(sum(bool(p.get('estimated')) for p in out),9)
        self.assertFalse(any(1<p['t']<5 for p in out))


if __name__=='__main__':unittest.main()
