import unittest
from team_paths import partial_team_matches, resolve_tracks


def track(identifier, times):
    return {'id':identifier,'points':[{'t':t,'x':t,'y':0} for t in times]}


def reads(identifier, team, times):
    return [{'id':identifier,'team':team,'t':t,'confidence':.95} for t in times]


class IdentityTests(unittest.TestCase):
    def test_repeated_reads_merge_nonoverlapping_fragments(self):
        tracks,audit=resolve_tracks({'tracks':[track(1,[0,1]),track(9,[2,3])]},reads(1,'254',[0,1])+reads(9,'254',[2,3]),{'254'})
        self.assertEqual(len(tracks),1)
        self.assertEqual(tracks[0]['source_ids'],[1,9])
        self.assertEqual(len(tracks[0]['points']),4)

    def test_single_read_and_duplicate_crop_do_not_establish_identity(self):
        evidence=reads(1,'254',[0,0])
        self.assertEqual(resolve_tracks({'tracks':[track(1,[0,1])]},evidence,{'254'})[0],[])

    def test_supported_conflicting_numbers_quarantine_track(self):
        tracks,audit=resolve_tracks({'tracks':[track(1,[0,1,2,3])]},reads(1,'254',[0,1])+reads(1,'971',[2,3]),{'254','971'})
        self.assertEqual(tracks,[])
        self.assertTrue(audit)

    def test_simultaneous_tracks_cannot_claim_one_team(self):
        tracks,audit=resolve_tracks({'tracks':[track(1,[0,1]),track(2,[0,1])]},reads(1,'254',[0,1])+reads(2,'254',[0,1]),{'254'})
        self.assertEqual(tracks,[])
        self.assertTrue(audit)

    def test_unique_two_digit_partial_read_can_anchor(self):
        tracks, _ = resolve_tracks({'tracks':[track(1,[0,1])]}, reads(1,'13',[0,1]), {'1323','868','2910'})
        self.assertEqual(tracks[0]['team'], '1323')

    def test_ocr_plus_color_can_stitch_without_explicit_alliance_rosters(self):
        observations=[track(1,[0,.1]),track(2,[.3,.4])]
        evidence=reads(1,'254',[0,.1])+[
            {'id':i,'team':'','t':t,'confidence':.95,'alliance':'red'}
            for i,times in [(1,[0,.1]),(2,[.3,.4])] for t in times]
        tracks,_=resolve_tracks({'tracks':observations},evidence,{'254'})
        self.assertEqual(tracks[0]['source_ids'],[1,2])

    def test_ambiguous_or_one_digit_partial_read_does_not_anchor(self):
        self.assertIsNone(partial_team_matches('1', {'1323','2910'}))
        self.assertIsNone(partial_team_matches('23', {'1323','2230'}))

    def test_same_color_close_fragment_inherits_anchored_team(self):
        evidence = reads(1,'254',[0,1]) + [
            {'id':2,'team':'','t':1.2,'confidence':.9,'alliance':'red'},
            {'id':2,'team':'','t':1.3,'confidence':.9,'alliance':'red'},
        ]
        tracks, audit = resolve_tracks({'tracks':[track(1,[0,1]),track(2,[1.2,1.3])]}, evidence,
                                       {'254','971','111','222','333','444'},
                                       {'254':'red','971':'red','111':'red','222':'blue','333':'blue','444':'blue'})
        self.assertEqual(tracks[0]['source_ids'], [1,2])
        self.assertTrue(any(row['reason'] == 'nearby same-color fragment joined' for row in audit))

    def test_path_velocity_can_bridge_a_short_occlusion(self):
        evidence = reads(1,'254',[0,1]) + [
            {'id':2,'team':'','t':2,'confidence':.9,'alliance':'red'},
            {'id':2,'team':'','t':2.1,'confidence':.9,'alliance':'red'},
        ]
        first = {'id':1,'points':[{'t':0,'x':0,'y':0},{'t':1,'x':5,'y':0}]}
        second = {'id':2,'points':[{'t':2,'x':10,'y':0},{'t':2.1,'x':10.5,'y':0}]}
        tracks, _ = resolve_tracks({'tracks':[first,second]}, evidence, {'254','971','111','222','333','444'},
                                   {'254':'red','971':'red','111':'red','222':'blue','333':'blue','444':'blue'})
        self.assertEqual(tracks[0]['source_ids'], [1,2])

    def test_leftover_team_needs_one_color_track_only(self):
        evidence = reads(1,'254',[0,1])+reads(2,'971',[0,1]) + [
            {'id':3,'team':'','t':0,'confidence':.9,'alliance':'red'},
            {'id':3,'team':'','t':1,'confidence':.9,'alliance':'red'},
        ]
        observations=[track(1,[0,1]),track(2,[0,1]),track(3,[0,1])]
        for index, observation in enumerate(observations):
            for point in observation['points']:point['x']+=index*25
        tracks, _ = resolve_tracks({'tracks':observations}, evidence,
                                   {'254','971','111','222','333','444'},
                                   {'254':'red','971':'red','111':'red','222':'blue','333':'blue','444':'blue'})
        self.assertEqual({item['team'] for item in tracks}, {'254','971','111'})


if __name__=='__main__':unittest.main()
