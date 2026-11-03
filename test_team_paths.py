import unittest
from team_paths import resolve_tracks


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


if __name__=='__main__':unittest.main()
