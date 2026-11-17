import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_identity_checkpoints import matching


class IdentityCheckpointTests(unittest.TestCase):
    def test_matching_uses_geometry_not_proposed_names(self):
        refs=[dict(team='3970',box=[0,0,10,10]),dict(team='7137',box=[20,0,30,10])]
        preds=[dict(team='7137',box=[0,0,10,10]),dict(team='3970',box=[20,0,30,10])]
        self.assertEqual(matching(refs,preds),{0:0,1:1})
        self.assertEqual(matching(refs,[dict(box=[100,100,110,110])]),{})

    def test_compare_runs_accepts_new_match_roster(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);roster=r/'roster.json';roster.write_text(json.dumps({'3970':'blue','9470':'red'}))
            for folder in ['reference','candidate']:
                (r/folder).mkdir();(r/folder/'f_0000.json').write_text(json.dumps({'robots':[dict(team='3970',box=[0,0,10,10],visible=True,named_by='test')]}))
            result=subprocess.run([sys.executable,'compare_runs.py','--roster',str(roster),str(r/'reference'),'opus',str(r/'candidate'),'haiku55'],capture_output=True,text=True,check=True)
            self.assertIn('100%',result.stdout)


if __name__=='__main__':unittest.main()
