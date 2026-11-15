"""Export resolved identities for the existing answer-key comparison; no relabeling."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from compare_runs import iou


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',type=Path)
    p.add_argument('--reference',type=Path,default=Path('track/final'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Choose a new output folder')
    data=json.loads(a.paths.read_text());mapping={i:t['team'] for t in data['tracks'] for i in t['source_ids']}
    frames={round(f['video_s']*1000):f for f in data['frames']}
    a.out.mkdir(parents=True)
    correct=matched=predictions=truth_count=0
    for path in sorted(a.reference.glob('f_*.json')):
        ref=json.loads(path.read_text());frame=frames.get(round(ref['time_s']*1000))
        if frame is None:continue
        robots=[dict(r,team=mapping[r['id']],visible=True,named_by='local_ocr_track_vote',bumper_text='') for r in frame['robots'] if r['id'] in mapping]
        truth=[r for r in ref['robots'] if r['visible']];truth_count+=len(truth);predictions+=len(robots)
        used=set()
        for r in sorted(robots,key=lambda r:r['confidence'],reverse=True):
            options=[(iou(r['box'],t['box']),i) for i,t in enumerate(truth) if i not in used]
            if options:
                overlap,index=max(options)
                if overlap>=.3:
                    used.add(index);matched+=1;correct+=r['team']==truth[index]['team']
        (a.out/path.name).write_text(json.dumps({'time_s':ref['time_s'],'robots':robots},indent=2))
    metrics={'resolved_predictions':predictions,'visible_reference_boxes':truth_count,'matched_iou_0_3':matched,'correct_identity':correct,
             'identity_accuracy_on_matched':correct/matched if matched else None,
             'correct_named_reference_coverage':correct/truth_count if truth_count else None,
             'scope':'Same-match demo; measures resolved identities only. Unresolved coverage included. Not independent venue validation.'}
    result=subprocess.run([sys.executable,'compare_runs.py',str(a.reference),'opus',str(a.out),'haiku55'],capture_output=True,text=True,check=True)
    (a.out/'comparison.txt').write_text(result.stdout)
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(result.stdout);print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
