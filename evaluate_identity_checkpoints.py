"""Score listed identity checkpoints without treating unlisted robots as negatives."""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import lap
import numpy as np

from evaluate_detector import iou
from teacher_labels import atomic_json


def matching(references, predictions, threshold=.3):
    if not references or not predictions:return {}
    cost=np.ones((len(references),len(predictions)))
    for i,ref in enumerate(references):
        for j,pred in enumerate(predictions):
            overlap=iou(ref['box'],pred['box'])
            if overlap>=threshold:cost[i,j]=-(len(references)+1+overlap)
    _,assignment,_=lap.lapjv(cost,extend_cost=True,cost_limit=0)
    return {i:int(j) for i,j in enumerate(assignment) if j>=0 and cost[i,j]<0}


def load_predictions(path):
    data=json.loads(path.read_text());frames={}
    for f in data['frames']:
        if 'cameras' in f:
            rows=[]
            for camera,robots in f['cameras'].items():
                for r in robots:
                    key=f'{camera}:{r["id"]}'
                    rows.append(dict(r,team=data['identities'].get(key),track_key=key))
        else:
            rows=[dict(r,track_key=r.get('track_key','teacher:'+str(r.get('proposed_team',r.get('team'))))) for r in f['robots']]
        frames[round(f['video_s']*1000)]=rows
    return data['source'],frames


def evaluate(reference, predictions, out, split='heldout', exclude_enrollment=False):
    if out.exists():raise ValueError('Choose a new score folder')
    source,frames=load_predictions(predictions);out.mkdir(parents=True)
    reference_data=json.loads(reference.read_text());refs=reference_data[split]
    if exclude_enrollment:refs=[f for f in refs if not f.get('used_for_enrollment') and not f.get('used_for_gallery')]
    total=correct=wrong=unknown=missing=0;records=[];history=defaultdict(list)
    comparison_ref=out/'comparison_reference';comparison_run=out/'comparison_predictions';comparison_ref.mkdir();comparison_run.mkdir()
    for index,frame in enumerate(refs):
        rows=frames.get(round(frame['video_s']*1000),[]);assignments=matching(frame['robots'],rows)
        total+=len(frame['robots']);export=[]
        for i,ref in enumerate(frame['robots']):
            matched=rows[assignments[i]] if i in assignments else None
            team=matched.get('team') if matched else None
            status='missing' if matched is None else 'unknown' if team is None else 'correct' if team==ref['team'] else 'wrong'
            correct+=status=='correct';wrong+=status=='wrong';unknown+=status=='unknown';missing+=status=='missing'
            if matched:
                history[matched['track_key']].append((frame['video_s'],ref['team']))
                if team:export.append(dict(box=matched['box'],team=team,visible=True,named_by='identity_controller',bumper_text=''))
            records.append(dict(video_s=frame['video_s'],reference=ref,prediction=matched,status=status))
        name=f'f_{index:04d}.json';atomic_json(comparison_ref/name,dict(time_s=frame['video_s'],robots=[dict(r,visible=True,named_by='reviewed_checkpoint',bumper_text='') for r in frame['robots']]))
        atomic_json(comparison_run/name,dict(time_s=frame['video_s'],robots=export))
    switches=sum(a[1]!=b[1] for sequence in history.values() for a,b in zip(sorted(sequence),sorted(sequence)[1:]))
    result=dict(split=split,exclude_enrollment=exclude_enrollment,checkpoints=total,correct=correct,wrong=wrong,unknown=unknown,missing=missing,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        correct_named_coverage=correct/total if total else None,accuracy_on_named=correct/(correct+wrong) if correct+wrong else None,
        confirmed_track_identity_changes=switches,records=records,
        scope=reference_data['scope'],reference_type=reference_data['reference_type'],
        limitations='Only listed references are scored. Track changes are checkpoint witnesses, not an exhaustive ID-switch count. compare_runs.txt is supplemental on the geometrically matched named subset; use these metrics for missing/unknown/wrong counts.')
    atomic_json(out/'metrics.json',result);atomic_json(out/'roster.json',source['teams'])
    compare=subprocess.run([sys.executable,'compare_runs.py','--roster',str(out/'roster.json'),str(comparison_ref),'opus',str(comparison_run),'haiku55'],text=True,capture_output=True,check=True)
    (out/'compare_runs.txt').write_text(compare.stdout)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path)
    p.add_argument('--reference',type=Path,default=Path('configs/mqz4_identity_checkpoints_v1.json'))
    p.add_argument('--split',choices=['development','heldout'],default='heldout')
    p.add_argument('--exclude-enrollment',action='store_true')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.reference,a.predictions,a.out,a.split,a.exclude_enrollment)


if __name__=='__main__':main()
