"""Train appearance from exact local OCR of teacher-localized crops; no API calls.

Teacher-assigned names are proposals only. Training/test identities here come from
Apple Vision exact roster reads, with a temporal gap and no validation-match data.
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from teacher_labels import atomic_json
from train_identity_gallery import features
from train_memory_appearance import learn_weights


def split_index(index):
    return 'train' if index<60 else 'test' if index>=70 else None


def exact_identity(reads, roster):
    teams={str(r['team']) for r in reads if r.get('confidence',0)>=.8 and str(r.get('team')) in roster}
    return next(iter(teams)) if len(teams)==1 else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('datasets/temporal_multicamera_v1_run'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);crops=args.out/'crops';crops.mkdir()
    excluded=set(json.loads((args.root/'review_gate.json').read_text())['excluded_streams'])
    samples=[]
    for path in sorted((args.root/'labels').glob('*/*/*.json')):
        data=json.loads(path.read_text());record=data['record']
        stream=record['match']+'/'+record['camera'];split=split_index(record['index'])
        if stream in excluded or record['split']!='train' or record['index']%2 or split is None or not data['usable']:
            continue
        image=cv2.imread(str(args.root/record['image']))
        if image is None:raise ValueError('Missing teacher image')
        for robot in data['robots']:
            if not robot['visible'] or robot['box_confidence']<.7:
                continue
            x1,y1,x2,y2=map(round,robot['box']);dx=round((x2-x1)*.1);dy=round((y2-y1)*.1)
            box=[max(0,x1-dx),max(0,y1-dy),min(image.shape[1],x2+dx),min(image.shape[0],y2+dy)]
            crop=image[box[1]:box[3],box[0]:box[2]]
            if min(crop.shape[:2])<12:continue
            name=f'crop_{len(samples):05d}.jpg'
            # Upscaling only aids text OCR; it adds no independent image detail.
            scale=min(3.,max(1.,300/crop.shape[1]));crop=cv2.resize(crop,None,fx=scale,fy=scale)
            cv2.imwrite(str(crops/name),crop)
            samples.append(dict(file=name,split=split,match=record['match'],camera=record['camera'],
                index=record['index'],video_s=record['video_s'],box=box,roster=list(record['teams']),
                source_image=record['image'],source_image_sha256=record['image_sha256'],
                teacher_proposal=robot['team']))
    atomic_json(args.out/'crop_manifest.json',samples)
    with (args.out/'ocr_reads.json').open('w') as out:
        subprocess.run(['swift','read_team_numbers.swift',str(crops)],stdout=out,check=True)
    reads={}
    for row in json.loads((args.out/'ocr_reads.json').read_text()):reads.setdefault(row['file'],[]).append(row)
    accepted=[]
    for s in samples:
        team=exact_identity(reads.get(s['file'],[]),s['roster'])
        if team:accepted.append(dict(s,team=team,label_source='local_exact_roster_ocr'))
    counts=Counter(s['team'] for s in accepted if s['split']=='train')
    # Require repeated independent times in training, never use test reads to choose classes.
    teams={team for team,n in counts.items() if n>=3 and len({s['index'] for s in accepted if s['split']=='train' and s['team']==team})>=3}
    train=[s for s in accepted if s['split']=='train' and s['team'] in teams]
    test=[s for s in accepted if s['split']=='test' and s['team'] in teams]
    report=dict(state='insufficient_ocr_data',candidate_crops=len(samples),exact_ocr_crops=len(accepted),
        training_crops=len(train),test_crops=len(test),training_class_counts=dict(counts),teams=sorted(teams),
        scope='Same-match chronological holdout with a two-second gap; independent local OCR pseudo-labels, not human ground truth. Excluded streams and PNW are not used. Never installed as default.',
        accepted_crops=accepted)
    if len(train)>=20 and len(teams)>=2 and test:
        vectors=np.stack([features(crops/s['file']) for s in train]);labels=np.asarray([s['team'] for s in train])
        weights=learn_weights(vectors,labels);scaled=vectors*weights;scaled/=np.maximum(np.linalg.norm(scaled,axis=1,keepdims=True),1e-8)
        np.savez_compressed(args.out/'appearance.npz',weights=weights,features=scaled,teams=labels)
        predictions=[]
        for sample in test:
            raw=features(crops/sample['file']);v=raw*weights;v/=max(float(np.linalg.norm(v)),1e-8)
            predicted=str(labels[np.argmax(scaled@v)])
            baseline_vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
            raw/=max(float(np.linalg.norm(raw)),1e-8)
            baseline=str(labels[np.argmax(baseline_vectors@raw)])
            predictions.append(dict(file=sample['file'],team=sample['team'],prediction=predicted,baseline_prediction=baseline,
                                    camera=sample['camera'],correct=predicted==sample['team']))
        report.update(state='trained_experiment',agreement=sum(s['correct'] for s in predictions)/len(predictions),
            unweighted_agreement=sum(s['baseline_prediction']==s['team'] for s in predictions)/len(predictions),predictions=predictions)
    atomic_json(args.out/'metrics.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['accepted_crops','predictions']},indent=2),flush=True)


if __name__=='__main__':main()
