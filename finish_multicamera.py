"""Offline labeled teaching videos and an appearance metric from repeated anchors."""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from teacher_labels import atomic_json
from train_identity_gallery import features
from train_memory_appearance import learn_weights


def appearance(root, out):
    out.mkdir(parents=True, exist_ok=False)
    labels = [json.loads(p.read_text()) for p in sorted((root/'labels').glob('*/*/*.json'))]
    anchors = {}
    for d in labels:
        record = d['record']
        if record['split'] != 'train':
            continue
        for r in d['robots']:
            if (r['visible'] and r['named_by']=='bumper_number' and r['bumper_text'].strip()==r['team']
                    and r['confidence']>=.8 and r['box_confidence']>=.7):
                anchors.setdefault((record['match'],record['camera'],r['team']),set()).add(record['index'])
    anchors = {key:times for key,times in anchors.items() if len(times)>=2 and max(times)-min(times)>=2}
    train, test = [], []
    records = []
    for d in labels:
        record=d['record']
        if record['split']!='train' or record['index']%5 or not d['usable']:
            continue
        image=cv2.imread(str(root/record['image']))
        if image is None:
            raise ValueError('Missing appearance crop source')
        for r in d['robots']:
            key=(record['match'],record['camera'],r['team'])
            if key not in anchors or not r['visible'] or r['confidence']<.8 or r['box_confidence']<.8:
                continue
            # Chronological holdout; no test observations train feature weights.
            split='train' if record['index']<60 else 'test' if record['index']>=70 else None
            if split is None:
                continue
            x1,y1,x2,y2=map(round,r['box']);crop=image[y1:y2,x1:x2]
            if min(crop.shape[:2])<8:
                continue
            vector=features(crop)
            (train if split=='train' else test).append((vector,r['team']))
            records.append(dict(match=record['match'],camera=record['camera'],index=record['index'],team=r['team'],split=split))
    report={'training_crops':len(train),'test_crops':len(test),'repeated_anchor_groups':len(anchors),
            'scope':'Same-match chronological holdout with a two-second gap. Teacher/bumper pseudo-label agreement, not independent identity accuracy. PNW is excluded from metric training. Not installed as the default model.',
            'records':records}
    if len(train)<10 or len({t for _,t in train})<2:
        report['state']='insufficient_trusted_identity_crops'
    else:
        vectors=np.stack([v for v,_ in train]);teams=np.asarray([t for _,t in train]);weights=learn_weights(vectors,teams)
        vectors*=weights;vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
        np.savez_compressed(out/'appearance.npz',weights=weights,features=vectors,teams=teams)
        evaluated=correct=0
        for vector,target in test:
            if target not in teams:
                continue
            vector=vector*weights;vector/=max(float(np.linalg.norm(vector)),1e-8)
            correct+=str(teams[np.argmax(vectors@vector)])==target;evaluated+=1
        report.update(state='trained_experiment',evaluated_pseudo_labels=evaluated,correct_pseudo_labels=correct,
                      agreement=correct/evaluated if evaluated else None,teams=sorted(set(teams.tolist())))
    atomic_json(out/'metrics.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


def videos(root, out):
    import imageio_ffmpeg
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'frames.json').read_text());fps=json.loads((root/'plan.json').read_text())['fps']
    groups={}
    for r in manifest['frames']:
        groups.setdefault((r['match'],r['camera']),[]).append(r)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
    for (match,camera),records in groups.items():
        frames=out/(match+'_'+camera);frames.mkdir()
        for r in records:
            with Image.open(root/r['image']) as source:
                image=source.convert('RGB')
            draw=ImageDraw.Draw(image)
            label=root/'labels'/match/camera/(Path(r['image']).stem+'.json')
            model='No usable teacher label'
            if label.exists():
                d=json.loads(label.read_text());model=d['teacher_model']
                for robot in d['robots']:
                    if not robot['visible']:
                        continue
                    certain=robot['confidence']>=.8
                    color='lime' if certain else 'yellow'
                    draw.rectangle(robot['box'],outline=color,width=3)
                    x,y=robot['box'][:2]
                    draw.text((x,max(26,y-20)),f"{robot['team']}{'' if certain else '?'} ID {robot['confidence']:.2f}",
                              fill=color,font=font,stroke_width=1,stroke_fill='black')
            draw.rectangle((0,0,image.width,25),fill='black')
            draw.text((4,3),f'{match} {camera} t={r["video_s"]:.1f}s | {model} | TEACHER PROPOSAL',fill='white',font=font)
            image.save(frames/f"f_{r['index']:04d}.jpg",quality=90)
        path=out/(match+'_'+camera+'.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate',str(fps),'-i',str(frames/'f_%04d.jpg'),
                        '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
        print('Labeled video:',path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('datasets/temporal_multicamera_v1_run'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=('appearance','videos'),required=True)
    args=p.parse_args()
    (appearance if args.stage=='appearance' else videos)(args.root,args.out)


if __name__=='__main__':
    main()
