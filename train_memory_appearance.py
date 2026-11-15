"""Learn a regularized local appearance metric from saved Haiku/OCR gallery labels."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from train_identity_gallery import features


def learn_weights(vectors,labels):
    mean=vectors.mean(axis=0);between=np.zeros(vectors.shape[1]);within=np.zeros_like(between)
    for label in sorted(set(labels)):
        group=vectors[labels==label];center=group.mean(axis=0)
        between+=len(group)*(center-mean)**2
        within+=((group-center)**2).sum(axis=0)
    # Heavy shrinkage keeps a tiny dataset from assigning extreme feature weights.
    ratio=(between/len(vectors)+.003)/(within/len(vectors)+.003)
    return np.sqrt(np.clip(ratio,.25,4)).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--paths',type=Path);p.add_argument('--video',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    if a.paths:
        if not a.video:raise ValueError('--paths needs --video')
        data=json.loads(a.paths.read_text());mapping={i:t['team'] for t in data['tracks'] for i in t['source_ids']}
        cap=cv2.VideoCapture(str(a.video));samples=[]
        try:
            for index,frame in enumerate(data['frames']):
                if index%2:continue
                rows=[r for r in frame['robots'] if r['id'] in mapping and r['confidence']>=.5]
                if not rows:continue
                cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000);ok,image=cap.read()
                if not ok:raise ValueError('Cannot read training frame')
                for row in rows:
                    x1,y1,x2,y2=map(round,row['box']);crop=image[max(0,y1):y2,max(0,x1):x2]
                    if crop.size and min(crop.shape[:2])>=2:samples.append((features(crop),mapping[row['id']],frame['t']))
        finally:cap.release()
        train=[];test=[]
        for team in sorted({s[1] for s in samples}):
            group=[s for s in samples if s[1]==team];cut=max(1,int(len(group)*.7));end=group[cut-1][2]
            train.extend(group[:cut]);test.extend(s for s in group[cut:] if s[2]-end>=.3)
        vectors=np.stack([s[0] for s in train]);labels=np.asarray([s[1] for s in train]);weights=learn_weights(vectors,labels)
        scaled=vectors*weights;scaled/=np.maximum(np.linalg.norm(scaled,axis=1,keepdims=True),1e-8)
        np.savez_compressed(a.out/'appearance.npz',weights=weights,features=scaled,teams=labels)
        correct=0
        for vector,team,t in test:
            vector=vector*weights;vector/=max(float(np.linalg.norm(vector)),1e-8)
            correct+=str(labels[np.argmax(scaled@vector)])==team
        report={'train_crops':len(train),'test_crops':len(test),'correct_pseudo_labels':correct,'agreement':correct/len(test) if test else None,
            'scope':'Additional crops from accepted teacher/OCR tracks; chronological per-team holdout with >=0.3 second gap, same camera/match, not independent identity truth.'}
        (a.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(report);return
    gallery=np.load('runs/local_identity_gallery_v1/gallery.npz',allow_pickle=False)
    weights=learn_weights(gallery['features'],gallery['teams'])
    vectors=gallery['features']*weights;vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
    np.savez_compressed(a.out/'appearance.npz',weights=weights,features=vectors,teams=gallery['teams'])
    old=json.loads(Path('runs/local_identity_gallery_v1/metrics.json').read_text());correct=0;predictions=[]
    for row in old['predictions']:
        vector=features(Path('runs/team_paths_free_evidence_pixels_v1/ocr_crops')/row['image'])*weights
        vector/=max(float(np.linalg.norm(vector)),1e-8)
        team=str(gallery['teams'][np.argmax(vectors@vector)]);correct+=team==row['pseudo_label']
        predictions.append(dict(image=row['image'],pseudo_label=row['pseudo_label'],prediction=team))
    report={'train_crops':len(vectors),'test_crops':len(predictions),'correct_pseudo_labels':correct,
        'agreement':correct/len(predictions),'scope':'Same-track held-out crops with saved Haiku/OCR pseudo-labels. Metric weights are trained only on gallery training crops; not independent identity accuracy.',
        'predictions':predictions}
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='predictions'})


if __name__=='__main__':main()
