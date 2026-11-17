"""Train/infer a small local identity head from reviewed source crops; no paid calls.

The detector remains frozen. Scores are regression support, not probabilities.
"""
import argparse
import hashlib
import json
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageEnhance,ImageOps

from teacher_labels import atomic_json
from train_detector import heartbeat
from train_identity_gallery import features as hsv_features


class Encoder:
    def __init__(self):
        import torch
        from torchvision.models import resnet18,ResNet18_Weights
        self.torch=torch;self.device='mps' if torch.backends.mps.is_available() else 'cpu'
        self.model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);self.model.fc=torch.nn.Identity();self.model.eval().to(self.device)

    def encode(self,images):
        torch=self.torch;vectors=[];mean=np.array([.485,.456,.406]);std=np.array([.229,.224,.225])
        for start in range(0,len(images),32):
            batch=[];hist=[]
            for image in images[start:start+32]:
                image=image.convert('RGB');padded=ImageOps.pad(image,(224,224),color=(123,116,103))
                batch.append(torch.from_numpy(((np.asarray(padded)/255-mean)/std).astype(np.float32).transpose(2,0,1)))
                h=hsv_features(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR));hist.append(h/max(np.linalg.norm(h),1e-8))
            with torch.inference_mode():deep=self.model(torch.stack(batch).to(self.device)).cpu().numpy()
            deep/=np.maximum(np.linalg.norm(deep,axis=1,keepdims=True),1e-8)
            vectors.extend(np.concatenate([deep,np.stack(hist)],axis=1)/np.sqrt(2))
        return np.stack(vectors)


def fit_head(features,labels,alpha):
    classes=np.array(sorted(set(labels)));targets=(np.array(labels)[:,None]==classes[None,:]).astype(float)
    x=np.column_stack([features,np.ones(len(features))]);weights=x.T@np.linalg.solve(x@x.T+alpha*np.eye(len(x)),targets)
    return weights,classes


def decide(scores,classes,teams,minimum_score,minimum_margin):
    accepted={}
    for i,row in enumerate(scores):
        order=np.argsort(row)[::-1];team=str(classes[order[0]])
        if team in teams and row[order[0]]>=minimum_score and row[order[0]]-row[order[1]]>=minimum_margin:accepted[i]=team
    counts={team:sum(v==team for v in accepted.values()) for team in set(accepted.values())}
    return {i:team for i,team in accepted.items() if counts[team]==1}


def source_crop(row,plans,captures):
    source=plans[row['match']];video=source['video'];cap=captures.setdefault(video,cv2.VideoCapture(video));cap.set(cv2.CAP_PROP_POS_MSEC,row['video_s']*1000);ok,bgr=cap.read()
    if not ok:raise ValueError('Missing training frame')
    image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));return image.crop(tuple(round(v) for v in row['full_frame_box']))


def train(a):
    if a.out.exists():raise ValueError('Choose a new model folder')
    config=json.loads(a.config.read_text());plans={s['match']:s for p in ['configs/focus_MQZ4Yrw86uQ.json','configs/norcal_identity_pool_v1.json'] for s in json.loads(Path(p).read_text())['sources']}
    rows=[json.loads(s) for s in Path(config['training_examples']).read_text().splitlines()];images=[];labels=[];manifest=[];captures={}
    try:
        for row in rows:
            if row['video_s'] in config['excluded_development_times'].get(row['match'],[]):continue
            if row['review_status']!='agent_reviewed_source_crop':raise ValueError('Unreviewed training input')
            image=source_crop(row,plans,captures);images.append(image);labels.append(row['label']);manifest.append(row)
        gallery=json.loads(Path(config['gallery_manifest']).read_text())
        for team,row in gallery['teams'].items():
            r=dict(row,match='2026cancmp_qm3',full_frame_box=row['full_box']);images.append(source_crop(r,plans,captures));labels.append(team);manifest.append(dict(r,label=team,review_status='agent_reviewed_enrollment_card'))
    finally:
        for cap in captures.values():cap.release()
    expanded=[];targets=[]
    for image,label in zip(images,labels):
        for flipped in [False,True]:
            base=ImageOps.mirror(image) if flipped else image
            for brightness in [.8,1.,1.2]:expanded.append(ImageEnhance.Brightness(base).enhance(brightness));targets.append(label)
    a.out.mkdir(parents=True);atomic_json(a.out/'config.json',config);atomic_json(a.out/'training_manifest.json',manifest)
    vectors=Encoder().encode(expanded);weights,classes=fit_head(vectors,targets,config['ridge_alpha'])
    np.savez_compressed(a.out/'model.npz',weights=weights,classes=classes)
    atomic_json(a.out/'provenance.json',dict(config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_rows=len(images),augmented_rows=len(expanded),classes=classes.tolist(),scope='Agent-reviewed development crops and enrollment cards; no held-out frames'))
    print(f'Trained local identity head on {len(images)} source crops / {len(expanded)} augmented embeddings; detector unchanged',flush=True)


def infer(a):
    if a.out.exists():raise ValueError('Choose a new inference folder')
    config=json.loads((a.model/'config.json').read_text());model=np.load(a.model/'model.npz',allow_pickle=False);data=json.loads(a.tracks.read_text());source=data['source'];encoder=Encoder();cap=cv2.VideoCapture(source['video']);predictions=[];a.out.mkdir(parents=True)
    atomic_json(a.out/'contract.json',dict(model_sha256=hashlib.sha256((a.model/'model.npz').read_bytes()).hexdigest(),tracks_sha256=hashlib.sha256(a.tracks.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        for index,f in enumerate(data['frames']):
            cap.set(cv2.CAP_PROP_POS_MSEC,f['video_s']*1000);ok,bgr=cap.read()
            if not ok:raise ValueError('Missing inference frame')
            x,y,x2,y2=source['views']['main'];image=Image.fromarray(cv2.cvtColor(bgr[y:y2,x:x2],cv2.COLOR_BGR2RGB));rows=f['cameras']['main']
            if rows:
                vectors=encoder.encode([image.crop(tuple(round(v) for v in r['box'])) for r in rows]);scores=np.column_stack([vectors,np.ones(len(rows))])@model['weights']
                accepted=decide(scores,model['classes'],source['teams'],config['minimum_score'],config['minimum_margin'])
            else:scores=[];accepted={}
            predictions.append(dict(video_s=f['video_s'],robots=[dict(box=r['box'],team=accepted.get(i),track_key='main:'+str(r['id']),class_scores={str(k):float(v) for k,v in zip(model['classes'],scores[i])}) for i,r in enumerate(rows)]))
            if index%10==0:print(f'Local identities {index+1}/{len(data["frames"])}',flush=True)
    finally:cap.release()
    atomic_json(a.out/'predictions.json',dict(source=source,frames=predictions,scope='Experimental local supervised identity head; no paid inference or hidden positions'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--config',type=Path,default=Path('configs/norcal_identity_student_v1.json'));t.add_argument('--out',type=Path,required=True)
    i=sub.add_parser('infer');i.add_argument('--model',type=Path,required=True);i.add_argument('--tracks',type=Path,required=True);i.add_argument('--out',type=Path,required=True)
    a=p.parse_args();stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:train(a) if a.command=='train' else infer(a)
    finally:stop.set()


if __name__=='__main__':main()
