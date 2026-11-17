"""Classify frozen detector crops with reviewed identity references, without trusting track IDs.

Outputs paid teacher proposals, not checked truth or trained model weights.
Use development first; all calls share the original temporal teaching ledger.
"""
import argparse
import base64
import hashlib
import json
import math
import threading
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageOps

from teacher_labels import atomic_json
from temporal_multicamera import Budget,HAIKU,OPUS
from train_detector import heartbeat

SYSTEM='''Identify physical FRC robots in independent numbered detector crops. A crop may be a person, monitor, field structure, partial robot, or two overlapping robots. Classify ONLY its centered physical subject. Ignore robots displayed on monitors. Match chassis, hopper and mechanisms to the supplied identity reference cards; loose yellow fuel and people are not identifying features. Use CURRENT roster bumper colors, which can change between matches. Read actual digits when legible. Do not infer a team from what remains in the roster. Return unknown for ambiguous identities or nonrobots. Detector track IDs are intentionally omitted and are not evidence of identity. Confidence measures identity certainty; never invent readable digits. Reference cards are not current crops.'''


def classify(rows, teams):
    """Conservative per-frame naming; conflicting duplicate team claims stay unknown."""
    claims={}
    for i,row in enumerate(rows):
        team=row['team']
        if team in teams and row['kind']=='robot' and row['confidence']>=.8:
            claims.setdefault(team,[]).append(i)
    accepted={}
    for team,indices in claims.items():
        if len(indices)==1:accepted[indices[0]]=team
    return accepted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks',type=Path,required=True)
    p.add_argument('--gallery',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+',required=True)
    p.add_argument('--model',choices=[HAIKU,OPUS],default=HAIKU)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    data=json.loads(a.tracks.read_text());source=data['source'];teams=source['teams'];a.out.mkdir(parents=True,exist_ok=True)
    with Path(source['video']).open('rb') as video:video_sha256=hashlib.file_digest(video,'sha256').hexdigest()
    contract=dict(tracks_sha256=hashlib.sha256(a.tracks.read_bytes()).hexdigest(),source_video_sha256=video_sha256,gallery_sha256=hashlib.sha256(a.gallery.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),times=a.times,model=a.model)
    cp=a.out/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Choose a new run; context changed')
    atomic_json(cp,contract)
    cap=cv2.VideoCapture(source['video']);plans=[]
    try:
        for index,t in enumerate(a.times):
            frame=next(f for f in data['frames'] if abs(f['video_s']-t)<.001)
            rows=frame['cameras']['main'];cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read()
            if not ok:raise ValueError('Missing source frame')
            x1,y1,x2,y2=source['views']['main'];image=Image.fromarray(cv2.cvtColor(bgr[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
            panel=Image.new('RGB',(960,max(1,math.ceil(len(rows)/4))*230),'#222');draw=ImageDraw.Draw(panel)
            for j,row in enumerate(rows):
                x,y,x2,y2=row['box'];crop=image.crop((max(0,int(x)-8),max(0,int(y)-8),min(image.width,int(x2)+8),min(image.height,int(y2)+8)))
                tile=ImageOps.contain(crop,(236,200));px=j%4*240;py=j//4*230;panel.paste(tile,(px,py+28));draw.text((px+4,py+6),f'Crop {j}',fill='white')
            path=a.out/f'panel_{index:04d}.jpg';panel.save(path,quality=95)
            plans.append(dict(video_s=t,rows=rows,panel=str(path),panel_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    finally:cap.release()
    atomic_json(a.out/'plan.json',plans)
    if not a.execute:print(f'Prepared {len(plans)} panels; no API calls');return
    root=Path('datasets/temporal_multicamera_v1_run');budget=Budget(root,json.loads((root/'frames.json').read_text())['contract'])
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start();predictions=[]
    schema={'type':'object','properties':{'crops':{'type':'array','items':{'type':'object','properties':{
        'crop':{'type':'integer'},'team':{'type':'string','enum':list(teams)+['unknown']},'kind':{'type':'string','enum':['robot','person','field_equipment','unclear']},
        'confidence':{'type':'number'},'bumper_text':{'type':'string'},'reason':{'type':'string'}},
        'required':['crop','team','kind','confidence','bumper_text','reason'],'additionalProperties':False}}},'required':['crops'],'additionalProperties':False}
    try:
        for index,plan in enumerate(plans):
            content=[{'type':'text','text':'CURRENT roster/colors: '+json.dumps(teams)+'; first image is labeled reference cards; second image is independent CURRENT numbered crops. Return exactly one record for each crop numbered 0..'+str(len(plan['rows'])-1)+'.'}]
            for path in [a.gallery,Path(plan['panel'])]:content.append({'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode(path.read_bytes()).decode()}})
            fingerprint=hashlib.sha256((SYSTEM+json.dumps(content,sort_keys=True)+json.dumps(schema,sort_keys=True)+a.model).encode()).hexdigest()
            result=budget.request(a.model,content,schema,a.out/f'response_{index:04d}.json',fingerprint,system=SYSTEM)
            if result is None:raise RuntimeError('Shared model allowance exhausted')
            if result['stop_reason']!='end_turn':raise ValueError('Incomplete response; no paid retry')
            labels=json.loads(result['text'])['crops'];labels.sort(key=lambda r:r['crop'])
            if [r['crop'] for r in labels]!=list(range(len(plan['rows']))):raise ValueError('Crop list mismatch; no paid retry')
            if any(type(r['confidence']) not in (int,float) or not math.isfinite(r['confidence']) or not 0<=r['confidence']<=1 for r in labels):raise ValueError('Invalid confidence')
            accepted=classify(labels,teams);rows=[]
            for j,(row,label) in enumerate(zip(plan['rows'],labels)):
                rows.append(dict(box=row['box'],team=accepted.get(j),proposed_team=label['team'],confidence=label['confidence'],track_key='main:'+str(row['id']),teacher=label))
            predictions.append(dict(video_s=plan['video_s'],robots=rows))
            atomic_json(a.out/f'labels_{index:04d}.json',dict(plan=plan,labels=labels,teacher_model=a.model,request_fingerprint=fingerprint,source_video_sha256=video_sha256))
            atomic_json(a.out/'predictions.json',dict(source=source,frames=predictions,scope='Independent crop teacher proposals, not checked truth or trained weights'))
            print(f'Crops {index+1}/{len(plans)} t={plan["video_s"]} accepted={len(accepted)} shared=${budget.data["spent_usd"]:.4f}',flush=True)
        cost=sum(r.get('cost_usd',0) for r in budget.data['attempts'] if r['target'].startswith(str(a.out)+'/'))
        atomic_json(a.out/'billing.json',dict(run_spend_usd=cost,total_spent_usd=budget.data['spent_usd']))
    finally:budget.lock.close();stop.set()


if __name__=='__main__':main()
