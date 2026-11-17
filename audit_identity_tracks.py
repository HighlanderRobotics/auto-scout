"""Prepare or execute budgeted identity audits of existing local track crops.

Audits are proposals, never an answer key. Uses the original shared $4.90 ledger,
including prior spend and the existing Opus sublimit; does not train a detector.
"""
import argparse
import base64
import hashlib
import json
import threading
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps

from temporal_multicamera import Budget,OPUS
from teacher_labels import atomic_json
from train_detector import heartbeat

SYSTEM='''Inspect chronological crops of a proposed FRC robot track. A tracker can switch physical robots or follow a monitor/person/field structure. Do NOT assume these crops are the same robot. For EACH numbered sample, identify the physical robot's team only if evidence supports it; otherwise team=unknown. Distinguish physical bumper digits from team numbers printed on field displays. Use chassis shape/mechanisms and bumper color for continuity, and report switches rather than forcing one name. Never identify a field monitor as a robot. Return observations in the same sample order. A confident answer is evidence to review, not ground truth. Be conservative about digits blurred by motion. Bumper_text must contain only what you can actually read in that crop.'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks',type=Path,default=Path('runs/mqz4_identity_baseline_v1/tracks.json'))
    p.add_argument('--out',type=Path,default=Path('datasets/focus_MQZ4Yrw86uQ/opus_audit_v1'))
    p.add_argument('--max-tracks',type=int,default=12)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.tracks.read_text());source=data['source'];groups=defaultdict(list)
    for frame in data['frames']:
        for camera,rows in frame['cameras'].items():
            for row in rows:groups[(camera,row['id'])].append(dict(row,video_s=frame['video_s']))
    chosen=sorted(((key,rows) for key,rows in groups.items() if len(rows)>=10),key=lambda x:-len(x[1]))[:args.max_tracks]
    contract=hashlib.sha256(args.tracks.read_bytes()+str(args.max_tracks).encode()+SYSTEM.encode()).hexdigest()
    plan_path=args.out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())['contract']!=contract:raise ValueError('Audit context changed; use a new folder')
    plan={'contract':contract,'source_tracks':str(args.tracks),'source_sha256':hashlib.sha256(args.tracks.read_bytes()).hexdigest(),'roster':source['teams'],'tracks':[]}
    cap=cv2.VideoCapture(source['video']);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
    try:
        for (camera,identifier),rows in chosen:
            indices=sorted({round(i*(len(rows)-1)/7) for i in range(8)});sheet=Image.new('RGB',(1280,520),'#202020');samples=[]
            for tile,index in enumerate(indices):
                row=rows[index];cap.set(cv2.CAP_PROP_POS_MSEC,row['video_s']*1000);ok,frame=cap.read()
                if not ok:raise ValueError('Missing audit video frame')
                cx,cy,_,_=source['views'][camera];x1,y1,x2,y2=row['box'];dx=(x2-x1)*.1;dy=(y2-y1)*.1
                crop=frame[max(0,round(cy+y1-dy)):round(cy+y2+dy),max(0,round(cx+x1-dx)):round(cx+x2+dx)]
                image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));image=ImageOps.contain(image,(320,230));x=tile%4*320;y=tile//4*260
                sheet.paste(image,(x,y+28));ImageDraw.Draw(sheet).text((x+5,y+4),f"sample {tile} | {row['video_s']:.1f}s",font=font,fill='white')
                samples.append(dict(sample=tile,video_s=row['video_s'],box=row['box'],raw_id=row.get('raw_id')))
            image_path=args.out/f'{camera}_{identifier}.jpg';sheet.save(image_path,quality=95)
            plan['tracks'].append(dict(camera=camera,id=identifier,samples=samples,panel=image_path.name))
    finally:cap.release()
    atomic_json(plan_path,plan)
    if not args.execute:
        print(f'Prepared {len(chosen)} panels. No API calls.',flush=True);return
    root=Path('datasets/temporal_multicamera_v1_run');base=json.loads((root/'frames.json').read_text())
    budget=Budget(root,base['contract']);start=budget.data['spent_usd'];stop=threading.Event()
    threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        for track in plan['tracks']:
            schema={'type':'object','properties':{'same_physical_robot':{'type':'boolean'},'observations':{'type':'array','items':{'type':'object','properties':{
                'sample':{'type':'integer'},'team':{'type':'string','enum':list(source['teams'])+['unknown']},'confidence':{'type':'number'},
                'kind':{'type':'string','enum':['robot','field_equipment','person','unclear']},'bumper_text':{'type':'string'},'reason':{'type':'string'}},
                'required':['sample','team','confidence','kind','bumper_text','reason'],'additionalProperties':False}},'notes':{'type':'string'}},
                'required':['same_physical_robot','observations','notes'],'additionalProperties':False}
            panel=args.out/track['panel']
            content=[{'type':'text','text':'Match roster and bumper colors: '+json.dumps(source['teams'])+'. Samples are chronological. Report EACH sample independently, and flag physical identity switches.'},
                     {'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode(panel.read_bytes()).decode()}}]
            fingerprint=hashlib.sha256((SYSTEM+json.dumps(content,sort_keys=True)+json.dumps(schema,sort_keys=True)+OPUS).encode()).hexdigest()
            target=args.out/f"{track['camera']}_{track['id']}_response.json"
            response=budget.request(OPUS,content,schema,target,fingerprint,system=SYSTEM)
            if response is None:
                print('Existing shared budget or Opus sublimit reached; no further audit calls',flush=True);break
            if response['stop_reason']!='end_turn':raise ValueError('Incomplete audit; paid response preserved, not retried')
            result=json.loads(response['text']);rows=result['observations']
            if len(rows)!=len(track['samples']) or sorted(r['sample'] for r in rows)!=list(range(len(rows))):raise ValueError('Audit sample mismatch')
            if any(type(r['confidence']) not in (int,float) or not 0<=r['confidence']<=1 for r in rows):raise ValueError('Invalid audit confidence')
            atomic_json(args.out/f"{track['camera']}_{track['id']}_proposal.json",dict(track=track,audit=result,scope='Unreviewed teacher proposals; not reference truth'))
            print(f"Audited {track['camera']}:{track['id']} same_robot={result['same_physical_robot']} teams={sorted(set(r['team'] for r in rows))}; shared spend=${budget.data['spent_usd']:.4f}",flush=True)
        atomic_json(args.out/'billing_summary.json',dict(start_spent_usd=start,total_spent_usd=budget.data['spent_usd'],new_spend_usd=budget.data['spent_usd']-start,cap_usd=budget.data['cap_usd']))
    finally:budget.lock.close();stop.set()


if __name__=='__main__':main()
