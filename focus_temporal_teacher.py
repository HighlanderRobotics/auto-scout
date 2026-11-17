"""A Jesse-style temporal identity controller for one configured match window.

Keeps the detector experiment frozen. This is paid inference/teaching evidence,
not reference truth or a claim that a local identity model has been trained.
"""
import argparse
import hashlib
import json
import threading
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps

from teacher_labels import atomic_json,InvalidTeacherLabel
from temporal_multicamera import Budget,HAIKU,OPUS,query,hard_frame
from train_detector import heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('configs/focus_MQZ4Yrw86uQ.json'))
    p.add_argument('--source-index',type=int,default=0)
    p.add_argument('--start',type=float,default=45)
    p.add_argument('--duration',type=float,default=20)
    p.add_argument('--fps',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--gallery',type=Path,help='Reviewed per-robot reference cards; never use held-out frames to build them')
    p.add_argument('--haiku-only',action='store_true',help='Fixed-model comparison without Opus initialization/corrections')
    p.add_argument('--seed',type=Path,help='Reviewed first-frame enrollment assignments, not held-out truth')
    p.add_argument('--review-after',type=float,help='Stop after this many development seconds unless development_gate.json passes')
    args=p.parse_args();source=json.loads(args.plan.read_text())['sources'][args.source_index]
    if args.duration<=0 or not 1<=args.fps<=5:p.error('Invalid clip sampling')
    if args.review_after is not None and not 0<args.review_after<args.duration:p.error('Review must precede the end of the clip')
    seed=None
    if args.seed:
        seed=json.loads(args.seed.read_text())
        if abs(seed['video_s']-args.start)>.001:raise ValueError('Enrollment seed must belong to the first frame, not held-out frames')
        if set(r['team'] for r in seed['robots'])!=set(source['teams']):raise ValueError('Enrollment roster mismatch')
    args.out.mkdir(parents=True,exist_ok=True)
    contract={'source':source,'start_s':args.start,'duration_s':args.duration,'fps':args.fps,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'query_sha256':hashlib.sha256(Path('temporal_multicamera.py').read_bytes()).hexdigest(),
              'gallery_sha256':hashlib.sha256(args.gallery.read_bytes()).hexdigest() if args.gallery else None,'haiku_only':args.haiku_only,
              'seed_sha256':hashlib.sha256(args.seed.read_bytes()).hexdigest() if args.seed else None,'review_after':args.review_after}
    path=args.out/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Context changed; choose a new run')
    atomic_json(path,contract)
    cap=cv2.VideoCapture(source['video']);native=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():raise ValueError('Missing source video')
    camera='main';crop=source['views'][camera];records=[]
    try:
        for i in range(round(args.duration*args.fps)):
            t=round(args.start+i/args.fps,3);name=f'frames/{source["match"]}/main/f_{i:04d}.jpg';image=args.out/name
            image.parent.mkdir(parents=True,exist_ok=True)
            if not image.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*native));ok,frame=cap.read()
                if not ok:raise ValueError('Missing requested frame')
                x1,y1,x2,y2=crop
                if not cv2.imwrite(str(image),frame[y1:y2,x1:x2]):raise ValueError('Could not save frame')
            records.append(dict(image=name,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),match=source['match'],camera=camera,
                                split='identity_controller',index=i,video_s=t,crop=crop,teams=source['teams']))
    finally:cap.release()
    atomic_json(args.out/'frames.json',records)
    if not args.execute:
        print(f'Prepared {len(records)} frames; no API calls',flush=True);return
    root=Path('datasets/temporal_multicamera_v1_run');original=json.loads((root/'frames.json').read_text());budget=Budget(root,original['contract'])
    start=budget.data['spent_usd'];prior=previous=None;expert_count=0;invalid=0;awaiting_review=False
    if seed:
        prior=seed['robots']
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    labels=[]
    try:
        for index,record in enumerate(records):
            if args.review_after is not None and index>=round(args.review_after*args.fps):
                gate=args.out/'development_gate.json'
                if not gate.exists() or not json.loads(gate.read_text()).get('passed'):
                    awaiting_review=True;print('Development review required before processing later frames; no further requests',flush=True);break
            target=args.out/'labels'/f'f_{index:04d}.json'
            if target.exists():
                data=json.loads(target.read_text())
                if data['image_sha256']!=record['image_sha256']:raise ValueError('Source image changed')
                expert_count+=data['teacher_model']==OPUS
            else:
                try:
                    # Opus initializes the roster at the first frame when affordable;
                    # Haiku then sees adjacent frames/state at 0.2-second intervals.
                    use_expert=index==0 and not args.haiku_only
                    data=query(budget,args.out,record,previous,prior,OPUS if use_expert else HAIKU,
                               following=records[index+1] if use_expert and index+1<len(records) else None,gallery=args.gallery)
                    if data is None and index==0:data=query(budget,args.out,record,previous,prior,HAIKU,gallery=args.gallery)
                    if data is None:raise RuntimeError('Shared Haiku budget exhausted')
                    reasons=hard_frame(data,prior,crop[2]-crop[0])
                    if not args.haiku_only and index>0 and index%40==0 and reasons and expert_count<3:
                        expert=query(budget,args.out,record,previous,prior,OPUS,data['robots'],records[index+1] if index+1<len(records) else None,gallery=args.gallery)
                        if expert is not None:expert['haiku_proposal']=data;data=expert
                    expert_count+=data['teacher_model']==OPUS
                    target.parent.mkdir(parents=True,exist_ok=True);atomic_json(target,data);invalid=0
                except InvalidTeacherLabel as error:
                    invalid+=1;out=args.out/'quarantine';out.mkdir(exist_ok=True);atomic_json(out/target.name,dict(record=record,reason=str(error)))
                    if invalid>=3:raise RuntimeError('Three invalid responses; no paid retries')
                    print(f'Excluded frame {index}; no retry',flush=True);continue
            prior,previous=data['robots'],record;labels.append(data)
            print(f'{source["match"]} temporal {index+1}/{len(records)} teacher={data["teacher_model"]} visible={sum(r["visible"] for r in prior)} shared=${budget.data["spent_usd"]:.4f}',flush=True)
        run_spend=sum(a.get('cost_usd',0) for a in budget.data['attempts'] if a['target'].startswith(str(args.out)+'/'))
        atomic_json(args.out/'billing.json',dict(run_spend_usd=run_spend,session_spend_usd=budget.data['spent_usd']-start,total_spent_usd=budget.data['spent_usd'],cap_usd=budget.data['cap_usd']))
        atomic_json(args.out/'status.json',dict(state='awaiting_development_review' if awaiting_review else 'complete',labeled_frames=len(labels)))
    finally:budget.lock.close();stop.set()
    import imageio_ffmpeg
    prefix='development_' if awaiting_review else ''
    writer=imageio_ffmpeg.write_frames(str(args.out/(prefix+'labeled.mp4')),(1280,400),fps=args.fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,ffmpeg_log_level='error');next(writer)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',22)
    frames=[]
    try:
        for data in labels:
            record=data['record'];image=Image.open(args.out/record['image']).convert('RGB');draw=ImageDraw.Draw(image);robots=[]
            for row in data['robots']:
                if not row['visible']:continue
                trusted=row['confidence']>=.8 and row['box_confidence']>=.7
                color='lime' if trusted else 'yellow';draw.rectangle(row['box'],outline=color,width=4)
                x,y=row['box'][:2];draw.text((x,max(30,y-26)),row['team']+('' if trusted else '?'),fill=color,font=font,stroke_width=1,stroke_fill='black')
                robots.append(dict(box=row['box'],team=row['team'] if trusted else None,proposed_team=row['team'],confidence=row['confidence'],box_confidence=row['box_confidence'],named_by=row['named_by']))
            tile=ImageOps.contain(image,(1280,360));sheet=Image.new('RGB',(1280,400),'#161b22');sheet.paste(tile,(0,40));ImageDraw.Draw(sheet).text((5,5),f"{source['match']} video {record['video_s']}s | temporal teacher proposals",fill='white',font=font)
            writer.send(__import__('numpy').asarray(sheet));frames.append(dict(video_s=record['video_s'],robots=robots))
            if len(frames)==len(labels)//2:sheet.save(args.out/(prefix+'preview.jpg'))
    finally:writer.close()
    atomic_json(args.out/(prefix+'predictions.json'),dict(source=source,frames=frames,scope='Paid temporal inference/controller, not a trained local identity classifier or checked reference'))


if __name__=='__main__':main()
