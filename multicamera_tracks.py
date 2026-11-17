"""Offline synchronized camera tracking with repeated exact bumper OCR.

Camera IDs are separate namespaces. Cross-camera sightings share a team only after
each local track has its own repeated identity evidence. No field-map fusion or
hidden positions are fabricated. No Claude requests or credentials are used.
"""
import argparse
import hashlib
import html
import json
import os
import subprocess
import threading
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from teacher_labels import atomic_json
from train_detector import heartbeat


def track_key(camera, identifier):
    return f'{camera}:{int(identifier)}'


def resolve_identities(frames, evidence, roster):
    votes=defaultdict(lambda:defaultdict(set));seen=defaultdict(set)
    for frame in frames:
        for camera,rows in frame['cameras'].items():
            for row in rows:seen[track_key(camera,row['id'])].add(round(frame['video_s'],3))
    for row in evidence:
        if (row['confidence']>=.8 and row['team'] in roster and row['key'] in seen
                and round(row['video_s'],3) in seen[row['key']]):
            votes[row['key']][row['team']].add(round(row['video_s'],3))
    assignments={};audit=[]
    for key,names in votes.items():
        supported=[team for team,times in names.items() if len(times)>=2 and max(times)-min(times)>=.4-1e-6]
        if len(supported)==1:
            assignments[key]=supported[0]
        elif len(supported)>1:
            audit.append(dict(key=key,reason='conflicting repeated bumper reads',teams=sorted(supported)))
    # A team can appear simultaneously in different cameras. Two same-camera
    # tracks claiming it at one time are conflicting evidence, never two robots.
    blocked=set();keys=sorted(assignments)
    for i,first in enumerate(keys):
        for second in keys[i+1:]:
            if (first.split(':')[0]==second.split(':')[0] and assignments[first]==assignments[second]
                    and seen[first]&seen[second]):
                blocked.update((first,second))
                audit.append(dict(keys=[first,second],team=assignments[first],reason='simultaneous duplicate team in one camera'))
    return {key:team for key,team in assignments.items() if key not in blocked},audit


def draw_camera(image, rows, identities, camera, teams, font):
    image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(image)
    for row in rows:
        key=track_key(camera,row['id']);team=identities.get(key)
        color=('#60a5fa' if teams.get(team)=='blue' else '#f87171') if team else '#fde047'
        draw.rectangle(row['box'],outline=color,width=3)
        x,y=row['box'][:2]
        draw.text((max(0,x),max(26,y-20)),f'{team or "Unknown"} [{key}]',font=font,fill=color,stroke_width=1,stroke_fill='black')
    draw.rectangle((0,0,image.width,25),fill='black');draw.text((4,3),camera,font=font,fill='white')
    return image


def render(source, frames, identities, out, fps, weights):
    import imageio_ffmpeg
    cap=cv2.VideoCapture(source['video']);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
    writer=imageio_ffmpeg.write_frames(str(out/'synchronized_tracking.mp4'),(1280,720),fps=fps,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,ffmpeg_log_level='error');next(writer)
    cameras=list(source['views'])
    try:
        for index,frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000);ok,raw=cap.read()
            if not ok:raise ValueError('Missing render source frame')
            sheet=Image.new('RGB',(1280,720),'#161b22');draw=ImageDraw.Draw(sheet)
            draw.text((12,5),f"{source['match']} | video {frame['video_s']:.1f}s | local detector + OCR; unknowns retained",font=font,fill='white')
            for i,camera in enumerate(cameras):
                x1,y1,x2,y2=source['views'][camera]
                tile=draw_camera(raw[y1:y2,x1:x2],frame['cameras'][camera],identities,camera,source['teams'],font)
                if i==0:
                    bounds=(1280,675) if len(cameras)==1 else (1280,345)
                    tile=ImageOps.contain(tile,bounds);sheet.paste(tile,((1280-tile.width)//2,30))
                else:
                    tile=ImageOps.contain(tile,(640,310));sheet.paste(tile,((i-1)*640,385))
            writer.send(np.asarray(sheet))
            if index==len(frames)//2:sheet.save(out/'preview.jpg',quality=95)
            if index%25==0:print(f'Rendering {index+1}/{len(frames)}',flush=True)
    finally:
        writer.close();cap.release()
    rows=''.join(f'<tr><td>{html.escape(team)}</td><td>{html.escape(", ".join(sorted({key.split(":")[0] for key,value in identities.items() if value==team}))) or "Unresolved"}</td></tr>' for team in sorted(source['teams'],key=int))
    (out/'viewer.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synchronized robot cameras</title><style>body{{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1280px;margin:24px auto;padding:0 16px}}video{{width:100%;background:black}}td,th{{padding:8px 18px;text-align:left;border-bottom:1px solid #374151}}a{{color:#93c5fd}}</style><h1>{html.escape(source['match'])}: synchronized cameras</h1><p>Each camera follows its own tracks. Names require repeated local bumper reads; unknown robots remain visible. This is a prototype with no field-coordinate fusion or independently verified identity score.</p><video controls preload="metadata" src="synchronized_tracking.mp4"></video><table><tr><th>Team</th><th>Cameras with supported names</th></tr>{rows}</table><p><a href="tracks.json">Track observations</a> · <a href="evidence.json">OCR evidence</a> · <a href="summary.json">Summary and limits</a></p></html>''')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('configs/temporal_multicamera_v1.json'))
    p.add_argument('--match',default='2026wicmp_f1m1')
    p.add_argument('--weights',type=Path,default=Path('runs/local_robot_v1/weights/best.pt'))
    p.add_argument('--start',type=float,default=45)
    p.add_argument('--duration',type=float,default=20)
    p.add_argument('--fps',type=int,default=5)
    p.add_argument('--visual-memory',action='store_true',help='Reconnect camera-local observations using appearance and motion before OCR voting')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() or not args.weights.is_file():p.error('Use a new output folder and existing local weights')
    plan=json.loads(args.plan.read_text());source=next((s for s in plan['sources'] if s['match']==args.match),None)
    if source is None or not 1<=len(source['views'])<=3:p.error('Choose a configured match with up to three views')
    if args.start<0 or args.duration<=0 or not 1<=args.fps<=30:p.error('Invalid time window or sample rate')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path('.venv/ultralytics').resolve()))
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.venv/matplotlib').resolve()))
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO, settings
    settings.update({k:False for k in ('sync','hub','wandb','mlflow') if k in settings})
    args.out.mkdir(parents=True);crops=args.out/'ocr_crops';crops.mkdir()
    tracker=args.out/'bytetrack.yaml'
    tracker.write_text(f'tracker_type: bytetrack\ntrack_high_thresh: 0.3\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.4\ntrack_buffer: {args.fps}\nmatch_thresh: 0.8\nfuse_score: true\n')
    # A separate tracker/model per camera prevents cross-camera association by ID.
    models={camera:YOLO(str(args.weights)) for camera in source['views']}
    if args.visual_memory:
        from local_memory import VisualMemory, appearance
        memories={camera:VisualMemory() for camera in source['views']}
    cap=cv2.VideoCapture(source['video']);native_fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or args.fps>native_fps:raise ValueError('Missing source or sampling faster than video')
    if (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))!=(1920,1080):
        raise ValueError('Configured crop layouts require a 1920x1080 source')
    frames=[];manifest={};last_crop={};stop=threading.Event()
    threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        for i in range(round(args.duration*args.fps)):
            t=round(args.start+i/args.fps,3);cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,image=cap.read()
            if not ok:raise ValueError('Source ended before requested window')
            cameras={}
            for camera,(x1,y1,x2,y2) in source['views'].items():
                roi=image[y1:y2,x1:x2]
                result=models[camera].track(roi,persist=True,tracker=str(tracker.resolve()),imgsz=960,conf=.1,device='mps',max_det=12,verbose=False)[0]
                rows=[]
                if result.boxes.id is not None:
                    for b,identifier,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.id.int().cpu().tolist(),result.boxes.conf.cpu().tolist()):
                        rows.append(dict(id=identifier,box=[round(v,2) for v in b],confidence=round(c,4)))
                if args.visual_memory:
                    observations=[]
                    for row in rows:
                        bx1,by1,bx2,by2=map(round,row['box']);crop=roi[max(0,by1):by2,max(0,bx1):bx2]
                        if min(crop.shape[:2])<2:continue
                        vector,alliance=appearance(crop)
                        observations.append(dict(row,raw_id=row['id'],appearance=vector,alliance=alliance))
                    rows=[dict(id=state['id'],raw_id=row['raw_id'],box=row['box'],confidence=row['confidence'])
                          for state,row in memories[camera].update(observations,i/args.fps)]
                for row in rows:
                    b,identifier,c=row['box'],row['id'],row['confidence']
                    key=track_key(camera,identifier)
                    if t-last_crop.get(key,-100)<.4-1e-6 or c<.25:continue
                    bx1,by1,bx2,by2=map(round,b);dx=round((bx2-bx1)*.1);dy=round((by2-by1)*.1)
                    crop=roi[max(0,by1-dy):min(roi.shape[0],by2+dy),max(0,bx1-dx):min(roi.shape[1],bx2+dx)]
                    if min(crop.shape[:2])<12:continue
                    scale=min(3.,max(1.,300/crop.shape[1]));crop=cv2.resize(crop,None,fx=scale,fy=scale)
                    name=f'crop_{len(manifest):05d}.jpg';cv2.imwrite(str(crops/name),crop)
                    manifest[name]=dict(key=key,video_s=t,camera=camera,box=b);last_crop[key]=t
                cameras[camera]=rows
            frames.append(dict(video_s=t,cameras=cameras))
            if i%5==0:print(f'Tracked {i+1}/{round(args.duration*args.fps)} synchronized frames',flush=True)
        atomic_json(args.out/'crop_manifest.json',manifest)
        with (args.out/'ocr_raw.json').open('w') as output:
            subprocess.run(['swift','read_team_numbers.swift',str(crops)],stdout=output,check=True)
        evidence=[]
        for r in json.loads((args.out/'ocr_raw.json').read_text()):
            if str(r['team']) in source['teams'] and r['confidence']>=.8:
                evidence.append(dict(manifest[r['file']],team=str(r['team']),confidence=r['confidence'],file=r['file'],method='local_exact_bumper_ocr'))
        identities,audit=resolve_identities(frames,evidence,source['teams'])
        observed=sum(len(rows) for frame in frames for rows in frame['cameras'].values())
        named=sum(track_key(camera,r['id']) in identities for f in frames for camera,rows in f['cameras'].items() for r in rows)
        summary=dict(match=args.match,frames=len(frames),observed_boxes=observed,named_boxes=named,
            supported_teams=sorted(set(identities.values()),key=int),identity_tracks=len(identities),
            weights=str(args.weights),weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            video_sha256=hashlib.sha256(Path(source['video']).read_bytes()).hexdigest(),fps=args.fps,
            visual_memory=args.visual_memory,
            limitations=['Names are repeated OCR evidence, not independently verified truth.','Track swaps can survive between reads; conflicting supported names stay unresolved.','Cameras are synchronized crops of one configured fixed broadcast layout; cuts/zooms require different profiles.','No spatial fusion, inferred hidden positions, or six-robot completeness claim.'])
        atomic_json(args.out/'evidence.json',evidence)
        atomic_json(args.out/'tracks.json',dict(source=source,frames=frames,identities=identities,audit=audit))
        atomic_json(args.out/'summary.json',summary)
        render(source,frames,identities,args.out,args.fps,args.weights)
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        cap.release();stop.set()


if __name__=='__main__':main()
