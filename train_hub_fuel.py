"""Experimental alliance fuel-increase estimator from visible hub motion and fuel OCR.

Never an attempt counter or per-robot made counter. Targets are unreviewed dedicated
fuel-counter OCR with a fixed two-second nominal delay. Does not deploy predictions.
"""
import argparse,hashlib,json,threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from hybrid_shooting import feature_sequence,score_increments
from teacher_labels import atomic_json
from train_detector import heartbeat


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--train-run',type=Path,nargs='+');p.add_argument('--validation-run',type=Path);a=p.parse_args()
 if a.out.exists():raise ValueError('Choose a new model folder')
 # Fixed before validation: hub plus airborne approach area; native full-frame pixels.
 hubs={'blue':[465,220,685,465],'red':[1175,220,1420,470]};arrows={'blue':[300,65,350,120],'red':[1560,65,1620,120]};examples=[];source_hashes={};board_hashes={}
 if bool(a.train_run)!=bool(a.validation_run):p.error('Provide both --train-run and --validation-run for a separate-match check')
 jobs=[(root,'train',range(10,108)) for root in a.train_run]+[(a.validation_run,'validation',range(10,108))] if a.train_run else [(a.run,'train',range(10,108)),(a.run,'validation',range(120,138))]
 if a.train_run and a.validation_run.resolve() in {root.resolve() for root in a.train_run}:raise ValueError('Separate-match validation run overlaps training')
 stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
 try:
  for source,split,times in jobs:
   m=json.loads((source/'manifest.json').read_text());board=json.loads((source/'scoreboard.json').read_text());incs=score_increments(board['rows']);bykey={(r['alliance'],r['start_s']):r['fuel_delta'] for r in incs};source_hashes[str(source)]=m['source_video_sha256'];board_hashes[str(source)]=hashlib.sha256((source/'scoreboard.json').read_bytes()).hexdigest()
   cap=cv2.VideoCapture(m['config']['video'])
   try:
    for index,t in enumerate(times):
     images=[]
     for sample in np.linspace(t,t+1,6):
      cap.set(cv2.CAP_PROP_POS_MSEC,float(sample)*1000);ok,bgr=cap.read()
      if not ok:raise ValueError('Missing training source')
      images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
     for alliance,roi in hubs.items():
      target=bykey.get((alliance,t+2));previous=bykey.get((alliance,t-1))
      if target is None or previous is None:continue
      visual=feature_sequence([im.crop(tuple(roi)) for im in images]);arrow=np.asarray(images[3].crop(tuple(arrows[alliance])));hsv=cv2.cvtColor(arrow,cv2.COLOR_RGB2HSV);active=float(((hsv[:,:,0]>=18)&(hsv[:,:,0]<=42)&(hsv[:,:,1]>=90)&(hsv[:,:,2]>=100)).mean())
      features=[*visual,active,float(previous),float(alliance=='blue')]
      examples.append(dict(source_run=str(source),video_start_s=t,counter_start_s=t+2,counter_end_s=t+3,alliance=alliance,features=features,target_fuel_delta=target,persistence_baseline=previous,split=split,review_status='unreviewed_ocr_target'))
     if index%10==0:print(f'{source.name} {split} hub examples {index+1}/{len(times)}',flush=True)
   finally:cap.release()
 finally:stop.set()
 train=[r for r in examples if r['split']=='train'];valid=[r for r in examples if r['split']=='validation']
 if len(train)<20 or not valid:raise ValueError('Insufficient valid fuel-counter targets')
 x=np.array([r['features'] for r in train]);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);z=np.column_stack([(x-mean)/scale,np.ones(len(x))]);y=np.log1p([r['target_fuel_delta'] for r in train]);w=np.linalg.solve(z.T@z+5*np.eye(z.shape[1]),z.T@y)
 predictions=[]
 for r in valid:
  expected=float(np.expm1(np.clip(np.append((np.array(r['features'])-mean)/scale,1)@w,0,np.log1p(40))))
  predictions.append(dict(r,expected_alliance_fuel_delta=expected,review_required=True,attributed_team=None))
 mae=float(np.mean([abs(r['expected_alliance_fuel_delta']-r['target_fuel_delta']) for r in predictions]));persist=float(np.mean([abs(r['persistence_baseline']-r['target_fuel_delta']) for r in predictions]));zero=float(np.mean([r['target_fuel_delta'] for r in predictions]))
 a.out.mkdir(parents=True);np.savez_compressed(a.out/'alliance_fuel_model.npz',weights=w,mean=mean,scale=scale)
 atomic_json(a.out/'training_examples.json',examples);atomic_json(a.out/'validation_predictions.json',predictions)
 atomic_json(a.out/'result.json',dict(status='experimental_not_deployed',training_examples=len(train),validation_examples=len(valid),validation_mae_against_ocr=mae,persistence_mae=persist,zero_baseline_mae=zero,beats_persistence=mae<persist,nominal_counter_delay_s=2,hub_crops=hubs,arrow_crops=arrows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_video_sha256=source_hashes,scoreboard_sha256=board_hashes,separate_match_validation=bool(a.validation_run),scope='Alliance credited fuel increments only, weak OCR targets. Not human-verified accuracy, robot attribution, attempted balls, or measured shooting accuracy. No validation tuning or deployment.'))
 print(json.dumps(dict(mae=mae,persistence_mae=persist,zero_mae=zero,training=len(train),validation=len(valid))),flush=True)

if __name__=='__main__':main()
