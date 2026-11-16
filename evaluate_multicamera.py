"""Evaluate robot boxes on frozen side-camera annotations without paid calls."""
import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from evaluate_detector import count_matches, iou
from teacher_labels import atomic_json


def excluded_prediction(box, ignored, references):
    # Never hide a prediction which could match a scored reference.
    if any(iou(box,r['box'])>=.3 for r in references):
        return False
    area=(box[2]-box[0])*(box[3]-box[1])
    return area>0 and any(max(0,min(box[2],b[2])-max(box[0],b[0]))*
                         max(0,min(box[3],b[3])-max(box[1],b[1]))/area>.5 for b in ignored)


def prepare(config, out):
    if out.exists():
        raise ValueError('Choose a new evaluation source folder')
    plan=json.loads(config.read_text())
    out.mkdir(parents=True)
    cap=cv2.VideoCapture(plan['video'])
    if not cap.isOpened():
        raise ValueError('Missing source video')
    records=[]
    try:
        for index,row in enumerate(plan['frames']):
            cap.set(cv2.CAP_PROP_POS_MSEC,row['video_s']*1000)
            ok,frame=cap.read()
            if not ok:
                raise ValueError('Missing held-out video frame')
            x1,y1,x2,y2=plan['cameras'][row['camera']]
            image=Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
            for b in row['boxes']+row['ignore']:
                if not (0<=b[0]<b[2]<=image.width and 0<=b[1]<b[3]<=image.height):
                    raise ValueError('Invalid reference rectangle')
            path=out/f'f_{index:04d}.jpg';image.save(path,quality=95)
            records.append(dict(row,image=path.name,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    finally:
        cap.release()
    manifest=dict(plan,frames=records,annotation_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  video_sha256=hashlib.sha256(Path(plan['video']).read_bytes()).hexdigest())
    atomic_json(out/'manifest.json',manifest)
    return manifest


def evaluate(weights, dataset, out):
    if out.exists():
        raise ValueError('Choose a new evaluation output folder')
    manifest=json.loads((dataset/'manifest.json').read_text())
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path('.venv/ultralytics').resolve()))
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.venv/matplotlib').resolve()))
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO, settings
    import imageio_ffmpeg
    settings.update({k:False for k in ('sync','hub','wandb','mlflow') if k in settings})
    model=YOLO(str(weights));out.mkdir(parents=True);(out/'frames').mkdir()
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15)
    totals={str(t):Counter() for t in (.3,.5)};records=[]
    for index,row in enumerate(manifest['frames']):
        path=dataset/row['image']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError('Held-out image changed')
        result=model.predict(str(path),imgsz=960,conf=.25,device='mps',verbose=False,save=False)[0]
        references=[dict(class_id=0,box=b) for b in row['boxes']]
        predictions=[];ignored=[]
        for b,cls,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            item=dict(box=b,class_id=int(cls),confidence=c)
            (ignored if excluded_prediction(b,row['ignore'],references) else predictions).append(item)
        counts={str(t):count_matches(predictions,references,t) for t in (.3,.5)}
        for key,c in counts.items():totals[key].update(c)
        image=Image.open(path).convert('RGB');draw=ImageDraw.Draw(image)
        for b in row['ignore']:draw.rectangle(b,outline='gray',width=2)
        for ref in references:draw.rectangle(ref['box'],outline='lime',width=3)
        for prediction in predictions:draw.rectangle(prediction['box'],outline='orange',width=3)
        draw.rectangle((0,0,image.width,22),fill='black')
        draw.text((3,2),f"HELD OUT {row['camera']} t={row['video_s']} | green reference, orange prediction",font=font,fill='white')
        image.save(out/'frames'/f'f_{index:04d}.jpg',quality=95)
        records.append(dict(row,predictions=predictions,ignored_predictions=ignored,counts=counts))
        print(f'Evaluated {index+1}/{len(manifest["frames"])}',flush=True)
    metrics={}
    for key,counts in totals.items():
        tp,fp,fn=(counts[k] for k in ('true_positive','false_positive','false_negative'))
        metrics[key]=dict(counts,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0)
    report=dict(weights=str(weights),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        dataset=str(dataset),annotation_sha256=manifest['annotation_sha256'],metrics=metrics,frames=records,
        confidence_threshold=.25,imgsz=960,scope=manifest['scope'],reference_type=manifest['reference_type'])
    atomic_json(out/'metrics.json',report)
    sheet=Image.new('RGB',(1300,6*300),'#222')
    for i in range(len(records)):
        im=Image.open(out/'frames'/f'f_{i:04d}.jpg');im=ImageOps.contain(im,(650,300));sheet.paste(im,((i%2)*650,(i//2)*300))
    sheet.save(out/'preview.jpg',quality=95)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate','1','-i',str(out/'frames/f_%04d.jpg'),
                    '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',str(out/'sampled_predictions.mp4')],check=True)
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,default=Path('configs/multicamera_holdout_v1.json'))
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--dataset',type=Path,default=Path('datasets/multicamera_holdout_v1_frozen'))
    p.add_argument('--weights',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.prepare:
        prepare(args.annotations,args.dataset)
    else:
        if not args.weights or not args.out:p.error('Evaluation requires --weights and --out')
        evaluate(args.weights,args.dataset,args.out)


if __name__=='__main__':main()
