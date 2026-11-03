"""Offline landmark-recognition pilot. Rough same-camera labels, not field calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import cv2
from PIL import Image, ImageDraw

NAMES = ['blue_hub', 'red_hub', 'blue_far_bump', 'blue_near_bump', 'red_far_bump', 'red_near_bump']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('datasets/landmarks_v1'))
    p.add_argument('--run', type=Path, default=Path('runs/landmarks_v1'))
    p.add_argument('--epochs', type=int, default=30)
    a = p.parse_args()
    if a.out.exists() or a.run.exists():
        p.error('Choose new dataset and run folders')
    labels = json.loads(Path('start/prematch_labels.json').read_text())
    cap = cv2.VideoCapture('einstein_final_1.mp4')
    if not cap.isOpened():
        p.error('Restore einstein_final_1.mp4 first')
    records = []
    # A temporal gap separates train and validation, but this is still one camera/match.
    samples = [('train', 5.5+i*.4) for i in range(50)] + [('val', 35.5+i*.5) for i in range(20)]
    for split, t in samples:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read {t}')
        height, width = frame.shape[:2]
        frame = frame[110:720]
        h, w = frame.shape[:2]
        rows = []
        for k, name in enumerate(NAMES):
            pts = labels['areas'][name]
            x1, x2 = min(v[0] for v in pts), max(v[0] for v in pts)
            y1, y2 = min(v[1] for v in pts)-110, max(v[1] for v in pts)-110
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(name)
            rows.append(f'{k} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        stem = f'f_{round(t*1000):06d}'
        image = a.out/'images'/split/(stem+'.jpg')
        label = a.out/'labels'/split/(stem+'.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image), frame)
        label.write_text('\n'.join(rows)+'\n')
        records.append({'split':split, 'time_s':t, 'image':str(image), 'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
        if split == 'train' and t == 5.5:
            preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(preview)
            for name in NAMES:
                pts=labels['areas'][name]; box=(min(v[0] for v in pts),min(v[1] for v in pts)-110,max(v[0] for v in pts),max(v[1] for v in pts)-110)
                draw.rectangle(box, outline='lime', width=3);draw.text(box[:2], name, fill='white')
            preview.save(a.out/'labels_preview.jpg')
    cap.release()
    (a.out/'data.yaml').write_text(f'path: {json.dumps(str(a.out.resolve()))}\ntrain: images/train\nval: images/val\nnames: {json.dumps(NAMES)}\n')
    manifest={'scope':'Rough static-camera pilot only. Not surveyed floor landmarks; no cross-camera validation.', 'source':'einstein_final_1.mp4', 'annotation':'start/prematch_labels.json', 'annotation_sha256':hashlib.sha256(Path('start/prematch_labels.json').read_bytes()).hexdigest(), 'frames':records}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared 50 train / 20 validation frames. Rough labels; same-camera validation only.',flush=True)
    subprocess.run([sys.executable,'-u','train_detector.py','--data',str(a.out/'data.yaml'),'--out',str(a.run),'--epochs',str(a.epochs),'--imgsz','640','--batch','4'],check=True)


if __name__ == '__main__':
    main()
