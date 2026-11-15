"""Train and test a small local nearest-gallery identity classifier from teacher anchors.

Experimental HSV appearance features; held-out crops are same-track, same-camera,
pseudo-labeled examples. These results do not establish independent team accuracy.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def features(path):
    image=cv2.imread(str(path))
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    rows=[]
    for region in (hsv,hsv[:hsv.shape[0]//2],hsv[hsv.shape[0]//2:]):
        hist=cv2.calcHist([region],[0,1],None,[18,8],[0,180,0,256]).reshape(-1)
        rows.extend((hist/max(float(hist.sum()),1))**.5)
    return np.asarray(rows,dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--named-paths',type=Path,required=True)
    p.add_argument('--crops',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.named_paths.read_text())
    train=[];test=[]
    for track in data['tracks']:
        for identifier in track['source_ids']:
            files=sorted(args.crops.glob(f'id{identifier}_t*.jpg'))
            for index,path in enumerate(files):
                (train if index%2==0 else test).append((path,track['team']))
    if not train or not test:
        raise ValueError('Need separate training and held-out crops')
    vectors=np.stack([features(path) for path,_ in train])
    labels=np.array([team for _,team in train])
    np.savez_compressed(args.out/'gallery.npz',features=vectors,teams=labels)
    predictions=[]
    for path,target in test:
        distances=((vectors-features(path))**2).sum(axis=1)
        predicted=str(labels[int(distances.argmin())])
        predictions.append({'image':path.name,'pseudo_label':target,'prediction':predicted,'correct':predicted==target})
    report={'model':'HSV spatial nearest-gallery classifier','training_crops':len(train),
        'held_out_crops':len(test),'teams':sorted(set(labels.tolist())),
        'pseudo_label_agreement':sum(row['correct'] for row in predictions)/len(predictions),
        'scope':'Same-track held-out crops with teacher/OCR pseudo-labels. Not independent identity accuracy; no unseen-team capability.',
        'predictions':predictions}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))


if __name__=='__main__':main()
