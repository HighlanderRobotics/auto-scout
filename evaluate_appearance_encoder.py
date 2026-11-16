"""One predeclared local ResNet18 appearance comparison on the frozen OCR split.

This downloads public torchvision weights if absent, with no paid API requests.
No weight tuning or model selection uses the held-out crops.
"""
import argparse
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from teacher_labels import atomic_json
from train_detector import heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gallery',type=Path,default=Path('runs/temporal_multicamera_v1_ocr_appearance'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.gallery/'metrics.json').read_text());teams=set(report['teams'])
    samples=[s for s in report['accepted_crops'] if s['team'] in teams]
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        import torch
        from torchvision.models import resnet18,ResNet18_Weights
        weights=ResNet18_Weights.IMAGENET1K_V1
        model=resnet18(weights=weights);model.fc=torch.nn.Identity();model.eval().to('mps')
        transform=weights.transforms();vectors=[]
        with torch.inference_mode():
            for start in range(0,len(samples),16):
                rows=[]
                for s in samples[start:start+16]:
                    with Image.open(args.gallery/'crops'/s['file']) as image:
                        rows.append(transform(image.convert('RGB')))
                batch=torch.stack(rows).to('mps');features=model(batch).cpu().numpy()
                features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
                vectors.extend(features)
                print(f'Encoded {min(start+16,len(samples))}/{len(samples)} crops',flush=True)
        vectors=np.stack(vectors);train=[i for i,s in enumerate(samples) if s['split']=='train'];test=[i for i,s in enumerate(samples) if s['split']=='test']
        predictions=[]
        for index in test:
            choice=train[int(np.argmax(vectors[train]@vectors[index]))]
            predictions.append(dict(file=samples[index]['file'],team=samples[index]['team'],prediction=samples[choice]['team'],
                                    correct=samples[index]['team']==samples[choice]['team']))
        correct=sum(s['correct'] for s in predictions)
        result=dict(encoder='torchvision ResNet18 IMAGENET1K_V1, frozen pooled features, cosine nearest gallery',
            training_crops=len(train),test_crops=len(test),correct_pseudo_labels=correct,agreement=correct/len(test),
            hsv_agreement=report['unweighted_agreement'],scope=report['scope'],
            verdict='Experimental comparison only; no default identity model changed.',predictions=predictions)
        atomic_json(args.out/'metrics.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2),flush=True)
    finally:stop.set()


if __name__=='__main__':main()
