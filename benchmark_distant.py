"""Compare fixed tiled recipes on validation only before testing the selected recipe."""
import argparse,json
from pathlib import Path
import cv2
import auto_paths  # local runtime configuration
from distant_detector import predict_robots
from evaluate_detector import count_matches,read_labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',default='val',choices=['val','test'])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    from ultralytics import YOLO
    model=YOLO('runs/local_robot_v1/weights/best.pt')
    root=Path('datasets/checked_clip_v1')
    reports=[]
    recipes=[('baseline',1.,1.01),('tiles65',.65,.4),('tiles80',.8,.4)] if a.split=='val' else [('tiles65',.65,.4)]
    for name,scale,threshold in recipes:
        totals={'all':dict(true_positive=0,false_positive=0,false_negative=0),'far':dict(true_positive=0,false_positive=0,false_negative=0)}
        records=[]
        for file in sorted((root/'images'/a.split).glob('*.jpg')):
            image=cv2.imread(str(file));h,w=image.shape[:2]
            refs=read_labels(root/'labels'/a.split/(file.stem+'.txt'),w,h)
            preds=predict_robots(model,image,scale=scale,tile_confidence=threshold)
            for group in totals:
                select=lambda rows:rows if group=='all' else [r for r in rows if (r['box'][1]+r['box'][3])/2<h*.5]
                counts=count_matches(select(preds),select(refs),.5)
                for key in counts:totals[group][key]+=counts[key]
            records.append({'image':file.name,'predictions':preds,'references':refs})
            print(name,file.name,flush=True)
        report={'recipe':name,'scale':scale,'tile_confidence':threshold,'split':a.split,'metrics':totals,'frames':records}
        reports.append(report);print(name,totals,flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise ValueError('Choose a new report path')
    a.out.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
