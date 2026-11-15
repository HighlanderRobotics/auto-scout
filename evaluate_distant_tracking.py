"""Score the visible SF robots in saved tracking output against reviewed floor-crop boxes."""
import argparse,json
from pathlib import Path
from evaluate_detector import count_matches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Choose new output')
    data=json.loads(a.paths.read_text())
    review=json.loads(Path('configs/sf_distant_reviewed_boxes.json').read_text())
    rows=[]
    for time,boxes in review['frames'].items():
        frame=min(data['frames'],key=lambda f:abs(f['video_s']-float(time)))
        if abs(frame['video_s']-float(time))>.051:continue
        preds=[dict(box=r['box'],class_id=0) for r in frame['robots']]
        refs=[dict(box=[x1,y1+300,x2,y2+300],class_id=0) for x1,y1,x2,y2 in boxes]
        farrefs=[r for r in refs if (r['box'][1]+r['box'][3])/2<510]
        farpreds=[r for r in preds if (r['box'][1]+r['box'][3])/2<510]
        rows.append({'video_s':float(time),'split':'held_out' if time in ['22','24'] else 'training_time',
                     'all':count_matches(preds,refs,.5),'far':count_matches(farpreds,farrefs,.5)})
    report={'scope':review['scope'],'iou':.5,'far_definition':'box center above source y=510 in this fixed SF camera','frames':rows}
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
