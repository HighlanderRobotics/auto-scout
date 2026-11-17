"""Select varied high-evidence teacher crops for visual review, never auto-approve labels.

One best full bumper read per team / two-second bucket, plus one person and one
field-equipment example per bucket. All candidates remain unreviewed.
"""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps

from teacher_labels import atomic_json


def select(rows):
    chosen={}
    for index,r in enumerate(rows):
        positive=r['teacher_kind']=='robot' and r['teacher_confidence']>=.8 and r['teacher_team']==r['bumper_text']
        negative=r['teacher_kind'] in ['person','field_equipment'] and r['teacher_confidence']>=.85
        if not(positive or negative):continue
        label=r['teacher_team'] if positive else 'unknown';category=label if positive else r['teacher_kind']
        key=(r['match'],r['camera'],int(r['video_s']//2),category)
        a,b,c,d=r['full_frame_box'];rank=(r['teacher_confidence'],(c-a)*(d-b))
        if key not in chosen or rank>chosen[key][0]:chosen[key]=(rank,dict(r,label=label,input_index=index))
    return sorted((row for _,row in chosen.values()),key=lambda r:(r['match'],r['video_s'],r['input_index']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('proposals',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new review folder')
    rows=[json.loads(s) for path in a.proposals for s in path.read_text().splitlines()];chosen=select(rows);a.out.mkdir(parents=True)
    atomic_json(a.out/'candidates.json',chosen)
    plans={s['match']:s for path in ['configs/focus_MQZ4Yrw86uQ.json','configs/norcal_identity_pool_v1.json'] for s in json.loads(Path(path).read_text())['sources']}
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',18);captures={};frames={}
    try:
        for page in range((len(chosen)+15)//16):
            sheet=Image.new('RGB',(1280,1120),'#161b22');draw=ImageDraw.Draw(sheet)
            for j,r in enumerate(chosen[page*16:page*16+16]):
                key=(r['match'],r['video_s']);video=plans[r['match']]['video']
                if key not in frames:
                    cap=captures.setdefault(video,cv2.VideoCapture(video));cap.set(cv2.CAP_PROP_POS_MSEC,r['video_s']*1000);ok,bgr=cap.read()
                    if not ok:raise ValueError('Missing review frame')
                    frames.clear();frames[key]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                image=frames[key].crop(tuple(round(v) for v in r['full_frame_box']));tile=ImageOps.contain(image,(310,215));x=j%4*320;y=j//4*280;sheet.paste(tile,(x,y+60))
                draw.text((x+4,y+4),f"candidate {page*16+j} | {r['label']}",font=font,fill='white');draw.text((x+4,y+29),f"{r['match']} {r['video_s']}s",font=font,fill='white')
            sheet.save(a.out/f'review_{page:02d}.jpg',quality=95);print(f'Review sheet {page+1}',flush=True)
    finally:
        for cap in captures.values():cap.release()
    print(f'{len(chosen)} unreviewed candidates; no API calls or label approval',flush=True)


if __name__=='__main__':main()
