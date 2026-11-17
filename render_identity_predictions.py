"""Render observed identity predictions side by side with a frozen local baseline."""
import argparse
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from evaluate_identity_checkpoints import load_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fps',type=float,default=5);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new video filename')
    source,frames=load_predictions(a.predictions);_,baseline=load_predictions(a.baseline)
    if len(frames)>1:
        steps=[b-a for a,b in zip(sorted(frames),sorted(frames)[1:])]
        if max(steps)!=min(steps):print('Irregularly sampled diagnostic: displayed frames are evenly spaced in the video, timestamps show actual source times',flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(a.out),(1920,440),fps=a.fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,ffmpeg_log_level='error');next(writer)
    cap=cv2.VideoCapture(source['video']);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',24)
    try:
        for index,t in enumerate(sorted(frames)):
            cap.set(cv2.CAP_PROP_POS_MSEC,t);ok,bgr=cap.read()
            if not ok:raise ValueError('Missing source frame')
            x,y,x2,y2=source['views']['main'];original=Image.fromarray(cv2.cvtColor(bgr[y:y2,x:x2],cv2.COLOR_BGR2RGB))
            sheet=Image.new('RGB',(1920,440),'#161b22');draw=ImageDraw.Draw(sheet)
            for side,(title,rows) in enumerate([('Frozen local baseline',baseline.get(t,[])),('Candidate | observed boxes only',frames[t])]):
                image=original.copy();d=ImageDraw.Draw(image)
                for r in rows:
                    team=r.get('team');color='lime' if team else '#aaaaaa';d.rectangle(r['box'],outline=color,width=4)
                    bx,by=r['box'][:2];label=str(team) if team else '? '+r.get('track_key','')
                    d.text((bx,max(0,by-28)),label,font=font,fill=color,stroke_fill='black',stroke_width=2)
                sheet.paste(image.resize((960,250)),(side*960,60));draw.text((side*960+10,20),title,font=font,fill='white')
            draw.text((10,350),f"{source['match']} | video {t/1000:.1f}s | green = accepted proposal, gray = unresolved",font=font,fill='white')
            draw.text((10,395),'Experimental identities require review. No hidden positions or full-match accuracy claim.',font=font,fill='white')
            writer.send(np.asarray(sheet))
            if index==len(frames)//2:sheet.save(a.out.with_suffix('.jpg'))
            if index%25==0:print(f'Rendered {index+1}/{len(frames)}',flush=True)
    finally:cap.release();writer.close()


if __name__=='__main__':main()
