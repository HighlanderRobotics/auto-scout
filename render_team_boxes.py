"""Render predicted team boxes directly on source video, without inference or API calls."""
import argparse,json
from pathlib import Path

import cv2
import imageio_ffmpeg


def render_team_boxes(data,video,out):
    names={identifier:str(track['team']) for track in data['tracks'] if 'team' in track for identifier in track.get('source_ids',[track['id']])}
    colors={str(track['team']):tuple(track['color']) for track in data['tracks'] if 'team' in track}
    x1,y1,x2,y2=data['calibration']['crop']
    width=1280;height=round((y2-y1)*width/(x2-x1)/2)*2
    cap=cv2.VideoCapture(str(video));writer=imageio_ffmpeg.write_frames(str(out),(width,height),fps=data['fps'],codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,ffmpeg_log_level='error');next(writer)
    try:
        for index,frame in enumerate(data['frames']):
            cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000);ok,image=cap.read()
            if not ok:raise ValueError('Cannot read source frame for team overlay')
            for robot in frame['robots']:
                if robot.get('estimated') or robot.get('visible') is False:continue
                team=names.get(robot['id']);label=team or 'Unknown'
                rgb=colors.get(team,(220,220,220));bgr=tuple(reversed(rgb))
                left,top,right,bottom=map(round,robot['box'])
                cv2.rectangle(image,(left,top),(right,bottom),bgr,3)
                font=cv2.FONT_HERSHEY_SIMPLEX;scale=.8;thickness=2
                (tw,th),baseline=cv2.getTextSize(label,font,scale,thickness)
                tx=max(x1,min(left,x2-tw-8));ty=max(y1+th+6,min(top-6,y2-baseline-2))
                cv2.rectangle(image,(tx,ty-th-5),(tx+tw+7,ty+baseline+2),(15,20,30),-1)
                cv2.putText(image,label,(tx+3,ty),font,scale,bgr,thickness,cv2.LINE_AA)
            writer.send(cv2.cvtColor(cv2.resize(image[y1:y2,x1:x2],(width,height)),cv2.COLOR_BGR2RGB))
            if index%50==0:print(f'Team video {index+1}/{len(data["frames"])} frames; no API calls',flush=True)
    finally:cap.release();writer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',type=Path);p.add_argument('--video',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('Choose a new video output path')
    render_team_boxes(json.loads(a.paths.read_text()),a.video,a.out)


if __name__=='__main__':main()
