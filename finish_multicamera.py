"""Offline labeled teaching videos. Appearance fitting uses train_temporal_appearance.py."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def videos(root, out):
    import imageio_ffmpeg
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'frames.json').read_text());fps=json.loads((root/'plan.json').read_text())['fps']
    groups={}
    for r in manifest['frames']:
        groups.setdefault((r['match'],r['camera']),[]).append(r)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
    for (match,camera),records in groups.items():
        frames=out/(match+'_'+camera);frames.mkdir()
        for r in records:
            with Image.open(root/r['image']) as source:
                image=source.convert('RGB')
            draw=ImageDraw.Draw(image)
            label=root/'labels'/match/camera/(Path(r['image']).stem+'.json')
            model='No usable teacher label'
            if label.exists():
                d=json.loads(label.read_text());model=d['teacher_model']
                for robot in d['robots']:
                    if not robot['visible']:
                        continue
                    certain=robot['confidence']>=.8
                    color='lime' if certain else 'yellow'
                    draw.rectangle(robot['box'],outline=color,width=3)
                    x,y=robot['box'][:2]
                    draw.text((x,max(26,y-20)),f"{robot['team']}{'' if certain else '?'} ID {robot['confidence']:.2f}",
                              fill=color,font=font,stroke_width=1,stroke_fill='black')
            draw.rectangle((0,0,image.width,25),fill='black')
            draw.text((4,3),f'{match} {camera} t={r["video_s"]:.1f}s | {model} | TEACHER PROPOSAL',fill='white',font=font)
            image.save(frames/f"f_{r['index']:04d}.jpg",quality=90)
        path=out/(match+'_'+camera+'.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate',str(fps),'-i',str(frames/'f_%04d.jpg'),
                        '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
        print('Labeled video:',path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('datasets/temporal_multicamera_v1_run'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=('videos',),required=True)
    args=p.parse_args()
    videos(args.root,args.out)


if __name__=='__main__':
    main()
