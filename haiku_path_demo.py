"""Paid Haiku forward/backward tracking demo on the public Glendale comparison clip."""
import argparse,copy,fcntl,json,shutil,subprocess,sys,threading
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw

from auto_paths import load_calibration,field_position,render_animation,heartbeat
from teacher_labels import Teacher,InvalidTeacherLabel,atomic_json

TEAMS={'9772':'blue','9696':'blue','3328':'blue','5089':'red','11024':'red','3501':'red'}
SOURCE=Path('datasets/scout_VZCS8a4rRn0/source.mp4')
PROFILE=Path('configs/glendale_qual5_camera.json')


def render(root):
    import backward_pass
    backward_pass.TEAMS=TEAMS
    previous=sys.argv
    try:
        sys.argv=['backward_pass.py',str(root/'forward'),str(root/'backward')]
        backward_pass.main()
    finally:sys.argv=previous
    calibration,matrix=load_calibration(PROFILE)
    tracks={team:dict(id=team,team=team,source_ids=[team],color=[45,130,255] if alliance=='blue' else [245,70,85],points=[]) for team,alliance in TEAMS.items()}
    frames=[];cap=cv2.VideoCapture(str(SOURCE))
    writer=imageio_ffmpeg.write_frames(str(root/'tracking.mp4'),(1920,320),fps=5,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,ffmpeg_log_level='error');next(writer)
    font=__import__('auto_paths').font(24)
    try:
        for index in range(100):
            t=round(index/5,2);video_s=6+t;path=root/'backward'/f'f_{index:04d}.json'
            data=json.loads(path.read_text()) if path.exists() else {'robots':[]}
            cap.set(cv2.CAP_PROP_POS_MSEC,video_s*1000);ok,image=cap.read()
            if not ok:raise RuntimeError('Cannot read render frame')
            pane=Image.fromarray(cv2.cvtColor(image[300:620],cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pane);rows=[]
            for robot in data['robots']:
                team=robot['team'];position=field_position(robot['box'],matrix)
                if position is not None:
                    point=dict(t=t,x=position[0],y=position[1],confidence=robot['confidence'])
                    if not robot['visible']:point.update(estimated=True,visible=False)
                    tracks[team]['points'].append(point)
                if robot['visible']:
                    rows.append(dict(id=team,box=robot['box'],confidence=robot['confidence'],field=position))
                    x1,y1,x2,y2=robot['box'];rgb=tuple(tracks[team]['color'])
                    draw.rectangle((x1,y1-300,x2,y2-300),outline=rgb,width=3)
                    draw.text((x1,max(0,y1-330)),team,font=font,fill=rgb,stroke_width=1,stroke_fill='black')
            draw.text((10,2),f'Haiku | auto {t:.1f}s',font=font,fill='white',stroke_width=2,stroke_fill='black')
            writer.send(np.asarray(pane));frames.append(dict(t=t,video_s=video_s,robots=rows))
    finally:writer.close();cap.release()
    data=dict(source=str(SOURCE),fps=5,duration=20,auto_start_s=6,calibration=calibration,max_gap_s=.6,
        identity_mode='team',video_id_space='team',roster=list(TEAMS),identity_note='Haiku forward tracking plus backward identity voting. Names and boxes are AI predictions, not verified ground truth. Dashed points are hidden-position estimates. Same approximate calibration as the local comparison.',
        tracks=list(tracks.values()),frames=frames,summary={'frames':100,'shown_track_ids':6,'short_tracks_omitted':0,'unresolved_track_ids':0,'visible_observations':sum(len(f['robots']) for f in frames),'quarantined_frames':100-len(list((root/'forward').glob('f_*.json')))},
        limitations=['Haiku identities and hidden estimates may be wrong.','Approximate field mapping shared with local comparison; this does not measure position accuracy.'])
    shutil.copyfile(PROFILE,root/'calibration.json');render_animation(data,root)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(root/'tracking.mp4'),'-i',str(root/'paths.mp4'),
        '-filter_complex','[0:v]scale=960:160[top];[top][1:v]vstack=inputs=2[out]',
        '-map','[out]','-c:v','libx264','-pix_fmt','yuv420p','-an',str(root/'demo.mp4')],check=True)
    # This single held-out Glendale image is inside the same 20-second clip.
    # Its labels are agent-reviewed proposals, not independent human truth.
    from evaluate_detector import count_matches,read_labels
    labels=Path('datasets/glendale_detector_v1/labels/test/g_25_0.txt')
    sample=root/'backward/f_0095.json'
    if labels.exists() and sample.exists():
        robots=json.loads(sample.read_text())['robots']
        predictions=[dict(class_id=0,box=[r['box'][0],r['box'][1]-300,r['box'][2],r['box'][3]-300]) for r in robots if r['visible']]
        references=read_labels(labels,1920,320)
        atomic_json(root/'held_out_spot_check.json',dict(video_s=25,metrics={str(t):count_matches(predictions,references,t) for t in [.3,.5]},scope='One nearby same-camera test image with agent-reviewed labels. Measures boxes, not team identity accuracy.'))
    atomic_json(root/'status.json',dict(stage='complete',**data['summary']))
    print(f"Demo complete: {root/'animation.html'}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/glendale_haiku_demo_v1'))
    parser.add_argument('--execute',action='store_true');parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();root=args.out
    if args.render_only:render(root);return
    if not args.execute:parser.error('Paid run requires cost approval and --execute')
    root.mkdir(parents=True,exist_ok=True)
    for name in ['images','teacher','forward','quarantine']:(root/name).mkdir(exist_ok=True)
    ledger_root=Path('datasets/identity_coach_v1');lock=(ledger_root/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=json.loads((ledger_root/'ledger.json').read_text());teacher=Teacher(ledger_root,cap=ledger['cap_usd'])
    start_spend=teacher.ledger['spent_usd'];cap=cv2.VideoCapture(str(SOURCE));prior=None;previous=None;previous_index=None
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        for index in range(100):
            image=root/'images'/f'f_{index:04d}.jpg';output=root/'teacher'/f'f_{index:04d}.json'
            if not image.exists():
                cap.set(cv2.CAP_PROP_POS_MSEC,(6+index/5)*1000);ok,frame=cap.read()
                if not ok:raise RuntimeError('Cannot read source video')
                cv2.imwrite(str(image),frame[300:620])
            try:
                data=teacher.label(image,output,teams=TEAMS,previous=previous,prior=prior,gap=round((index-previous_index)/5,2) if previous_index is not None else .2)
            except InvalidTeacherLabel as error:
                atomic_json(root/'quarantine'/output.name,{'frame':index,'reason':str(error)})
                print(f'Quarantined frame {index}: {error}',flush=True);continue
            prior=data['robots'];previous=image;previous_index=index
            full=copy.deepcopy(data)
            for robot in full['robots']:robot['box'][1]+=300;robot['box'][3]+=300
            full.update(frame=image.name,time_s=round(6+index/5,2));atomic_json(root/'forward'/output.name,full)
            status=dict(stage='haiku_tracking',frames=index+1,total_frames=100,spent_usd=teacher.ledger['spent_usd'],cap_usd=teacher.ledger['cap_usd'])
            atomic_json(root/'status.json',status)
            print(f"Haiku {index+1}/100; {sum(r['visible'] for r in prior)} visible; shared spend ${teacher.ledger['spent_usd']:.5f}/${teacher.ledger['cap_usd']:.2f}",flush=True)
        atomic_json(root/'billing_summary.json',dict(run_spend_usd=teacher.ledger['spent_usd']-start_spend,shared_spend_usd=teacher.ledger['spent_usd'],reserved_usd=teacher.ledger['reserved_usd'],cap_usd=teacher.ledger['cap_usd']))
        render(root)
    except Exception as error:
        atomic_json(root/'status.json',dict(stage='stopped',error_type=type(error).__name__,reason=str(error)))
        raise
    finally:cap.release();stop.set()


if __name__=='__main__':main()
