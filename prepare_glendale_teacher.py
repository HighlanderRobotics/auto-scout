"""Additional full-field Glendale labels under the user's shared approved cap."""
import argparse,json,threading
from pathlib import Path
import cv2
from teacher_labels import Teacher
from auto_paths import heartbeat


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Make paid requests after cost and cap approval')
    args=parser.parse_args()
    if not args.execute:parser.error('Paid labeling requires approved --execute')
    ledger=json.loads(Path('datasets/identity_coach_v1/ledger.json').read_text())
    if not 0<ledger['cap_usd']<=4:raise ValueError('Unexpected approved cap')
    teacher=Teacher('datasets/identity_coach_v1',cap=ledger['cap_usd'])
    root=Path('datasets/glendale_teacher_v1');root.mkdir(exist_ok=True)
    cap=cv2.VideoCapture('datasets/scout_VZCS8a4rRn0/source.mp4')
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        for t in [8,10,12,14,16,18,20,22,24,25,26,27]:
            image=root/f'g_{t}.jpg';label=root/f'g_{t}.json'
            if not image.exists():
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
                if not ok:raise ValueError('Cannot read frame')
                cv2.imwrite(str(image),frame[300:620])
            result=teacher.label(image,label)
            print(t,len(result['robots']),[r['confidence'] for r in result['robots']],flush=True)
    finally:cap.release();stop.set()


if __name__=='__main__':main()
