"""Haiku full-frame coaching for the saved SF camera, within the existing $2 ledger."""
import argparse,json,shutil,threading
from pathlib import Path
import cv2
from auto_paths import heartbeat
from teacher_labels import Teacher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    root=Path('datasets/distant_sf_native_labels_v1');root.mkdir(exist_ok=True)
    if not a.execute:raise ValueError('Requires approved --execute')
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    teacher=Teacher('datasets/identity_coach_v1',cap=2)
    cap=cv2.VideoCapture('datasets/distant_sf_v1/native_source.mp4')
    try:
        for t in [8,10,12,14,16,18,22,24]:
            file=root/f'sf_{t}.jpg'
            if not file.exists():
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,image=cap.read();assert ok
                cv2.imwrite(str(file),image[300:720])
            label=root/f'sf_{t}.json'
            result=teacher.label(file,label)
            print(t,len(result['robots']),[(r['confidence']) for r in result['robots']],flush=True)
    finally:
        cap.release();stop.set()


if __name__=='__main__':main()
