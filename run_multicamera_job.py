"""Run the reviewed teaching/training job with durable stage status and no retries."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from teacher_labels import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Use the explicitly approved $4.90 shared budget')
    args=parser.parse_args()
    if not args.execute:
        parser.error('Paid teaching requires --execute and prior budget approval')
    root=Path('datasets/temporal_multicamera_v1_run')
    stages=[
        ('teaching',['temporal_multicamera.py','--stage','teach','--execute']),
        ('labeled_videos',['finish_multicamera.py','--stage','videos','--out','runs/temporal_multicamera_v1_teaching_videos']),
        ('appearance_training',['train_temporal_appearance.py','--out','runs/temporal_multicamera_v1_ocr_appearance']),
        ('detector_training_and_evaluation',['temporal_multicamera.py','--stage','train']),
        ('baseline_validation',['evaluate_detector.py','runs/local_robot_v1/weights/best.pt',str(root/'dataset'),
                                'runs/temporal_multicamera_v1_baseline_val','--split','val'])]
    stage='preflight'
    try:
        for stage,command in stages:
            atomic_json(root/'job_status.json',{'state':'running','stage':stage,'updated_unix':time.time()})
            print('Starting:',stage,flush=True)
            subprocess.run([sys.executable,'-u',*command],check=True)
        ledger=json.loads((root/'ledger.json').read_text())
        report={'run':'temporal_multicamera_v1','spent_usd':ledger['spent_usd'],'reserved_usd':ledger['reserved_usd'],
                'cap_usd':ledger['cap_usd'],'dataset':json.loads((root/'dataset/manifest.json').read_text())['summary'],
                'einstein_test':json.loads(Path('runs/temporal_multicamera_v1_test/metrics.json').read_text())['metrics'],
                'pnw_teacher_validation':json.loads(Path('runs/temporal_multicamera_v1_val/metrics.json').read_text())['metrics'],
                'baseline_pnw_teacher_validation':json.loads(Path('runs/temporal_multicamera_v1_baseline_val/metrics.json').read_text())['metrics'],
                'appearance':json.loads(Path('runs/temporal_multicamera_v1_ocr_appearance/metrics.json').read_text()),
                'limitations':'No independent side-camera answer key or end-to-end team identity score. PNW validation is agreement with teacher labels. No trained model is promoted automatically; no automatic camera fusion is enabled.'}
        atomic_json(root/'result.json',report)
        atomic_json(root/'job_status.json',{'state':'complete','updated_unix':time.time(),'result':str(root/'result.json')})
        print('Job complete:',root/'result.json',flush=True)
    except BaseException as error:
        atomic_json(root/'job_status.json',{'state':'stopped','stage':stage,'updated_unix':time.time(),'error_type':type(error).__name__})
        print('Job stopped at',stage,'without automatic retries:',type(error).__name__,flush=True)
        raise


if __name__=='__main__':
    main()
