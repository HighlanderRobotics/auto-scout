"""Suppress learned names contradicted by exact CURRENT-crop local bumper reads.

This post-training guard never creates a name or propagates a read along a track ID.
It is not independent validation of the trained head.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from teacher_labels import atomic_json


def veto(data,manifest,reads):
    teams=data['source']['teams'];votes={}
    for read in reads:
        meta=manifest.get(read['file'])
        if not meta or meta['camera']!='main' or read['confidence']<.9 or read['team'] not in teams:continue
        key=(round(meta['video_s']*1000),tuple(meta['box']));votes.setdefault(key,set()).add(read['team'])
    result=copy.deepcopy(data);changes=[]
    for frame in result['frames']:
        for row in frame['robots']:
            seen=votes.get((round(frame['video_s']*1000),tuple(row['box'])),set())
            if row['team'] and any(team!=row['team'] for team in seen):
                changes.append(dict(video_s=frame['video_s'],box=row['box'],vetoed_team=row['team'],current_bumper_reads=sorted(seen)))
                row.update(vetoed_team=row['team'],team=None,current_bumper_reads=sorted(seen),veto_reason='Contradictory exact current-crop local bumper read')
    result['identity_guard']='Post-training current-crop OCR contradiction veto; suppresses names only; not independent validation'
    return result,changes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path);p.add_argument('--ocr-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new output folder')
    data=json.loads(a.predictions.read_text());tracks=json.loads((a.ocr_run/'tracks.json').read_text())
    if tracks['source']!=data['source']:raise ValueError('OCR belongs to another source')
    manifest_path=a.ocr_run/'crop_manifest.json';reads_path=a.ocr_run/'ocr_raw.json'
    result,changes=veto(data,json.loads(manifest_path.read_text()),json.loads(reads_path.read_text()))
    a.out.mkdir(parents=True);atomic_json(a.out/'predictions.json',result)
    atomic_json(a.out/'vetoes.json',dict(changes=changes,predictions_sha256=hashlib.sha256(a.predictions.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),reads_sha256=hashlib.sha256(reads_path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(f'Suppressed {len(changes)} conflicting names; created zero names; no API calls')


if __name__=='__main__':main()
