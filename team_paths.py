"""Resolve saved anonymous tracks using repeated local bumper OCR; preserve debug artifacts."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import cv2
from auto_paths import color, heartbeat, render_animation, youtube_id


def resolve_tracks(data, evidence, teams):
    """Require repeated independent crop reads and reject competing identities/overlap."""
    votes = defaultdict(lambda: defaultdict(set))
    for row in evidence:
        if row['team'] in teams and row['confidence'] >= .8:
            votes[int(row['id'])][row['team']].add(row['t'])
    assignments = {}
    audit = []
    for track in data['tracks']:
        counts = sorted(((len(times), team) for team, times in votes[track['id']].items()), reverse=True)
        if not counts or counts[0][0] < 2:
            continue
        # A second supported number signals a potential tracker identity swap.
        if len(counts) > 1 and counts[1][0] >= 2:
            audit.append({'id':track['id'], 'reason':'conflicting bumper reads', 'votes':counts})
            continue
        assignments[track['id']] = counts[0][1]
    groups = defaultdict(list)
    for track in data['tracks']:
        if track['id'] in assignments:
            groups[assignments[track['id']]].append(track)
    output = []
    for team, tracks in sorted(groups.items()):
        points = {}
        overlap = False
        for track in tracks:
            for point in track['points']:
                if point['t'] in points:
                    overlap = True
                points[point['t']] = dict(point, source_id=track['id'])
        if overlap:
            audit.append({'team':team, 'reason':'overlapping tracks claim the same team', 'ids':[t['id'] for t in tracks]})
            continue
        output.append({'id':team, 'team':team, 'color':list(color(int(team))),
                       'source_ids':[t['id'] for t in tracks],
                       'points':sorted(points.values(), key=lambda p:p['t'])})
    return output, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', type=Path)
    p.add_argument('--video', type=Path, required=True, help='Original full-frame source video, not tracking.mp4')
    p.add_argument('--teams', required=True, help='Comma-separated match roster; exact bumper reads only')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--evidence', type=Path, help='Reuse an existing OCR evidence.json without invoking Vision')
    a = p.parse_args()
    teams = set(a.teams.split(','))
    if not teams or any(not t.isdigit() or not 1 <= len(t) <= 5 for t in teams):
        p.error('Teams must be comma-separated team numbers')
    if a.out.exists():
        p.error('Choose a new output folder')
    if not a.evidence and sys.platform != 'darwin':
        p.error('Local OCR currently requires macOS; use --evidence for saved evidence on other platforms')
    data = json.loads(a.paths.read_text())
    a.out.mkdir(parents=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        if a.evidence:
            evidence = json.loads(a.evidence.read_text())
        else:
            crops = a.out/'ocr_crops'; crops.mkdir()
            manifest = {}; last = {}; cap = cv2.VideoCapture(str(a.video))
            if not cap.isOpened():
                raise ValueError('Cannot open original video')
            try:
                for frame in data['frames']:
                    candidates = [r for r in frame['robots'] if frame['t']-last.get(r['id'],-100) >= .5]
                    if not candidates: continue
                    cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000)
                    ok, image = cap.read()
                    if not ok: raise ValueError('Cannot read source frame')
                    height,width=image.shape[:2]
                    if [width,height] != data['calibration']['image_size']:
                        raise ValueError('Video size differs from original full-frame calibration')
                    for r in candidates:
                        x1,y1,x2,y2=map(round,r['box'])
                        crop=image[max(0,y1):min(height,y2),max(0,x1):min(width,x2)]
                        if not crop.size: continue
                        name=f"id{r['id']}_t{round(frame['t']*1000):06d}.jpg"
                        scale=min(4,1000/max(crop.shape[:2]))
                        cv2.imwrite(str(crops/name),cv2.resize(crop,None,fx=scale,fy=scale))
                        manifest[name]={'id':r['id'],'t':frame['t']};last[r['id']]=frame['t']
            finally:
                cap.release()
            print(f'Reading {len(manifest)} robot crops locally; no API calls',flush=True)
            cache=Path('.venv/swift-module-cache').resolve();cache.mkdir(parents=True,exist_ok=True)
            result=subprocess.run(['swift','-module-cache-path',str(cache),str(Path(__file__).with_name('read_team_numbers.swift')),str(crops.resolve())],capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError('Local Vision OCR failed: '+result.stderr[-2000:])
            evidence=[dict(row,**manifest[row['file']]) for row in json.loads(result.stdout)]
            (a.out/'crop_manifest.json').write_text(json.dumps(manifest,indent=2))
        (a.out/'evidence.json').write_text(json.dumps(evidence,indent=2))
        tracks,audit=resolve_tracks(data,evidence,teams)
        debug=a.out/'debug';debug.mkdir()
        for name in ('tracking.mp4','calibration.json','calibration_preview.jpg','bytetrack.yaml'):
            source=a.paths.parent/name
            if source.exists():
                shutil.copyfile(source,debug/name);shutil.copyfile(source,a.out/name)
        render_animation(copy.deepcopy(data),debug)
        named=copy.deepcopy(data)
        named.update(tracks=tracks,identity_mode='team',identity_audit=audit,roster=sorted(teams),
                     identity_note='Repeated local OCR anchors; unresolved identities remain in debug. Short dashed joins are estimated. Long gaps remain unknown.')
        named['summary']['shown_track_ids']=len(tracks)
        named['summary']['unresolved_track_ids']=len(data['tracks'])-sum(len(t['source_ids']) for t in tracks)
        render_animation(named,a.out)
        print(f"Resolved {len(tracks)}/{len(teams)} teams. Open {a.out/'animation.html'}; anonymous debug: {debug/'animation.html'}",flush=True)
    finally:
        stop.set()


if __name__=='__main__': main()
