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
from path_stitching import stitch_fragments, fill_short_gaps


def partial_team_matches(text, teams):
    """Return a team only when an OCR digit fragment uniquely identifies it."""
    digits = ''.join(ch for ch in str(text) if ch.isdigit())
    matches = [team for team in teams if len(digits) >= 2 and digits in team]
    return matches[0] if len(matches) == 1 else None


def supported_alliance(evidence, identifier):
    """Return a repeated, unopposed bumper-color observation for one track."""
    votes = defaultdict(set)
    for row in evidence:
        alliance = row.get('alliance')
        if row.get('id') == identifier and alliance in {'red', 'blue'} and row.get('confidence', 0) >= .7:
            votes[alliance].add(row['t'])
    counts = sorted(((len(times), alliance) for alliance, times in votes.items()), reverse=True)
    if counts and counts[0][0] >= 2 and (len(counts) == 1 or counts[1][0] < 2):
        return counts[0][1]
    return None


def can_join(first, second, max_gap_s=1.5, max_distance=9):
    """Use recent path velocity to reconnect a short lost tracker fragment.

    This only answers whether the second fragment could be the first robot after a
    brief occlusion. The caller still requires matching repeated bumper color, so a
    nearby opponent or same-alliance robot cannot inherit a name from path motion
    alone.
    """
    a, b = first['points'][-1], second['points'][0]
    gap = b['t'] - a['t']
    if not 0 < gap <= max_gap_s:
        return False
    predicted_x, predicted_y = a['x'], a['y']
    if len(first['points']) >= 2:
        prior = first['points'][-2]
        elapsed = a['t'] - prior['t']
        if elapsed > 0:
            predicted_x += (a['x'] - prior['x']) / elapsed * gap
            predicted_y += (a['y'] - prior['y']) / elapsed * gap
    return math.hypot(predicted_x - b['x'], predicted_y - b['y']) <= max_distance


def resolve_tracks(data, evidence, teams, alliance_by_team=None):
    """Resolve OCR anchors, then add only safe free evidence around those anchors."""
    votes = defaultdict(lambda: defaultdict(set))
    for row in evidence:
        if row.get('confidence', 0) < .8:
            continue
        exact = str(row.get('team', ''))
        team = exact if exact in teams else partial_team_matches(exact, teams)
        if team:
            votes[int(row['id'])][team].add(row['t'])
    assignments = {}
    audit = []
    blocked=set()
    for track in data['tracks']:
        counts = sorted(((len(times), team) for team, times in votes[track['id']].items()), reverse=True)
        if not counts or counts[0][0] < 2:
            continue
        # A second supported number signals a potential tracker identity swap.
        if len(counts) > 1 and counts[1][0] >= 2:
            audit.append({'id':track['id'], 'reason':'conflicting bumper reads', 'votes':counts})
            blocked.add(track['id'])
            continue
        assignments[track['id']] = counts[0][1]
        audit.append({'id':track['id'], 'team':counts[0][1], 'reason':'repeated identity evidence', 'votes':counts,
                      'methods':sorted({row.get('method','local_ocr') for row in evidence if row.get('id')==track['id'] and row.get('team')==counts[0][1]})})

    if alliance_by_team is None:
        observed={}
        for identifier,team in assignments.items():
            alliance=supported_alliance(evidence,identifier)
            if alliance:observed.setdefault(team,set()).add(alliance)
        alliance_by_team={team:next(iter(colors)) for team,colors in observed.items() if len(colors)==1}
    if alliance_by_team:
        by_id = {track['id']: track for track in data['tracks']}
        colors = {identifier: supported_alliance(evidence, identifier) for identifier in by_id}
        for identifier,team in assignments.items():
            expected=alliance_by_team.get(team)
            if colors.get(identifier) and colors[identifier]!=expected:
                blocked.add(identifier)
            elif expected:
                colors[identifier]=expected
        chains,links=stitch_fragments(data['tracks'],colors,assignments,blocked)
        audit.extend(links)
        for chain in chains:
            names={assignments[i] for i in chain if i in assignments}
            if len(names)==1:
                team=next(iter(names))
                for identifier in chain:
                    if identifier not in assignments:
                        assignments[identifier]=team
                        audit.append({'id':identifier,'team':team,'reason':'nearby same-color fragment joined','chain':chain})
        for alliance in ('red', 'blue'):
            candidates = sorted(team for team, value in alliance_by_team.items() if value == alliance)
            if len(candidates)!=3:continue
            remaining = [team for team in candidates if team not in assignments.values()]
            unknown = [chain for chain in chains if all(identifier not in assignments and identifier not in blocked and colors.get(identifier)==alliance for identifier in chain)]
            if len(remaining) == 1 and len(unknown) == 1:
                # The other two robots must actually be present at the same time.
                # Otherwise a later fragment of an already-named robot could be
                # mistaken for the missing third teammate.
                support=set()
                for identifier in unknown[0]:
                    for point in by_id[identifier]['points']:
                        present={}
                        for other,team in assignments.items():
                            if alliance_by_team.get(team)!=alliance:continue
                            for known in by_id[other]['points']:
                                if abs(known['t']-point['t'])<.051:
                                    present[team]=known
                        if len(present)==2 and all(math.hypot(p['x']-point['x'],p['y']-point['y'])>=8 for p in present.values()):
                            support.add(point['t'])
                if len(support)<2:continue
                for identifier in unknown[0]:
                    assignments[identifier] = remaining[0]
                    audit.append({'id':identifier, 'team':remaining[0], 'reason':'only remaining team in alliance after repeated bumper-color evidence','chain':unknown[0]})
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
                       'points':fill_short_gaps(sorted(points.values(), key=lambda p:p['t']),data['fps']) if 'fps' in data else sorted(points.values(),key=lambda p:p['t'])})
    return output, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', type=Path)
    p.add_argument('--video', type=Path, required=True, help='Original full-frame source video, not tracking.mp4')
    p.add_argument('--teams', required=True, help='Comma-separated match roster; exact bumper reads only')
    p.add_argument('--red-teams', help='Comma-separated red-alliance roster; enables local bumper-color evidence')
    p.add_argument('--blue-teams', help='Comma-separated blue-alliance roster; enables local bumper-color evidence')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--evidence', type=Path, help='Reuse an existing OCR evidence.json without invoking Vision')
    a = p.parse_args()
    teams = set(a.teams.split(','))
    if not teams or any(not t.isdigit() or not 1 <= len(t) <= 5 for t in teams):
        p.error('Teams must be comma-separated team numbers')
    if bool(a.red_teams) != bool(a.blue_teams):
        p.error('Provide both --red-teams and --blue-teams, or neither')
    alliance_by_team = None
    if a.red_teams:
        red = set(a.red_teams.split(',')); blue = set(a.blue_teams.split(','))
        if red | blue != teams or red & blue or len(red) != 3 or len(blue) != 3:
            p.error('Red and blue rosters must be three distinct teams each and match --teams')
        alliance_by_team = {team: 'red' for team in red} | {team: 'blue' for team in blue}
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
            for name, row in manifest.items():
                image = cv2.imread(str(crops/name))
                hsv = cv2.cvtColor(image[image.shape[0]//2:], cv2.COLOR_BGR2HSV)
                red = int((((hsv[:,:,0] <= 12) | (hsv[:,:,0] >= 170)) & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 60)).sum())
                blue = int(((hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 130) & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 60)).sum())
                total = red + blue
                if total and max(red, blue) / total >= .75:
                    evidence.append(dict(row, team='', alliance='red' if red > blue else 'blue', confidence=max(red, blue) / total))
            (a.out/'crop_manifest.json').write_text(json.dumps(manifest,indent=2))
        (a.out/'evidence.json').write_text(json.dumps(evidence,indent=2))
        tracks,audit=resolve_tracks(data,evidence,teams,alliance_by_team)
        debug=a.out/'debug';debug.mkdir()
        for name in ('tracking.mp4','calibration.json','calibration_preview.jpg','bytetrack.yaml'):
            source=a.paths.parent/name
            if source.exists():
                shutil.copyfile(source,debug/name);shutil.copyfile(source,a.out/name)
        render_animation(copy.deepcopy(data),debug)
        named=copy.deepcopy(data)
        named.update(tracks=tracks,identity_mode='team',identity_audit=audit,roster=sorted(teams),
                     identity_note='One joined path per resolved team using repeated identity evidence, matching bumper color and unambiguous motion. Dashed lines and hollow markers show short estimated gaps. Long gaps remain unknown; anonymous fragments remain in debug.')
        named['summary']['shown_track_ids']=len(tracks)
        named['summary']['unresolved_track_ids']=len(data['tracks'])-sum(len(t['source_ids']) for t in tracks)
        named['summary']['estimated_path_points']=sum(bool(p.get('estimated')) for t in tracks for p in t['points'])
        render_animation(named,a.out)
        print(f"Resolved {len(tracks)}/{len(teams)} teams. Open {a.out/'animation.html'}; anonymous debug: {debug/'animation.html'}",flush=True)
    finally:
        stop.set()


if __name__=='__main__': main()
