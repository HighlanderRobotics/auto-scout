"""Human shooting labels and conservative alliance-fuel reconciliation; no API calls."""
import argparse,json,os
from pathlib import Path
from teacher_labels import atomic_json


def reconcile(increments,events,teams):
    totals={team:0 for team in teams};unassigned={'red':0,'blue':0};rows=[]
    proposed=[]
    for inc in increments:
        if inc['fuel_delta']==0:continue
        candidates=[]
        for e in events:
            if not e.get('reviewed') or not e.get('counters_reviewed') or not e.get('exclusive_source'):continue
            if e.get('action')!='shooting' or teams.get(e.get('team'))!=inc['alliance']:continue
            # Human exclusivity assertion must cover the entire delayed-score interval.
            if e['start_s']-1<=inc['start_s'] and inc['end_s']<=e['end_s']+3:candidates.append(e)
        candidate=candidates[0] if len(candidates)==1 else None
        proposed.append((inc,candidate))
    by_event={}
    for inc,event in proposed:
        if event:by_event[event['id']]=by_event.get(event['id'],0)+inc['fuel_delta']
    blocked={e['id'] for e in events if e.get('attempts') is not None and by_event.get(e['id'],0)>e['attempts']}
    for inc,candidate in proposed:
        team=candidate['team'] if candidate and candidate['id'] not in blocked else None
        if team:totals[team]+=inc['fuel_delta']
        else:unassigned[inc['alliance']]+=inc['fuel_delta']
        rows.append(dict(inc,attributed_team=team,status='human_attributed_alliance_delta' if team else 'unassigned'))
    return dict(attributed_fuel_subtotals=totals,per_robot_total_made={team:None for team in teams},unassigned_observed_fuel=unassigned,intervals=rows,
                limitations='Only observed valid counter intervals included; missing OCR is not zero. Human-confirmed exclusivity includes delayed balls and human-player contributions. This is attributed alliance fuel, not a direct count of robot makes. No normalization or equal splitting.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--reviews',type=Path);p.add_argument('--out',type=Path);p.add_argument('--model',type=Path);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());scoreboard=json.loads((a.run/'scoreboard.json').read_text());examples=manifest['examples'];labels={}
    for e in examples:
        path=a.run/'labels'/f"{e['id']}.json"
        if path.exists():labels[e['id']]=json.loads(path.read_text())
    if a.reviews:
        if not a.out or a.out.exists():p.error('Choose a new output file')
        reviews=json.loads(a.reviews.read_text())
        if reviews['source_video_sha256']!=manifest['source_video_sha256']:raise ValueError('Review belongs to another source video')
        events=reviews['events'];ids={e['id'] for e in examples}
        for e in reviews.get('new_examples',[]):
            if not e['id'].startswith('custom_') or e['id'] in ids:raise ValueError('Invalid custom example ID')
            ids.add(e['id'])
        if len({e['id'] for e in events})!=len(events) or any(e['id'] not in ids for e in events):raise ValueError('Invalid/duplicate reviewed example IDs')
        for e in events:
            if e['team'] is not None and e['team'] not in manifest['config']['teams']:raise ValueError('Unknown roster team')
            if not 0<=e['start_s']<e['end_s']<=215:raise ValueError('Invalid event times')
            if e.get('attempts') is not None and (type(e['attempts'])!=int or e['attempts']<0):raise ValueError('Invalid attempt count')
            if e.get('attempts',0) and e['action']!='shooting':raise ValueError('Positive hub-shot count conflicts with a nonshooting label')
        from hybrid_shooting import score_increments
        for key,value in reviews.get('scoreboard_overrides',{}).items():
            t,alliance=key.split(':')
            if alliance not in ['blue','red'] or value is not None and (type(value)!=int or not 0<=value<=2000):raise ValueError('Invalid counter correction')
            row=next((r for r in scoreboard['rows'] if r['video_s']==int(t)),None)
            if row is None:raise ValueError('Counter correction outside source timeline')
            row[alliance]=[value,row[alliance][1] if row[alliance] else None] if value is not None else None
        atomic_json(a.out,reconcile(score_increments(scoreboard['rows']),events,manifest['config']['teams']));print(a.out);return
    source=a.run/'source.mp4'
    if not source.exists():source.symlink_to(Path(manifest['config']['video']).resolve())
    for e in examples:
        path=a.run/'labels_expert'/f"{e['id']}.json"
        if path.exists():labels[e['id']]=json.loads(path.read_text())
    for label in labels.values():
        if label.get('team') not in manifest['config']['teams'] or label.get('bumper_text','').strip()!=label.get('team'):label['team']='unknown'
    model=(a.model or a.run/'model_v1')/'predictions.json';student=json.loads(model.read_text()) if model.exists() else []
    payload=json.dumps(dict(manifest=manifest,labels=labels,student=student,scoreboard=scoreboard)).replace('<','\\u003c')
    template=Path('shooting_review.html').read_text();(a.run/'review.html').write_text(template.replace('__BUNDLE__',payload));print(a.run/'review.html')

if __name__=='__main__':main()
