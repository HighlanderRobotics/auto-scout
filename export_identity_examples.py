"""Export reproducible crop-teaching proposals without images, secrets or reference truth.

Nothing in this export is automatically a trusted training label. Review against
source footage before using identities; a track can contain different robots.
"""
import argparse
import json
from pathlib import Path


def records(root):
    contract=json.loads((root/'contract.json').read_text())
    source=json.loads((root/'predictions.json').read_text())['source']
    x,y,_,_=source['views']['main']
    for path in sorted(root.glob('labels_*.json')):
        data=json.loads(path.read_text())
        for row,label in zip(data['plan']['rows'],data['labels']):
            a,b,c,d=row['box']
            yield dict(schema_version=1,match=source['match'],youtube_url=source['youtube_url'],
                video_sha256=contract['source_video_sha256'],video_s=data['plan']['video_s'],camera='main',
                full_frame_box=[a+x,b+y,c+x,d+y],current_roster=source['teams'],
                teacher_team=label['team'],teacher_kind=label['kind'],teacher_confidence=label['confidence'],
                bumper_text=label['bumper_text'],reason=label['reason'],teacher_model=data['teacher_model'],
                request_fingerprint=data['request_fingerprint'],gallery_sha256=contract['gallery_sha256'],
                review_status='unreviewed_teacher_proposal',source_run=str(root),split='development')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new export; preserve previous annotations')
    rows=[row for root in a.runs for row in records(root)]
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    print(f'Exported {len(rows)} teacher crop proposals; none marked reviewed')


if __name__=='__main__':main()
