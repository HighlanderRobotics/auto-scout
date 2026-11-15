"""Export existing Haiku annotation data only; no credentials, billing logs, images or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from teacher_labels import validate_labels

FRAME_FIELDS=('usable','notes','image','image_sha256','teacher_model','response_model','image_size','prompt_sha256')
ROBOT_FIELDS=('team','visible','box','confidence','named_by','bumper_text')


def secret_check(text):
    if re.search(r'sk-ant-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9_-]{16,}|CLAUDE_API_KEY\s*=',text):
        raise ValueError('Credential-like content detected; export aborted')


def sanitize_frame(data):
    result={key:data[key] for key in FRAME_FIELDS if key in data}
    if not str(result.get('teacher_model','')).startswith('claude-haiku'):
        raise ValueError('Not a provenance-tagged Haiku annotation')
    width,height=result['image_size']
    robots=[{key:robot[key] for key in ROBOT_FIELDS if key in robot} for robot in data['robots']]
    visible=[r for r in robots if r.get('visible',True)]
    validate_labels({'usable':bool(visible),'robots':visible},width,height)
    result['robots']=robots
    secret_check(json.dumps(result))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('annotations/haiku_training_v1'))
    a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new export directory')
    bundles=[]
    for name,plan_file in [('teacher_v2','configs/haiku_teacher_v2.json'),('teacher_multivenue_v1','configs/haiku_multivenue_v1.json')]:
        plan=json.loads(Path(plan_file).read_text())
        for source in plan['sources']:
            extraction=json.loads((Path('datasets')/name/'frames'/source['match']/'extraction.json').read_text())['contract']
            bundles.append((f"{name}/{source['match']}.jsonl",sorted((Path('datasets')/name/'labels'/source['match']).glob('f_*.json')),
                dict(match=source['match'],youtube_id=source['youtube_id'],tba_url=source['tba_url'],split=source['split'],teams=source['teams'],crop=extraction['crop'],fps=extraction['fps'],start_s=extraction['start_s'],video_sha256=extraction['video_sha256'],annotation_status='unreviewed teacher proposals; not all passed training filters')))
    bundles.append(('audit/einstein_tracking.jsonl',sorted(Path('datasets/teacher_v2/tracking_audit/labels').glob('f_*.json')),
        dict(youtube_id='EjF9we707DA',crop=[0,110,1920,720],fps=5,start_s=5.5,annotation_status='teacher tracking audit; not the human-checked answer key')))
    for folder in ['distant_sf_v1','distant_sf_native_labels_v1']:
        bundles.append((f'{folder}/frame_proposals.jsonl',sorted((Path('datasets')/folder).glob('sf_*.json')),
            dict(youtube_id='teZtwE8bQqU',crop=[0,75,854,320] if folder=='distant_sf_v1' else [0,300,1920,720],annotation_status='unreviewed proposals excluded from direct training; reviewed replacements in configs/sf_distant_reviewed_boxes.json',timing_note='sf_N is seconds into its downloaded local section; low-resolution section 12500..12530, native section 12515..12545; keyframe trimming can affect timing')))
    files=[];rejected=[]
    # Collect and validate everything before creating tracked output.
    payloads=[]
    for relative,paths,meta in bundles:
        records=[]
        for path in paths:
            try:
                record=sanitize_frame(json.loads(path.read_text()))
            except (ValueError,KeyError,TypeError) as error:
                if 'Credential' in str(error):raise
                rejected.append({'bundle':relative,'frame':path.name,'reason':type(error).__name__})
                continue
            record['frame_id']=path.stem
            if path.stem.startswith('f_'):
                record['video_time_s']=round(meta['start_s']+int(path.stem[2:])/meta['fps'],6)
            records.append(record)
        payloads.append((relative,records,meta))
    for round_name,root in [('round1',Path('datasets/identity_coach_v1')),('round2',Path('datasets/identity_coach_v1/round2'))]:
        records=[]
        for path in sorted(root.glob('id*_panel*.json')):
            data=json.loads(path.read_text());match=re.fullmatch(r'id(\d+)_panel(\d+)\.json',path.name)
            row={key:data[key] for key in ('team','confidence','reason') if key in data}
            row.update(track_id=int(match[1]),panel=int(match[2]))
            secret_check(json.dumps(row));records.append(row)
        payloads.append((f'identity_coaching/{round_name}.jsonl',records,dict(youtube_id='EjF9we707DA',teacher_model='claude-haiku-5-5',annotation_status='appearance proposals, including uncertain/null identities; not bumper reads or verified truth',coordinate_note='Track IDs refer to runs/auto_paths_youtube_demo/paths.json; panels are separate crop groups, not per-frame boxes')))
    for relative,records,meta in payloads:
        content=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in records)
        secret_check(content)
        payload=content.encode()
        files.append(dict(file=relative,records=len(records),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),**meta))
    manifest={'format':'auto-scout-haiku-annotations-v1','files':files,'total_records':sum(f['records'] for f in files),'rejected_invalid_annotations':rejected,'includes':'Normalized frame proposals and appearance proposals only. Raw response transcripts, credentials, billing, videos and images excluded.','coordinates':'Frame boxes are xyxy pixels in image_size and the per-bundle cropped image, not original full-frame pixels. Hidden robot boxes are previous estimates, not detections.'}
    secret_check(json.dumps(manifest))
    a.out.mkdir(parents=True)
    for (relative,records,_),info in zip(payloads,files):
        path=a.out/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in records))
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(files),'records':manifest['total_records'],'bytes':sum(f['bytes'] for f in files),'invalid_excluded':len(rejected),'out':str(a.out)},indent=2))


if __name__=='__main__':main()
