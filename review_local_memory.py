"""One independent Haiku temporal review, charged to the existing shared ledger."""
import argparse,base64,fcntl,hashlib,json
from pathlib import Path
import cv2
import anthropic
from teacher_labels import MODEL,INPUT_RATE,OUTPUT_RATE,atomic_json,load_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',type=Path);p.add_argument('--video',type=Path,required=True)
    p.add_argument('--round',type=int,choices=[1,2],required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:p.error('Paid review requires the approved --execute flag')
    root=Path('datasets/identity_coach_v1');lock=(root/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=json.loads((root/'ledger.json').read_text())
    if ledger['model']!=MODEL or not 0<ledger['cap_usd']<=4:raise ValueError('Ledger model/cap mismatch')
    data=json.loads(a.paths.read_text());digest=hashlib.sha256(a.paths.read_bytes()).hexdigest()[:12]
    out=root/'temporal_reviews'/digest/f'round{a.round}.json';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():print(out.read_text());return
    names={identifier:str(tr['team']) for tr in data['tracks'] for identifier in tr['source_ids']}
    prompt='Review the LOCAL tracker using three chronological pairs of video crops, each pair 0.2 seconds apart. Review 1 and 2 are separate independent requests with different moments. Judge visible robot identity continuity, apparent identity swaps, unnecessary path resets, and false/missing robot observations. Numeric IDs are persistent local states, not bumper numbers. Unknown identities are allowed. Hidden field-position estimates are predictions, not observations. Do not demand a visible number every frame. Report uncertainty, do not invent mistakes or assume all six robots are visible. Do not infer a swap merely from motion between distant pairs. Return compact structured review and concrete recommendations.'
    prompt += ' The team value is the displayed canonical path identity after backward joining. Internal state IDs may change while the same team path remains continuous; that alone is not a displayed path reset. Unknown rows belong to the anonymous debug view. Assess their detector errors separately from errors in the named team paths. Flag any actual incorrect team assignments, and distinguish uncertain evidence from confirmed errors.'
    content=[{'type':'text','text':prompt}];cap=cv2.VideoCapture(str(a.video));sampled=[]
    starts=[.5,8,17] if a.round==1 else [2.5,10,18]
    try:
        for relative in [value+delta for value in starts for delta in [0,.2]]:
            frame=min(data['frames'],key=lambda f:abs(f['t']-relative))
            cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000);ok,image=cap.read()
            if not ok:raise ValueError('Cannot read review frame')
            x1,y1,x2,y2=data['calibration']['crop'];image=image[y1:y2,x1:x2]
            scale=960/image.shape[1];image=cv2.resize(image,(960,round(image.shape[0]*scale)))
            _,encoded=cv2.imencode('.jpg',image)
            rows=[dict(id=r['id'],team=names.get(r['id'],'unknown'),box=[round((r['box'][0]-x1)*scale),round((r['box'][1]-y1)*scale),round((r['box'][2]-x1)*scale),round((r['box'][3]-y1)*scale)],confidence=r['confidence']) for r in frame['robots']]
            sampled.append(frame['t']);content.extend([{'type':'text','text':f"t={frame['t']}s, crop coordinates: "+json.dumps(rows)},
                {'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode(encoded).decode()}}])
    finally:cap.release()
    reserve=.05
    if ledger['spent_usd']+ledger['reserved_usd']+reserve>ledger['cap_usd']:raise RuntimeError('Shared cap reached')
    attempt={'temporal_review':a.round,'paths_sha256':digest,'status':'pending'};ledger['attempts'].append(attempt);ledger['reserved_usd']+=reserve
    atomic_json(root/'ledger.json',ledger)
    schema={'type':'object','properties':{'quality':{'type':'string','enum':['poor','mixed','good']},
        'suspected_identity_swaps':{'type':'array','items':{'type':'string'}},
        'misses_or_false_detections':{'type':'array','items':{'type':'string'}},
        'continuity_notes':{'type':'string'},'recommendations':{'type':'array','items':{'type':'string'}}},
        'required':['quality','suspected_identity_swaps','misses_or_false_detections','continuity_notes','recommendations'],'additionalProperties':False}
    client=anthropic.Anthropic(api_key=load_key(),max_retries=0,timeout=60)
    try:
        response=client.messages.create(model=MODEL,max_tokens=3500,output_config={'format':{'type':'json_schema','schema':schema}},messages=[{'role':'user','content':content}])
    except Exception as error:
        attempt.update(status='failed_or_uncertain',error_type=type(error).__name__);atomic_json(root/'ledger.json',ledger)
        raise RuntimeError('Review failed; reserve retained, no automatic retry') from None
    usage=response.usage.model_dump();cost=(usage['input_tokens']*INPUT_RATE+usage['output_tokens']*OUTPUT_RATE+(usage.get('cache_creation_input_tokens') or 0)*INPUT_RATE*2+(usage.get('cache_read_input_tokens') or 0)*INPUT_RATE)/1e6
    ledger['spent_usd']+=cost;ledger['reserved_usd']-=reserve;attempt.update(status='billed',usage=usage,cost_usd=cost);atomic_json(root/'ledger.json',ledger)
    text=''.join(block.text for block in response.content if block.type=='text')
    atomic_json(out,{'round':a.round,'sampled_times':sampled,'review':text,'cost_usd':cost,'scope':'Supplemental fallible teacher opinion, not ground truth'})
    if cost>reserve or not response.model.startswith(MODEL):raise RuntimeError('Unexpected model/cost')
    print(out.read_text());print(f"Total spend ${ledger['spent_usd']:.6f} plus reserve ${ledger['reserved_usd']:.2f} / ${ledger['cap_usd']:.2f}")


if __name__=='__main__':main()
