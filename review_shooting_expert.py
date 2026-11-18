"""One bounded expert batch for shooting examples, using the existing shared ledger."""
import argparse,base64,copy,hashlib,json,threading
from pathlib import Path
from hybrid_shooting import SCHEMA,SYSTEM
from temporal_multicamera import Budget,OPUS
from teacher_labels import atomic_json
from train_detector import heartbeat

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--ids',nargs='+',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:p.error('Paid expert review requires --execute after cost authorization')
 m=json.loads((a.run/'manifest.json').read_text());examples={e['id']:e for e in m['examples']};schema=copy.deepcopy(SCHEMA);schema['properties']['example_id']={'type':'string','enum':a.ids};schema['required'].append('example_id');batch={'type':'object','properties':{'examples':{'type':'array','items':schema}},'required':['examples'],'additionalProperties':False}
 content=[{'type':'text','text':'Review each numbered example INDEPENDENTLY. The green rectangle in its CONTEXT identifies its subject at the anchor time. It need not be in the center of the tall crop: empty space above it contains its outgoing ball trajectory. Action confidence is separate from team-number or count confidence. You may be confident about a shooting stream even if you cannot count all balls or read a bumper. Conversely, do not force shooting because balls are merely in the air. Roster: '+json.dumps(m['config']['teams'])}]
 for eid in a.ids:
  e=examples[eid];content.append({'type':'text','text':f"EXAMPLE {eid}, t={e['anchor_s']}; context followed by {len(e['images'])} chronological crop frames covering {e['start_s']}–{e['end_s']}s."})
  for name in [e['context'],e['sequence']]:content.append({'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode((a.run/name).read_bytes()).decode()}})
 fingerprint=hashlib.sha256((SYSTEM+json.dumps(content,sort_keys=True)+json.dumps(batch,sort_keys=True)+OPUS).encode()).hexdigest();root=Path('datasets/temporal_multicamera_v1_run');b=Budget(root,json.loads((root/'frames.json').read_text())['contract']);stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
 try:
  response=b.request(OPUS,content,batch,a.run/'expert_response.json',fingerprint,system=SYSTEM)
  if response is None:raise RuntimeError('Shared Opus sublimit reached; no call made')
  if response['stop_reason']!='end_turn':raise ValueError('Incomplete expert response; do not retry')
  rows=json.loads(response['text'])['examples']
  if sorted(r['example_id'] for r in rows)!=sorted(a.ids):raise ValueError('Expert example mismatch')
  (a.run/'labels_expert').mkdir(exist_ok=True)
  for r in rows:
   for key in ['confidence','count_confidence']:
    if type(r[key]) not in (int,float) or not 0<=r[key]<=1:raise ValueError('Invalid confidence')
   if r['team'] not in m['config']['teams'] or r['bumper_text'].strip()!=r['team']:r['team']='unknown'
   lo,hi=r['visible_launches_min'],r['estimated_attempts_max']
   if any(v is not None and (type(v)!=int or not 0<=v<=100) for v in [lo,hi]) or lo is not None and hi is not None and lo>hi:raise ValueError('Invalid expert count proposal')
   r.update(teacher_model=OPUS,review_status='unreviewed_expert_proposal',request_fingerprint=fingerprint);atomic_json(a.run/'labels_expert'/f"{r['example_id']}.json",r);print(r['example_id'],r['action'],r['confidence'],flush=True)
  atomic_json(a.run/'expert_billing.json',dict(cost_usd=response['cost_usd'],shared_spent_usd=b.data['spent_usd'],shared_reserved_usd=b.data['reserved_usd']))
 finally:b.lock.close();stop.set()
if __name__=='__main__':main()
