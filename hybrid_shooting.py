"""Prepare/teach/train a human-assisted shooting pilot. Paid teaching requires --execute.

Counts remain proposals. Alliance fuel counters cannot identify an individual shooter.
"""
import argparse,base64,hashlib,json,math,re,subprocess,threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from teacher_labels import atomic_json
from temporal_multicamera import Budget,HAIKU,RESERVES
from train_detector import heartbeat

SYSTEM='''Teach a human-assisted 2026 FRC shooting scout. Observe the CENTERED robot subject in the local crop sequence, using the wide image for context. Detector crops may contain a person or field structure: reject these. Never assume a tracker ID stays on one robot. Identify shooting ONLY from fuel visibly leaving this robot toward a hub. Distinguish feeding (launching elsewhere), intaking, idle and uncertain/occluded. Scoreboard changes are ALLIANCE observations, not proof this robot shot. A stationary pile of yellow balls is not shooting. Read a team only from actual bumper digits; otherwise team="unknown". Count ONLY distinct launches you can follow across frames: sparse frames miss balls and can show the same ball repeatedly. If a defensible count is impossible, visible_launches_min and estimated_attempts_max must be null. A visible minimum is not total attempts or made fuel. Never invent per-robot made counts. State when simultaneous shooters, human-player throws, occlusion, camera changes or sparse sampling prevent attribution. All outputs are reviewable teacher proposals, not truth.'''

SCHEMA={'type':'object','properties':{
'action':{'type':'string','enum':['shooting','feeding','intaking','idle','uncertain','not_robot']},
'confidence':{'type':'number'},'team':{'type':'string'},'bumper_text':{'type':'string'},
'visible_launches_min':{'type':['integer','null']},'estimated_attempts_max':{'type':['integer','null']},
'count_confidence':{'type':'number'},'count_explanation':{'type':'string'},'evidence':{'type':'string'},
'needs_review':{'type':'boolean'}},'required':['action','confidence','team','bumper_text','visible_launches_min','estimated_attempts_max','count_confidence','count_explanation','evidence','needs_review'],'additionalProperties':False}


def parse_counter(reads):
    values=[]
    for r in reads:
        text=re.sub(r'\s+','',r['text']);match=re.fullmatch(r'(\d{1,4})/(240|360|500)',text)
        if match and r['confidence']>=.8:values.append((int(match[1]),int(match[2])))
    return list(values[0]) if len(set(values))==1 else None


def score_increments(rows):
    """Reject gaps/decreases rather than silently smoothing or fabricating fuel."""
    output=[]
    for alliance in ['blue','red']:
        previous=None
        for row in rows:
            value=row.get(alliance)
            if value is None:previous=None;continue
            if previous:
                t,p=previous;delta=value[0]-p[0]
                if delta<0 or delta>40:
                    previous=None;continue
                if row['video_s']-t==1 and value[1]==p[1] and 0<=delta<=40:
                    output.append(dict(start_s=t,end_s=row['video_s'],alliance=alliance,fuel_delta=delta,review_status='ocr_proposal',attributed_team=None))
            previous=(row['video_s'],value)
    return output


def refresh_scoreboard(a):
    manifest=json.loads((a.out/'manifest.json').read_text());config=manifest['config']
    if (a.out/'labels').exists():raise ValueError('Do not replace scoreboard inputs after teaching starts')
    times=[e['anchor_s'] for e in manifest['examples']];begin=max(0,math.floor(min(times)-4));end=math.ceil(max(times)+4)
    folder=a.out/'overlay_expanded';folder.mkdir(exist_ok=True);cap=cv2.VideoCapture(config['video'])
    try:
        for t in range(begin,end+1):
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read()
            if not ok:raise ValueError('Missing overlay source')
            image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            for alliance,box in config['fuel_crops'].items():image.crop(tuple(box)).resize((600,212)).save(folder/f'{t:04d}_{alliance}.jpg',quality=95)
            if (t-begin)%25==0:print(f'Fuel counter extraction {t-begin+1}/{end-begin+1}',flush=True)
    finally:cap.release()
    path=a.out/'overlay_expanded_raw.json'
    with path.open('w') as output:subprocess.run(['swift','read_overlay_crops.swift',str(folder)],stdout=output,check=True)
    raw={r['file']:r['reads'] for r in json.loads(path.read_text())};rows=[dict(video_s=t,**{c:parse_counter(raw[f'{t:04d}_{c}.jpg']) for c in ['blue','red']}) for t in range(begin,end+1)]
    atomic_json(a.out/'scoreboard.json',dict(rows=rows,increments=score_increments(rows),status='Unreviewed dedicated fuel-counter OCR; no total match points'))


def feature_sequence(images):
    masks=[];grays=[];motion=[];components=[]
    for image in images:
        bgr=cv2.cvtColor(np.asarray(image.resize((160,120))),cv2.COLOR_RGB2BGR);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([18,90,100]),np.array([42,255,255]));masks.append(mask>0);grays.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
        n,_,stats,_=cv2.connectedComponentsWithStats(mask);components.append(sum(2<=s[4]<=200 for s in stats[1:]))
    for i in range(1,len(images)):
        changed=cv2.absdiff(grays[i],grays[i-1])>20;yellow=masks[i]|masks[i-1]
        motion.append([float((changed&yellow).mean()),float((changed&yellow)[:60].mean()),float(changed.mean())])
    x=np.asarray(motion);return np.array([*x.mean(axis=0),*x.max(axis=0),np.mean(components),np.std(components),np.mean(masks)],dtype=float)


def prepare(a):
    if a.out.exists():raise ValueError('Choose a new output folder')
    config=json.loads(a.plan.read_text());a.out.mkdir(parents=True);(a.out/'images').mkdir();(a.out/'overlay').mkdir()
    with Path(config['video']).open('rb') as f:video_sha=hashlib.file_digest(f,'sha256').hexdigest()
    cap=cv2.VideoCapture(config['video']);examples=[];overlay=[]
    def frame(t):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read()
        if not ok:raise ValueError('Missing source frame')
        return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    try:
        for t in range(43,143):
            im=frame(t)
            for alliance,box in config['fuel_crops'].items():im.crop(tuple(box)).resize((600,212)).save(a.out/'overlay'/f'{t:04d}_{alliance}.jpg',quality=95)
        for window in config['windows']:
            data=json.loads(Path(window['tracks']).read_text()) if 'tracks' in window else None
            for t in window.get('times',sorted({r['time'] for r in window.get('subjects',[])})):
                middle=next(f for f in data['frames'] if abs(f['video_s']-t)<.001) if data else None
                # High-confidence geometry only; Claude must still reject nonrobots.
                rows=sorted(middle['cameras']['main'],key=lambda r:-r['confidence'])[:3] if middle else [r for r in window['subjects'] if r['time']==t]
                for row in rows:
                    x,y,x2,y2=row['box'];roi=[max(0,int(x)-90),max(220,int(y+220)-config.get('air_context_px',170)),min(1920,int(x2)+90),min(720,int(y2+220)+35)]
                    index=len(examples);local=[];paths=[];count=config.get('sample_count',6);times=[round(float(s),3) for s in np.linspace(t-.5,t+.5,count)]
                    for j,s in enumerate(times):
                        im=frame(s);crop=im.crop(tuple(roi));local.append(crop);path=f'images/e_{index:03d}_{j}.jpg';crop.save(a.out/path,quality=95);paths.append(path)
                    columns=3 if count==6 else 4;sheet=Image.new('RGB',(columns*320,math.ceil(count/columns)*350),'#161b22');draw=ImageDraw.Draw(sheet)
                    for j,im in enumerate(local):
                        tile=ImageOps.contain(im,(320,300));px=j%columns*320;py=j//columns*350;sheet.paste(tile,(px,py+35));draw.text((px+4,py+5),f'{times[j]:.2f}s',fill='white')
                    panel=f'images/e_{index:03d}_sequence.jpg';sheet.save(a.out/panel,quality=95)
                    wide=f'images/e_{index:03d}_context.jpg';context=frame(t).crop(tuple(config['main_crop']));ImageDraw.Draw(context).rectangle(row['box'],outline='lime',width=5);context.resize((1536,400)).save(a.out/wide,quality=95)
                    feature_images=[local[round(i*(count-1)/5)] for i in range(6)]
                    examples.append(dict(id=f'e_{index:03d}',start_s=times[0],end_s=times[-1],anchor_s=t,frame_times=times,box=row['box'],roi=roi,images=paths,sequence=panel,context=wide,features=feature_sequence(feature_images).tolist(),split=window['split']))
                print(f'Prepared shooting examples at {t}s ({len(examples)} total)',flush=True)
    finally:cap.release()
    atomic_json(a.out/'manifest.json',dict(config=config,source_video_sha256=video_sha,examples=examples,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    subprocess.run(['swift','read_overlay_crops.swift',str(a.out/'overlay')],stdout=(a.out/'overlay_raw.json').open('w'),check=True)
    raw={r['file']:r['reads'] for r in json.loads((a.out/'overlay_raw.json').read_text())}
    for t in range(43,143):overlay.append(dict(video_s=t,**{c:parse_counter(raw[f'{t:04d}_{c}.jpg']) for c in ['blue','red']}))
    atomic_json(a.out/'scoreboard.json',dict(rows=overlay,increments=score_increments(overlay),status='Unreviewed OCR; only dedicated fuel counters, never total match score'))
    print('Prepared sequences and local scoreboard OCR; zero paid calls',flush=True)


def teach(a):
    (a.out/'labels').mkdir(exist_ok=True)
    manifest=json.loads((a.out/'manifest.json').read_text())
    scoreboard=json.loads((a.out/'scoreboard.json').read_text())['rows']
    needed={round(e['anchor_s']+offset) for e in manifest['examples'] for offset in [-1,3]}
    if not needed.issubset({r['video_s'] for r in scoreboard}):raise ValueError('Run the scoreboard command to cover every clip before paid teaching')
    root=Path('datasets/temporal_multicamera_v1_run');budget=Budget(root,json.loads((root/'frames.json').read_text())['contract']);stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        for e in manifest['examples']:
            content=[{'type':'text','text':f"CURRENT roster: {json.dumps(manifest['config']['teams'])}. First image: wide context at {e['anchor_s']}s, GREEN BOX identifies the target subject. Second image: chronological {len(e['images'])}-frame sequence spanning {e['start_s']} to {e['end_s']} seconds. Track the GREEN-BOX SUBJECT through this fixed surrounding crop (not necessarily centered vertically; space ABOVE it shows outgoing trajectories). Identity confidence must not reduce action confidence: you can confidently see shooting even with unreadable bumper digits. Return that subject's action. A stationary fuel pile is not shooting. Never infer its action from score alone."}]
            context=[r for r in scoreboard if r['video_s'] in [round(e['anchor_s']-1),round(e['anchor_s']+3)]]
            content.append({'type':'text','text':'Dedicated ALLIANCE fuel counter OCR before/after, unreviewed supporting context only: '+json.dumps(context)+'. A counter increase can come from other robots or human players and can be delayed. It does not establish this subject shot or how many attempts it made. Null means OCR unavailable. Do not copy alliance changes into robot counts.'})
            for name in [e['context'],e['sequence']]:content.append({'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode((a.out/name).read_bytes()).decode()}})
            fingerprint=hashlib.sha256((SYSTEM+json.dumps(SCHEMA,sort_keys=True)+json.dumps(content,sort_keys=True)+HAIKU).encode()).hexdigest();target=a.out/'responses'/f"{e['id']}.json"
            spent=sum(r.get('cost_usd',0) if r['status']=='billed' else r['reserve_usd'] for r in budget.data['attempts'] if r['target'].startswith(str(a.out)+'/'))
            if not target.exists() and spent+RESERVES[HAIKU]>manifest['config']['new_run_cap_usd']:raise RuntimeError('Shooting run cap reached')
            response=budget.request(HAIKU,content,SCHEMA,target,fingerprint,system=SYSTEM)
            if response is None:raise RuntimeError('Shared approved cap reached')
            if response['stop_reason']!='end_turn':raise ValueError('Incomplete teacher response; no retry')
            label=json.loads(response['text'])
            if label['team'] not in manifest['config']['teams'] or label['bumper_text'].strip()!=label['team']:label['team']='unknown'
            for k in ['confidence','count_confidence']:
                if not isinstance(label[k],(int,float)) or not math.isfinite(label[k]) or not 0<=label[k]<=1:raise ValueError('Invalid confidence')
            lo,hi=label['visible_launches_min'],label['estimated_attempts_max']
            if any(v is not None and (type(v)!=int or v<0 or v>100) for v in [lo,hi]) or (lo is not None and hi is not None and hi<lo):raise ValueError('Invalid count proposal')
            atomic_json(a.out/'labels'/f"{e['id']}.json",dict(example_id=e['id'],teacher_model=HAIKU,request_fingerprint=fingerprint,review_status='unreviewed_teacher',**label))
            print(f"Teacher {e['id']}: {label['action']} | shared ${budget.data['spent_usd']:.4f}",flush=True)
        atomic_json(a.out/'billing.json',dict(recorded_run_spend_usd=sum(r.get('cost_usd',0) for r in budget.data['attempts'] if r['target'].startswith(str(a.out)+'/')),shared_spent_usd=budget.data['spent_usd'],shared_reserved_usd=budget.data['reserved_usd']))
    finally:budget.lock.close();stop.set()


def train(a):
    manifest=json.loads((a.out/'manifest.json').read_text());data=[];reviews={}
    if a.reviews:
        saved=json.loads(a.reviews.read_text())
        if saved['source_video_sha256']!=manifest['source_video_sha256']:raise ValueError('Reviews belong to another video')
        reviews={r['id']:r for r in saved['events'] if r.get('reviewed')}
        cap=None
        try:
            for custom in saved.get('new_examples',[]):
                r=reviews.get(custom['id'])
                if not r:continue
                if cap is None:cap=cv2.VideoCapture(manifest['config']['video'])
                if not custom['id'].startswith('custom_') or any(e['id']==custom['id'] for e in manifest['examples']):raise ValueError('Invalid custom example ID')
                x,y,x2,y2=custom['box']
                if not 0<=x<x2<=1920 or not 0<=y<y2<=500 or x2-x<10 or y2-y<10 or not 0<=r['start_s']<r['end_s']<=215:raise ValueError('Invalid custom source crop/window')
                roi=[max(0,int(x)-90),max(220,int(y+220)-350),min(1920,int(x2)+90),min(720,int(y2+220)+35)];images=[]
                for t in np.linspace(r['start_s'],r['end_s'],6):
                    cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,bgr=cap.read()
                    if not ok:raise ValueError('Missing custom training frame')
                    images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(tuple(roi)))
                manifest['examples'].append(dict(custom,start_s=r['start_s'],end_s=r['end_s'],roi=roi,features=feature_sequence(images).tolist(),split='train'))
        finally:
            if cap is not None:cap.release()
    for original in manifest['examples']:
        e=dict(original);path=a.out/'labels_expert'/f"{e['id']}.json"
        if not path.exists():path=a.out/'labels'/f"{e['id']}.json"
        if not path.exists() and e['id'] not in reviews:continue
        label=json.loads(path.read_text()) if path.exists() else {};review=reviews.get(e['id'])
        if review:
            if review['action'] not in SCHEMA['properties']['action']['enum']:raise ValueError('Invalid reviewed action')
            if not 0<=review['start_s']<review['end_s']<=215:raise ValueError('Invalid reviewed times')
            if review.get('attempts') is not None and (type(review['attempts'])!=int or review['attempts']<0):raise ValueError('Invalid reviewed count')
            if review.get('attempts',0) and review['action']!='shooting':raise ValueError('Positive hub-shot count conflicts with a nonshooting label')
            label=dict(label,action=review['action'],confidence=1.,review_status='human_reviewed',human_attempts=review.get('attempts'))
            if review['start_s']!=e['start_s'] or review['end_s']!=e['end_s']:
                cap=cv2.VideoCapture(manifest['config']['video']);images=[]
                try:
                    for t in np.linspace(review['start_s'],review['end_s'],6):
                        cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,bgr=cap.read()
                        if not ok:raise ValueError('Missing reviewed source frame')
                        images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(tuple(e['roi'])))
                finally:cap.release()
                e['features']=feature_sequence(images).tolist();e.update(start_s=review['start_s'],end_s=review['end_s'])
        if label['confidence']>0 and label['action']!='uncertain':data.append((e,label))
    training=[(e,l) for e,l in data if e['split']=='train'];validation=[(e,l) for e,l in data if e['split']=='validation']
    custom_windows=[(r['start_s'],r['end_s']) for r in reviews.values() if r['id'].startswith('custom_')]
    validation=[(e,l) for e,l in validation if not any(lo<=e['end_s'] and e['start_s']<=hi for lo,hi in custom_windows)]
    output=a.model_out or a.out/'model_v1'
    if output.exists():raise ValueError('Choose a new model output folder')
    output.mkdir(parents=True)
    if len(training)<6 or len({l['action']=='shooting' for _,l in training})<2:
        atomic_json(output/'training_result.json',dict(status='awaiting_more_reviewed_labels',training_examples=len(training),scope='No shooting model trained: need six labeled examples including shooting and nonshooting. Counts remain review proposals.'))
        print('Insufficient strong labels: review desk ready; no model weights fabricated',flush=True);return
    x=np.array([e['features'] for e,_ in training]);mean=x.mean(0);scale=np.maximum(x.std(0),1e-5);z=np.column_stack([(x-mean)/scale,np.ones(len(x))])
    # Distill uncertainty rather than turning tentative teacher guesses into hard truth.
    y=np.array([float(l['action']=='shooting') if l['review_status']=='human_reviewed' else .5+(1 if l['action']=='shooting' else -1)*l['confidence']/2 for _,l in training])
    weights=np.array([1. if l['review_status']=='human_reviewed' else l['confidence'] for _,l in training]);w=np.linalg.solve(z.T@(z*weights[:,None])+np.eye(z.shape[1]),z.T@(y*weights))
    np.savez_compressed(output/'shooting_model.npz',weights=w,mean=mean,scale=scale)
    atomic_json(output/'training_manifest.json',[dict(example=e,label=l) for e,l in training])
    predictions=[]
    for e in manifest['examples']:
        score=float(np.append((np.array(e['features'])-mean)/scale,1)@w);predictions.append(dict(example_id=e['id'],shooting_support=score,proposal='shooting' if score>=.65 else 'not_shooting' if score<=.35 else 'review',split=e['split']))
    byid={r['example_id']:r for r in predictions};agreement=[(byid[e['id']]['shooting_support']>=.5)==(l['action']=='shooting') for e,l in validation]
    count_rows=[(e,l) for e,l in training if l.get('review_status')=='human_reviewed' and l.get('human_attempts') is not None]
    count_status='awaiting_six_human_reviewed_counts_with_variation'
    if len(count_rows)>=6 and len({l['human_attempts'] for _,l in count_rows})>1:
        cx=np.array([e['features'] for e,_ in count_rows]);cz=np.column_stack([(cx-mean)/scale,np.ones(len(cx))]);cy=np.array([l['human_attempts']/(e['end_s']-e['start_s']) for e,l in count_rows]);cw=np.linalg.solve(cz.T@cz+np.eye(cz.shape[1]),cz.T@cy)
        np.savez_compressed(output/'attempt_rate_model.npz',weights=cw,mean=mean,scale=scale);count_status='experimental_human_count_rate_model_trained_needs_validation'
        for prediction,e in zip(predictions,manifest['examples']):
            rate=max(0,float(np.append((np.array(e['features'])-mean)/scale,1)@cw));prediction['attempt_rate_proposal']=rate;prediction['attempt_count_proposal']=rate*(e['end_s']-e['start_s'])
    atomic_json(output/'predictions.json',predictions);atomic_json(output/'training_result.json',dict(status='trained_experimental',source_manifest_sha256=hashlib.sha256((a.out/'manifest.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reviews_sha256=hashlib.sha256(a.reviews.read_bytes()).hexdigest() if a.reviews else None,training_examples=len(training),training_shooting_positive_examples=sum(l['action']=='shooting' for _,l in training),validation_shooting_positive_examples=sum(l['action']=='shooting' for _,l in validation),human_training_examples=sum(l['review_status']=='human_reviewed' for _,l in training),validation_examples=len(validation),validation_teacher_agreement=sum(agreement)/len(agreement) if agreement else None,count_model_status=count_status,scope='Weak-label shooting support; teacher agreement is not human accuracy. Human review required. Per-robot made counts never trained from alliance totals.'))
    print('Trained local shooting-support pilot; '+count_status,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','scoreboard','teach','train']);p.add_argument('--plan',type=Path,default=Path('configs/hybrid_shooting_qm3_v1.json'));p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--reviews',type=Path);p.add_argument('--model-out',type=Path);a=p.parse_args()
    if a.command=='teach' and not a.execute:p.error('Paid teaching requires --execute after authorization')
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:{'prepare':prepare,'scoreboard':refresh_scoreboard,'teach':teach,'train':train}[a.command](a)
    finally:stop.set()

if __name__=='__main__':main()
