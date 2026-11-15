"""Local visual forward memory; preserve identity through short detection losses.

Consumes the detector's observations, not an answer key or Claude predictions.
Estimated positions are kept out of frames[].robots so evaluators count only real
detections. Later team_paths OCR voting supplies the backward identity correction.
"""
import argparse,copy,json,math,shutil,threading,hashlib
from pathlib import Path
import cv2
import lap
import numpy as np
from auto_paths import color,field_position,load_calibration,render_animation,heartbeat


def appearance(image,weights=None):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);rows=[]
    for region in (hsv,hsv[:hsv.shape[0]//2],hsv[hsv.shape[0]//2:]):
        hist=cv2.calcHist([region],[0,1],None,[18,8],[0,180,0,256]).reshape(-1)
        rows.extend(np.sqrt(hist/max(float(hist.sum()),1)))
    vector=np.asarray(rows,dtype=np.float32)
    if weights is not None:vector*=weights
    vector/=max(float(np.linalg.norm(vector)),1e-8)
    lower=hsv[hsv.shape[0]//2:];saturated=(lower[:,:,1]>=80)&(lower[:,:,2]>=60)
    red=int((((lower[:,:,0]<=12)|(lower[:,:,0]>=170))&saturated).sum())
    blue=int(((lower[:,:,0]>=90)&(lower[:,:,0]<=130)&saturated).sum())
    alliance=None
    if max(red,blue)>lower.shape[0]*lower.shape[1]*.03 and max(red,blue)/max(red+blue,1)>=.8:
        alliance='red' if red>blue else 'blue'
    return vector,alliance


class VisualMemory:
    def __init__(self,max_gap=2,known_raw_ids=()):
        self.max_gap=max_gap;self.states={};self.next_id=1;self.known_raw_ids=set(known_raw_ids)

    def update(self,observations,t):
        active=[s for s in self.states.values() if 0<t-s['t']<=self.max_gap]
        costs=np.full((len(active),len(observations)),1e3)
        for i,state in enumerate(active):
            dt=t-state['t'];box=state['box'];predicted=box+np.tile(state['velocity']*min(dt,1),2)
            center=(predicted[:2]+predicted[2:])/2;size=np.maximum(box[2:]-box[:2],1)
            for j,row in enumerate(observations):
                if state['alliance'] and row['alliance'] and state['alliance']!=row['alliance']:continue
                candidate=np.asarray(row['box']);other_size=np.maximum(candidate[2:]-candidate[:2],1)
                delta=float(np.linalg.norm((candidate[:2]+candidate[2:])/2-center))
                gate=35+.7*float(size.max())+100*dt
                if delta>gate or np.max(np.maximum(size/other_size,other_size/size))>2.5:continue
                similarity=float(np.dot(state['appearance'],row['appearance']))
                # Bumper color alone cannot identify a teammate after an occlusion.
                # Longer losses need much stronger appearance agreement.
                if similarity<(.9 if dt>=.5 else .65):continue
                cost=.55*delta/gate+.35*(1-similarity)+.1*float(np.abs(np.log(size/other_size)).mean())
                if row['raw_id']==state['raw_id']:cost-=.08
                costs[i,j]=cost
        matches={}
        if costs.size:
            _,assignment,_=lap.lapjv(costs,extend_cost=True,cost_limit=.55)
            for i,j in enumerate(assignment):
                if j<0:continue
                # A small raw-ID continuity bonus is not enough to force a close crossing.
                alternatives=np.delete(costs[:,j],i)
                if len(alternatives) and float(alternatives.min())-costs[i,j]<.04:continue
                row_options=np.delete(costs[i],j)
                if len(row_options) and float(row_options.min())-costs[i,j]<.04:continue
                matches[j]=active[i]
        output=[]
        for j,row in enumerate(observations):
            box=np.asarray(row['box'],dtype=float);state=matches.get(j)
            if state is None:
                trusted=row['raw_id'] in self.known_raw_ids
                if not trusted and (row['confidence']<.3 or ('field' in row and row['field'] is None)):continue
                identifier=self.next_id;self.next_id+=1
                state={'id':identifier,'velocity':np.zeros(2),'appearance':row['appearance'].copy(),'alliance':None,'color_votes':{},'hits':0}
                self.states[identifier]=state
            else:
                elapsed=t-state['t'];velocity=((box[:2]+box[2:])-(state['box'][:2]+state['box'][2:]))/(2*elapsed)
                state['velocity']=.5*state['velocity']+.5*velocity
                vector=.9*state['appearance']+.1*row['appearance'];state['appearance']=vector/max(float(np.linalg.norm(vector)),1e-8)
            state.update(box=box,t=t,raw_id=row['raw_id'],confidence=row['confidence'],hits=state['hits']+1)
            if row['alliance']:
                votes=state['color_votes'];votes[row['alliance']]=votes.get(row['alliance'],0)+1
                best=max(votes,key=votes.get)
                if votes[best]>=3 and votes[best]/sum(votes.values())>=.8:state['alliance']=best
            output.append((state,row))
        return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',type=Path);p.add_argument('--video',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--appearance-model',type=Path);p.add_argument('--teams',help='Roster for backward OCR anchor validation');a=p.parse_args()
    if a.out.exists():p.error('Choose a new output folder')
    weights=np.load(a.appearance_model,allow_pickle=False)['weights'] if a.appearance_model else None
    data=json.loads(a.paths.read_text());matrix_path=a.paths.parent/'calibration.json'
    _,matrix=load_calibration(matrix_path)
    source_evidence=json.loads(a.evidence.read_text())
    roster=set(a.teams.split(',')) if a.teams else {r['team'] for r in source_evidence if r.get('method')=='haiku_appearance' and r.get('team')}
    votes={}
    for row in source_evidence:
        if row.get('team') in roster and row.get('confidence',0)>=.8:
            votes.setdefault(row['id'],{}).setdefault(row['team'],set()).add(row['t'])
    known={identifier for identifier,names in votes.items() if sum(len(times)>=2 for times in names.values())==1}
    cap=cv2.VideoCapture(str(a.video));memory=VisualMemory(known_raw_ids=known);tracks={};mapping={};frames=[]
    a.out.mkdir(parents=True);stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        for index,frame in enumerate(data['frames']):
            cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000);ok,image=cap.read()
            if not ok:raise ValueError('Unreadable video frame')
            observations=[]
            for row in frame['robots']:
                x1,y1,x2,y2=map(round,row['box']);crop=image[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
                if crop.size==0 or min(crop.shape[:2])<2:continue
                vector,alliance=appearance(crop,weights)
                observations.append(dict(row,raw_id=row['id'],appearance=vector,alliance=alliance))
            robots=[];seen=set()
            for state,row in memory.update(observations,frame['t']):
                identifier=state['id'];seen.add(identifier)
                mapping[(round(frame['t'],4),row['raw_id'])]=identifier
                robot={key:row[key] for key in ('box','confidence','field')};robot.update(id=identifier,raw_id=row['raw_id']);robots.append(robot)
                if row['field'] is not None:
                    track=tracks.setdefault(identifier,dict(id=identifier,color=list(color(identifier)),points=[]))
                    track['points'].append(dict(t=frame['t'],x=row['field'][0],y=row['field'][1],confidence=row['confidence'],raw_id=row['raw_id']))
            for identifier,state in memory.states.items():
                dt=frame['t']-state['t']
                if identifier in seen or state['hits']<3 or not 0<dt<=memory.max_gap or identifier not in tracks:continue
                predicted=state['box']+np.tile(state['velocity']*min(dt,1),2)
                position=field_position(predicted,matrix)
                if position is not None:
                    tracks[identifier]['points'].append(dict(t=frame['t'],x=position[0],y=position[1],estimated=True,visible=False))
            frames.append(dict(frame,robots=robots))
            if index%50==0:print(f"Memory {index+1}/{len(data['frames'])}: {len(tracks)} persistent IDs",flush=True)
    finally:
        cap.release();stop.set()
    evidence=[]
    for row in source_evidence:
        identifier=mapping.get((round(row['t'],4),row['id']))
        if identifier is not None:evidence.append(dict(row,id=identifier,raw_id=row['id']))
    (a.out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    shown=[tr for tr in tracks.values() if sum(not p.get('estimated',False) for p in tr['points'])>=3]
    confirmed={state['id'] for state in memory.states.values() if state['hits']>=3}
    for frame in frames:
        frame['robots']=[row for row in frame['robots'] if row['id'] in confirmed]
    result=copy.deepcopy(data);result.update(frames=frames,tracks=shown,memory_mode='local visual forward state plus later identity voting',video_id_space='raw_id')
    result['summary'].update(shown_track_ids=len(shown),memory_state_ids=len(memory.states),projected_track_ids=len(tracks),short_tracks_omitted=len(tracks)-len(shown),estimated_path_points=sum(bool(p.get('estimated')) for tr in shown for p in tr['points']))
    result['memory_parameters']={'max_hidden_seconds':memory.max_gap,'new_state_confidence':.3,'short_gap_similarity':.65,'long_gap_similarity':.9,'backward_anchor_raw_ids':sorted(known)}
    result['memory_input_sha256']=hashlib.sha256(a.paths.read_bytes()).hexdigest()
    if a.appearance_model:result['memory_appearance_sha256']=hashlib.sha256(a.appearance_model.read_bytes()).hexdigest()
    # The old video shows raw IDs; retain it as explicit evidence, not a relabeled video.
    for name in ['tracking.mp4','calibration.json','calibration_preview.jpg','bytetrack.yaml']:
        source=a.paths.parent/name
        if source.exists():shutil.copyfile(source,a.out/name)
    render_animation(result,a.out)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
