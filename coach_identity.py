"""Budgeted Haiku appearance coaching on local track crops; caches every response.

Produces evidence consumable by team_paths.py. The checked answer key is never
included in requests. This labels identity examples; it does not retrain YOLO.
"""
import argparse
import base64
import json
import hashlib
from pathlib import Path
import threading

import anthropic
import cv2

from auto_paths import heartbeat
from teacher_labels import atomic_json, load_key, MODEL, INPUT_RATE, OUTPUT_RATE
from track_llm import TEAMS


def grade_detector(args, ledger, client):
    """Supplemental teacher review of four evenly spaced local detector outputs."""
    if client is None:
        raise ValueError('--grade requires --execute (paid within shared cap)')
    data = json.loads(args.paths.read_text())
    cap = cv2.VideoCapture(str(args.video))
    try:
        for index in [0,len(data['frames'])//3,2*len(data['frames'])//3,len(data['frames'])-1]:
            fingerprint=hashlib.sha256(args.paths.read_bytes()+str(args.video.resolve()).encode()).hexdigest()[:12]
            output = args.out/f'grade_v2_{fingerprint}_frame_{index}.json'
            if output.exists():
                continue
            frame = data['frames'][index]
            cap.set(cv2.CAP_PROP_POS_MSEC,frame['video_s']*1000)
            ok,image = cap.read()
            if not ok:
                raise ValueError('Cannot read grading frame')
            # Supply image and original predictions as separate evidence, without
            # hiding anything underneath painted boxes or exposing the answer key.
            image = cv2.resize(image,(960,540))
            ok,encoded = cv2.imencode('.jpg',image)
            if not ok:
                raise ValueError('Cannot encode grading frame')
            predictions=[dict(row,box=[round(value/2,2) for value in row['box']]) for row in frame['robots']]
            prompt = 'Grade these LOCAL detector predictions against the attached frame. Image and prediction coordinates are BOTH 960x540. Main field view only, y=55..360; ignore inset cameras. Identify false detections, missed visible robots, and poorly placed boxes. Do not assume six robots visible. Return JSON {"false_positive_ids": [integer], "missed_visible_robots": integer, "poor_box_ids": [integer], "notes": string}. This is a subjective teacher review, not ground truth. Predictions: '+json.dumps(predictions)
            if ledger['spent_usd']+ledger['reserved_usd']+.05 > 2:
                raise RuntimeError('Shared $2 cap reached')
            attempt = {'grade_frame':index,'status':'pending'}
            ledger['attempts'].append(attempt);ledger['reserved_usd']+=.05
            atomic_json(args.out/'ledger.json',ledger)
            try:
                response = client.messages.create(model=MODEL,max_tokens=2400,
                    output_config={'format':{'type':'json_schema','schema':{'type':'object','properties':{
                        'false_positive_ids':{'type':'array','items':{'type':'integer'}},
                        'missed_visible_robots':{'type':'integer'},
                        'poor_box_ids':{'type':'array','items':{'type':'integer'}},
                        'notes':{'type':'string'}},'required':['false_positive_ids','missed_visible_robots','poor_box_ids','notes'],'additionalProperties':False}}},
                    messages=[{'role':'user','content':[
                    {'type':'text','text':prompt}, {'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode(encoded).decode()}}]}])
            except Exception as error:
                attempt.update(status='failed_or_uncertain',error_type=type(error).__name__)
                atomic_json(args.out/'ledger.json',ledger)
                raise RuntimeError('Grader failed; reservation retained; no automatic retry') from None
            usage = response.usage.model_dump()
            cost=(usage['input_tokens']*INPUT_RATE+usage['output_tokens']*OUTPUT_RATE+(usage.get('cache_creation_input_tokens') or 0)*INPUT_RATE*2+(usage.get('cache_read_input_tokens') or 0)*INPUT_RATE)/1e6
            ledger['spent_usd']+=cost;ledger['reserved_usd']-=.05
            attempt.update(status='billed',usage=usage,cost_usd=cost)
            atomic_json(args.out/'ledger.json',ledger)
            raw=''.join(block.text for block in response.content if block.type=='text')
            atomic_json(output,{'frame_index':index,'teacher_review':raw,'cost_usd':cost,'scope':'Supplemental teacher opinion, not independent ground truth'})
            print(f'Graded frame {index}; total ${ledger["spent_usd"]:.4f}/$2',flush=True)
            if cost>.05 or not response.model.startswith(MODEL):
                raise RuntimeError('Unexpected grading cost/model')
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, default=Path('runs/auto_paths_youtube_demo/paths.json'))
    parser.add_argument('--crops', type=Path, default=Path('runs/team_paths_free_evidence_pixels_v1/ocr_crops'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Send approved paid requests; otherwise only prepare crops')
    parser.add_argument('--round', type=int, default=1, choices=[1,2], help='Round 2 selects sharper crops from temporal bins, sharing the same ledger')
    parser.add_argument('--base-evidence', type=Path, help='Existing local OCR/color evidence to retain')
    parser.add_argument('--grade', action='store_true', help='Ask Haiku to grade four local detector frames using the shared ledger')
    parser.add_argument('--video', type=Path, default=Path('einstein_final_1.mp4'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out/'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'cap_usd':2.0,'spent_usd':0.0,'reserved_usd':0.0,'attempts':[],'model':MODEL}
    if ledger['cap_usd'] != 2.0 or ledger['model'] != MODEL:
        raise ValueError('Ledger cap/model mismatch')
    # Serialize runs so two processes cannot spend the same remaining budget.
    import fcntl
    lock = (args.out/'job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    descriptions = json.loads(Path('start/robot_looks.json').read_text())
    data = json.loads(args.paths.read_text())
    evidence = json.loads(args.base_evidence.read_text()) if args.base_evidence else []
    response_dir = args.out if args.round == 1 else args.out/'round2'
    response_dir.mkdir(exist_ok=True)
    client = anthropic.Anthropic(api_key=load_key(), max_retries=0, timeout=45) if args.execute else None
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    try:
        if args.grade:
            grade_detector(args,ledger,client)
            return
        for track in data['tracks']:
            files = sorted(args.crops.glob(f"id{track['id']}_t*.jpg"))
            # Alternate temporal samples into two separate views, requiring agreement.
            for panel in range(2):
                pool = files[panel::2]
                if not pool:
                    continue
                chosen = [pool[i] for i in sorted(set([0,len(pool)//2,len(pool)-1]))]
                if args.round == 2:
                    def sharpness(path):
                        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                        return cv2.Laplacian(gray, cv2.CV_64F).var()
                    bins = [pool[len(pool)*i//3:len(pool)*(i+1)//3] for i in range(3)]
                    chosen = [max(group,key=sharpness) for group in bins if group]
                content = [{'type':'text','text': 'Identify the SAME tracked robot across these chronological crops. Candidate profiles: '+json.dumps(descriptions)+'. Use bumper color, shape, materials and mechanisms; digits may be invisible. If crops contain different robots or evidence is weak, return team=null. Return JSON only: {"team": string or null, "confidence": number 0..1, "reason": short string}. Do not force a choice.'}]
                for path in chosen:
                    content.append({'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode(path.read_bytes()).decode()}})
                response_path = response_dir/f"id{track['id']}_panel{panel}.json"
                if response_path.exists():
                    result = json.loads(response_path.read_text())
                elif client:
                    reserve = .05
                    if ledger['spent_usd'] + ledger['reserved_usd'] + reserve > 2:
                        raise RuntimeError('Approved $2 cap reached')
                    attempt = {'id':track['id'],'panel':panel,'status':'pending','crops':[p.name for p in chosen]}
                    ledger['attempts'].append(attempt)
                    ledger['reserved_usd'] += reserve
                    atomic_json(ledger_path,ledger)
                    try:
                        response = client.messages.create(model=MODEL,max_tokens=1200,
                            output_config={'format':{'type':'json_schema','schema':{
                                'type':'object','properties':{'team':{'type':['string','null']},
                                'confidence':{'type':'number'},'reason':{'type':'string'}},
                                'required':['team','confidence','reason'],'additionalProperties':False}}},
                            messages=[{'role':'user','content':content}])
                    except Exception as error:
                        attempt.update(status='failed_or_uncertain',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
                        atomic_json(ledger_path,ledger)
                        raise RuntimeError(f"Haiku stopped: {type(error).__name__}, HTTP {getattr(error,'status_code',None)}. No automatic retry.") from None
                    usage = response.usage.model_dump()
                    cost = (usage['input_tokens']*INPUT_RATE + usage['output_tokens']*OUTPUT_RATE + (usage.get('cache_creation_input_tokens') or 0)*INPUT_RATE*2 + (usage.get('cache_read_input_tokens') or 0)*INPUT_RATE)/1e6
                    ledger['spent_usd'] += cost
                    ledger['reserved_usd'] -= reserve
                    attempt.update(status='billed',usage=usage,cost_usd=cost,response_model=response.model)
                    atomic_json(ledger_path,ledger)
                    raw = ''.join(block.text for block in response.content if block.type=='text')
                    (response_dir/f"id{track['id']}_panel{panel}.txt").write_text(raw)
                    if cost > reserve or not response.model.startswith(MODEL):
                        raise RuntimeError('Unexpected cost/model; stopped')
                    try:
                        result = json.loads(raw.strip().removeprefix('```json').removesuffix('```'))
                    except ValueError:
                        print(f"Track {track['id']} panel {panel}: invalid response excluded",flush=True)
                        continue
                    atomic_json(response_path,result)
                    print(f"Track {track['id']} panel {panel}: {result.get('team')} ; spent ${ledger['spent_usd']:.4f}/$2",flush=True)
                else:
                    continue
                team = result.get('team')
                confidence = result.get('confidence',0)
                if team in TEAMS and isinstance(confidence,(int,float)) and .8 <= confidence <= 1:
                    evidence.append({'id':track['id'],'team':team,'confidence':confidence,'t':float(chosen[0].stem.split('_t')[1])/1000,'method':'haiku_appearance','reason':result.get('reason','')})
                atomic_json(response_dir/'evidence.json',evidence)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
