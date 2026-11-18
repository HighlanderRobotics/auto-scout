"""Rebuild same-event fuel-counter data for the separate-match numerical pilot; no API."""
import argparse,hashlib,json,threading
from pathlib import Path
from types import SimpleNamespace
from hybrid_shooting import refresh_scoreboard
from teacher_labels import atomic_json
from train_detector import heartbeat

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('Choose a new output folder')
 base=json.loads(Path('configs/hybrid_shooting_qm3_dense_v2.json').read_text());pool=json.loads(Path('configs/norcal_identity_pool_v1.json').read_text());a.out.mkdir(parents=True);stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
 try:
  for source in pool['sources']:
   folder=a.out/source['match'].split('_')[-1];folder.mkdir();config=dict(base,video=source['video'],match=source['match'],teams=source['teams'],youtube_url=source['youtube_url'])
   with Path(source['video']).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
   atomic_json(folder/'manifest.json',dict(config=config,source_video_sha256=sha,examples=[{'anchor_s':10},{'anchor_s':138}],purpose='Fuel-counter-only data, not robot/action annotations'))
   refresh_scoreboard(SimpleNamespace(out=folder))
 finally:stop.set()
if __name__=='__main__':main()
