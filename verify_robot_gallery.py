"""Blindly check proposed reference-card identities using the shared API budget."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageOps

from teacher_labels import atomic_json
from temporal_multicamera import Budget,OPUS,HAIKU


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gallery',type=Path,default=Path('datasets/focus_MQZ4Yrw86uQ/gallery_v1'))
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    manifest=json.loads((args.gallery/'manifest.json').read_text());teams=list(manifest['teams'])
    sheet=Image.new('RGB',(960,550),'#222')
    for i,team in enumerate(teams):
        image=Image.open(args.gallery/f'team_{team}.jpg');image=ImageOps.contain(image,(320,240));x=i%3*320;y=i//3*275
        sheet.paste(image,(x,y+30));ImageDraw.Draw(sheet).text((x+5,y+5),f'Card {i}',fill='white')
    image=args.gallery/'blind_cards.jpg';sheet.save(image,quality=95)
    if not args.execute:
        print('Blind card image prepared. No API calls.');return
    root=Path('datasets/temporal_multicamera_v1_run');contract=json.loads((root/'frames.json').read_text())['contract'];budget=Budget(root,contract)
    system='Identify physical FRC robot team numbers from the visible bumpers on these numbered cards. Ignore any field monitors, people and robots behind the centered subject. Read actual digits; when blurred say unknown. Do not force a one-to-one assignment or infer a number merely because it is left over. These are six independent cards, not a tracking sequence.'
    schema={'type':'object','properties':{'cards':{'type':'array','items':{'type':'object','properties':{'card':{'type':'integer'},'team':{'type':'string','enum':teams+['unknown']},'confidence':{'type':'number'},'bumper_text':{'type':'string'},'notes':{'type':'string'}},'required':['card','team','confidence','bumper_text','notes'],'additionalProperties':False}}},'required':['cards'],'additionalProperties':False}
    content=[{'type':'text','text':'Candidate roster, unordered: '+', '.join(sorted(teams,key=int))},
             {'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':base64.b64encode(image.read_bytes()).decode()}}]
    try:
        response=None
        for model in [OPUS,HAIKU]:
            fingerprint=hashlib.sha256((system+json.dumps(content,sort_keys=True)+json.dumps(schema,sort_keys=True)+model).encode()).hexdigest()
            response=budget.request(model,content,schema,args.gallery/f'blind_check_{model}.json',fingerprint,system=system)
            if response is not None:break
        if response is None:raise RuntimeError('Existing shared budget exhausted')
        if response['stop_reason']!='end_turn':raise ValueError('Incomplete card check; no paid retry')
        result=json.loads(response['text']);assert sorted(row['card'] for row in result['cards'])==list(range(len(teams)))
        result['expected_by_card']=teams;result['scope']='Independent-model check of visible reference crops, still not independent human ground truth'
        atomic_json(args.gallery/'blind_check.json',result)
        print(json.dumps(result,indent=2),flush=True)
    finally:budget.lock.close()


if __name__=='__main__':main()
