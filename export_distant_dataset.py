"""Combine the checked baseline with reviewed full-frame SF examples, retaining background."""
import json,shutil
from pathlib import Path


def main():
    out=Path('datasets/distant_sf_training_v1')
    out.mkdir(exist_ok=False)
    old=Path('datasets/checked_clip_v1')
    for directory in ('images','labels'):
        for split in ('train','val','test'):
            shutil.copytree(old/directory/split,out/directory/split)
    review=json.loads(Path('configs/sf_distant_reviewed_boxes.json').read_text())
    for time,boxes in review['frames'].items():
        split='val' if time=='22' else 'test' if time=='24' else 'train'
        for repeat in range(6 if split=='train' else 1):
            name=f'sf_{time}_{repeat}'
            shutil.copyfile(f'datasets/distant_sf_native_labels_v1/sf_{time}.jpg',out/'images'/split/(name+'.jpg'))
            rows=[]
            for x1,y1,x2,y2 in boxes:
                rows.append(f'0 {(x1+x2)/3840} {(y1+y2)/840} {(x2-x1)/1920} {(y2-y1)/420}')
            (out/'labels'/split/(name+'.txt')).write_text('\n'.join(rows)+'\n')
    (out/'data.yaml').write_text(f'path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames: [robot]\n')
    (out/'manifest.json').write_text(json.dumps({'classes':['robot'],'scope':review['scope'],'original_dataset':str(old),'sf_train_times':[10,12,14,16,18],'sf_val_times':[22],'sf_test_times':[24],'sf_sampling_repeats':6},indent=2)+'\n')
    print('Exported original dataset plus 5 reviewed SF training frames, one SF validation frame, one SF test frame. Repeats are weighting, not independent examples.')


if __name__=='__main__':main()
