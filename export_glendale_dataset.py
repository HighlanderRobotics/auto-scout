"""Export additional reviewed Glendale full-field examples; preserve the baseline test."""
import json,shutil
from pathlib import Path


def main():
    out=Path('datasets/glendale_detector_v1');out.mkdir(exist_ok=False)
    old=Path('datasets/checked_clip_v1')
    for folder in ['images','labels']:
        for split in ['train','val','test']:shutil.copytree(old/folder/split,out/folder/split)
    review=json.loads(Path('configs/glendale_teacher_review.json').read_text());all_boxes={}
    for split,key in [('train','train_times'),('val','validation_times'),('test','test_times')]:
        for t in review[key]:
            proposals=json.loads(Path(f'datasets/glendale_teacher_v1/g_{t}.json').read_text())['robots']
            replacements=review['replace_boxes'].get(str(t),{});removed=review['remove_boxes'].get(str(t),[])
            boxes=[replacements.get(str(i),row['box']) for i,row in enumerate(proposals) if i not in removed]
            boxes.append(review['common_visible_robot_box']);boxes.extend(review['additional_boxes'].get(str(t),[]))
            if len(boxes)!=6:raise ValueError(f'Unexpected reviewed count at {t}: {len(boxes)}')
            all_boxes[str(t)]=boxes
            for repeat in range(4 if split=='train' else 1):
                name=f'g_{t}_{repeat}';shutil.copyfile(f'datasets/glendale_teacher_v1/g_{t}.jpg',out/'images'/split/(name+'.jpg'))
                rows=[f'0 {(x1+x2)/3840} {(y1+y2)/640} {(x2-x1)/1920} {(y2-y1)/320}' for x1,y1,x2,y2 in boxes]
                (out/'labels'/split/(name+'.txt')).write_text('\n'.join(rows)+'\n')
    (out/'data.yaml').write_text(f'path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames: [robot]\n')
    (out/'manifest.json').write_text(json.dumps({'classes':['robot'],'scope':review['scope'],'reviewed_boxes':all_boxes,'train_times':review['train_times'],'val_times':review['validation_times'],'test_times':review['test_times'],'sampling_repeats':4},indent=2)+'\n')
    print('Original dataset plus eight independent Glendale training frames (4x sampling), two validation and two test frames.')


if __name__=='__main__':main()
