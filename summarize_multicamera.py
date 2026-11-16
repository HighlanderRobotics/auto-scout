"""Build a local review report from saved experiments, without rerunning models."""
import argparse
import html
import json
from collections import Counter
from pathlib import Path

from teacher_labels import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path('datasets/temporal_multicamera_v1_run');ledger=read(root/'ledger.json')
    training=read(root/'result.json')
    side={key:read(f'runs/multicamera_holdout_v2_{key}/metrics.json')['metrics'] for key in ['baseline','student']}
    naming={key:read(f'runs/einstein_overnight_{key}_v1_score/metrics.json') for key in ['baseline','student']}
    tracked={key:read(f'runs/einstein_overnight_{key}_v1/paths.json')['reference_score'] for key in ['baseline','student']}
    appearance=read('runs/temporal_multicamera_v1_ocr_appearance/metrics.json')
    resnet=read('runs/temporal_multicamera_v1_resnet_appearance/metrics.json')
    report=dict(date='2026-11-16',run='temporal_multicamera_v1',
        verdict='Useful camera-specific detector improvement and higher checked named coverage; not ready for unattended six-robot scouting.',
        global_default_changed=False,
        billing=dict(spent_usd=ledger['spent_usd'],reserved_usd=ledger['reserved_usd'],cap_usd=ledger['cap_usd'],
                     calls_by_model=dict(Counter(a['model'] for a in ledger['attempts'] if a['status']=='billed'))),
        dataset=training['dataset'],einstein_detector_test=training['einstein_test'],
        pnw_teacher_validation=dict(student=training['pnw_teacher_validation'],baseline=training['baseline_pnw_teacher_validation']),
        side_camera_holdout=side,side_reference_type=read('configs/multicamera_holdout_v2.json')['reference_type'],
        side_annotation_revision=read('configs/multicamera_holdout_v2.json')['annotation_revision'],
        einstein_tracked_boxes=tracked,einstein_named_coverage=naming,
        appearance_experiments=dict(ocr_hsv=dict(training_crops=appearance['training_crops'],test_crops=appearance['test_crops'],
            correct=round(appearance['agreement']*appearance['test_crops']),agreement=appearance['agreement'],
            unweighted_agreement=appearance['unweighted_agreement']),resnet=dict(training_crops=resnet['training_crops'],test_crops=resnet['test_crops'],
            correct=resnet['correct_pseudo_labels'],agreement=resnet['agreement']),
            decision='Do not promote either classifier as an automatic team-number source.'),
        local_multicamera_demo=read('runs/wisconsin_multicamera_local_v1/summary.json'),
        local_memory_demo=read('runs/wisconsin_multicamera_local_memory_v1/summary.json'),
        checkpoint='runs/temporal_multicamera_v1/weights/best.pt',
        checkpoint_sha256=read('runs/multicamera_holdout_v2_student/metrics.json')['weights_sha256'],
        artifacts=dict(wisconsin_viewer='runs/wisconsin_multicamera_local_v1/viewer.html',
            einstein_student='runs/einstein_overnight_student_v1/memory/teams/animation.html',
            einstein_baseline='runs/einstein_overnight_baseline_v1/memory/teams/animation.html',
            side_preview='runs/multicamera_holdout_v2_student/preview.jpg'),
        limitations=['Only a small same-camera side benchmark, with agent-reviewed annotations and explicit ignored regions.',
                    'One benchmark annotation was corrected after inspecting predictions; both earlier scores remain available.',
                    'PNW validation uses teacher labels, not human ground truth.',
                    'Einstein is a small historical same-match benchmark; correct naming accuracy is conditional on matched observations.',
                    'Camera-specific crops and field mappings remain manual. No automatic camera-cut detection or geometric camera fusion.',
                    'The 14-image old-camera test gained one match but six extra detections; retain the global baseline.'])
    atomic_json(args.out/'results.json',report)
    def row(label,old,new):return f'<tr><th>{html.escape(label)}</th><td>{old}</td><td>{new}</td></tr>'
    def counts(m):return f"{m['true_positive']} matched / {m['false_positive']} extras / {m['false_negative']} misses"
    rows=row('Later Wisconsin side frames, IoU ≥ 0.5',counts(side['baseline']['0.5']),counts(side['student']['0.5']))
    rows+=row('Correct named Einstein observations',f"{naming['baseline']['correct_identity']} / 581",f"{naming['student']['correct_identity']} / 581")
    rows+=row('Correct named reference coverage',f"{naming['baseline']['correct_named_reference_coverage']:.1%}",f"{naming['student']['correct_named_reference_coverage']:.1%}")
    rows+=row('Wrong names on matched observations',naming['baseline']['matched_iou_0_3']-naming['baseline']['correct_identity'],naming['student']['matched_iou_0_3']-naming['student']['correct_identity'])
    rows+=row('PNW teacher-label agreement, IoU ≥ 0.5',counts(training['baseline_pnw_teacher_validation']['0.5']),counts(training['pnw_teacher_validation']['0.5']))
    limits=''.join(f'<li>{html.escape(s)}</li>' for s in report['limitations'])
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Auto-Scout overnight results</title><style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#e5e7eb;background:#101720}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:32px}}p{{max-width:950px}}a{{color:#93c5fd}}table{{border-collapse:collapse;width:100%;background:#182330}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #334155}}video,img{{width:100%;background:#000;border-radius:8px}}.badge{{display:inline-block;background:#25453c;color:#b6f7d5;padding:7px 12px;border-radius:6px}}.stats{{display:flex;gap:20px;flex-wrap:wrap}}.stats p{{background:#182330;padding:18px 24px;border-radius:8px}}small{{color:#b0bdcb}}</style>
    <p>Auto-Scout · November 16, 2026</p><h1>Better side-camera detection.<br>Team tracking is still incomplete.</h1><p class="badge">Experimental improvement — existing global default retained</p>
    <div class="stats"><p><strong>64.9%</strong><br>correct named Einstein coverage<br><small>up from 39.1%, same pipeline</small></p><p><strong>16 / 17</strong><br>later side-camera robots matched<br><small>3 extras; small agent-labeled check</small></p><p><strong>${ledger['spent_usd']:.2f} / $4.90</strong><br>total paid teaching cost<br><small>all later experiments ran locally</small></p></div>
    <p>The student detector finds robots; it does not read team numbers. Better boxes and associations helped the existing OCR/color resolver name more observations. Four Einstein teams are supported; 868 and 2910 remain unresolved in this local run.</p>
    <h2>Checked comparisons</h2><table><tr><th>Measure</th><th>Original detector</th><th>New student</th></tr>{rows}</table><p><small>Einstein: 100 checked timestamps / 581 visible reference observations. Named observations must overlap the answer key; estimated path points do not count. Side-camera references include a documented correction after visual review; original scores are preserved.</small></p>
    <h2>Synchronized local-camera demo</h2><video controls preload="metadata" poster="../wisconsin_multicamera_local_v1/preview.jpg" src="../wisconsin_multicamera_local_v1/synchronized_tracking.mp4"></video><p><a href="../wisconsin_multicamera_local_v1/viewer.html">Open camera viewer and identity evidence</a>. Camera-local IDs are separate. Repeated bumper reads support names; unknown robots remain visible. There is no spatial camera fusion.</p>
    <h2>Einstein source video and field paths</h2><p><a href="../einstein_overnight_student_v1/memory/teams/animation.html">New student</a> · <a href="../einstein_overnight_baseline_v1/memory/teams/animation.html">Original detector</a> · <a href="../einstein_overnight_student_v1_score/comparison.txt">Answer-key comparison</a></p>
    <h2>What I rejected</h2><p>The learned HSV appearance metric agreed with exact OCR on 19/23 held-out crops, equal to the unweighted method. A frozen ResNet18 encoder agreed on 17/23. Neither is reliable enough to assign team names by itself. Extra visual memory in the Wisconsin demo did not increase named observations (151/467 in both runs), so it remains optional.</p>
    <h2>Limits and next useful work</h2><ul>{limits}</ul><p>The next useful step is an independently human-checked side-camera identity set, then targeted work on missing identities and crossings. More paid labeling without that check could reinforce mistakes.</p><p><a href="results.json">Full measured results</a> · <a href="https://github.com/HighlanderRobotics/auto-scout/pull/10">Code review</a></p></html>'''
    (args.out/'index.html').write_text(content)
    print('Report:',args.out/'index.html',flush=True)


if __name__=='__main__':main()
