"""Build a local review page from a saved NorCal identity experiment result JSON."""
import argparse
import html
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    data=json.loads(a.result.read_text());a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    def link(path):return html.escape(os.path.relpath(Path(path).resolve(),a.out.resolve()),quote=True)
    rows=[]
    for name,check in data['checks'].items():
        rows.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{check[k]}</td>' for k in ['checkpoints','correct','wrong','unknown','missing'])+'</tr>')
    videos=[]
    for title,path in data.get('comparison_videos',{}).items():
        videos.append(f'<article><h2>{html.escape(title)}</h2><video controls preload="metadata" src="{link(path)}"></video></article>')
    scope=html.escape(data['limitations']);verdict=html.escape(data['status'])
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NorCal robot identity experiment</title>
<style>body{{background:#10151d;color:#e6edf3;font:17px/1.6 system-ui;margin:0}}main{{max-width:1150px;margin:auto;padding:35px 22px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px}}.status,article{{padding:20px;background:#1a2430;border-radius:12px;margin:22px 0}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #354252}}video{{width:100%;background:black}}a{{color:#87c9ff}}.scroll{{overflow:auto}}small{{color:#b0bdcc}}code{{background:#243142;padding:2px 5px}}</style>
<main><h1>NorCal robot team identity</h1><p class="status">{verdict}</p>
<p>One focus: identify the same physical robots despite crossings and brief occlusions. Detector weights and observed boxes stay fixed. Unknowns and missing detections remain visible in the scores.</p>
<h2>Observed identity checks</h2><div class="scroll"><table><thead><tr><th>Run / scope</th><th>References</th><th>Correct</th><th>Wrong</th><th>Unknown</th><th>Missing</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><small>{scope}</small></p>{''.join(videos)}
<h2>What changed</h2><p>Full-field temporal proposals were too uncertain. Independent close-up teaching separates physical robots from people, monitors and field structures. A local identity head learns from reviewed robot crops and enrollment cards, using frozen ResNet18 and HSV features. It names each observed box conservatively, without assuming a tracker ID always stays on the same robot.</p>
<p>These identity scores do not measure complete paths or every ID switch. A separate match exposed poor transfer in the first head. More training data can also contain confident teacher mistakes, so source review remains necessary.</p>
<h2>Data, spending and reproduction</h2><p>Training source crops: {data['training_source_crops']}. Augmented embeddings: {data['augmented_embeddings']}. Local identity inference: $0 API.</p>
<p>Total shared Claude ledger: ${data['claude_total_usd']:.4f} / $4.90. This phase: ${data['claude_phase_usd']:.4f}. Existing model sublimits are enforced; earlier work is included.</p>
<p><a href="result.json">Saved result JSON</a> · <a href="{link(data['model_path'])}">Experimental identity model</a> · <a href="https://github.com/HighlanderRobotics/auto-scout/pull/11">Draft pull request</a></p>
<p>Use <code>.venv/bin/python identity_student.py infer --model MODEL_FOLDER --tracks TRACKS_JSON --out NEW_FOLDER</code>. No paid calls. The default pipeline is unchanged.</p></main></html>'''
    (a.out/'index.html').write_text(content);print(a.out/'index.html')


if __name__=='__main__':main()
