"""Check sequential teacher boxes against checked frames before bulk labeling."""
import argparse
import json
import threading
from pathlib import Path

from PIL import Image
from evaluate_detector import count_matches
from teacher_labels import Teacher, atomic_json
from track_llm import TEAMS
from train_detector import heartbeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/teacher_v2"))
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    teacher = Teacher(args.root / "billing")
    audit = args.root / "tracking_audit"
    (audit / "frames").mkdir(parents=True, exist_ok=True)
    prior = json.loads(Path("track/final/f_0000.json").read_text())["robots"]
    prior = [{**r, "box": [r["box"][0], r["box"][1] - 110, r["box"][2], r["box"][3] - 110]} for r in prior]
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    totals = {str(th): dict(true_positive=0, false_positive=0, false_negative=0) for th in (.3, .5)}
    records = []
    try:
        for i in range(args.frames + 1):
            stem = f"f_{i:04d}"
            frame = audit / "frames" / (stem + ".jpg")
            if not frame.exists():
                with Image.open(Path("track/clip_frames") / frame.name) as image:
                    image.crop((0, 110, 1920, 720)).save(frame, quality=95)
            if i == 0:
                previous = frame
                continue
            data = teacher.label(frame, audit / "labels" / (stem + ".json"), teams=TEAMS, previous=previous, prior=prior, gap=.2)
            refs = [{"class_id": 0, "box": [r["box"][0], r["box"][1] - 110, r["box"][2], r["box"][3] - 110]}
                    for r in json.loads((Path("track/final") / (stem + ".json")).read_text())["robots"] if r["visible"]]
            predictions = [{"class_id": 0, "box": r["box"]} for r in data["robots"] if r["visible"]]
            for threshold in (.3, .5):
                for key, value in count_matches(predictions, refs, threshold).items():
                    totals[str(threshold)][key] += value
            records.append({"frame": frame.name, "predictions": predictions, "references": refs})
            prior, previous = data["robots"], frame
            print(f"Audit {i}/{args.frames}: {len(predictions)} detections; spend ${teacher.ledger['spent_usd']:.4f}", flush=True)
        for counts in totals.values():
            tp, fp, fn = (counts[k] for k in ("true_positive", "false_positive", "false_negative"))
            counts.update(precision=tp / (tp + fp) if tp + fp else 0, recall=tp / (tp + fn) if tp + fn else 0)
        passed = totals["0.5"]["precision"] >= .90 and totals["0.5"]["recall"] >= .90
        report = {"metrics": totals, "passed": passed, "prompt_sha256": teacher.prompt_hash,
                  "seed_frame": 0, "frames": records, "scope": "Teacher detection quality on checked old-match frames; identities not audited."}
        atomic_json(audit / "metrics.json", report)
        print(json.dumps({"passed": passed, "metrics": totals}, indent=2), flush=True)
        if not passed:
            raise SystemExit("Teacher did not meet the quality gate; bulk labeling must not proceed")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
