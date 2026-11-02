"""Haiku-only job: extract, label, filter, train, evaluate. No model substitutions."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from PIL import Image
from evaluate_detector import iou
from prepare_dataset import convert_robot, make_preview
from teacher_labels import MODEL, Teacher, atomic_json
from train_detector import heartbeat


def training_boxes(data, width=1920, height=610):
    """Conservative pseudo-labels: avoid treating omitted robots as background."""
    if not data.get("usable") or len(data.get("robots", [])) != 6:
        return None, "not_six_robots"
    if any(not r.get("visible") for r in data["robots"]):
        return None, "hidden_or_incomplete"
    if any(r.get("confidence", 0) < .5 for r in data["robots"]):
        return None, "low_confidence"
    for i, robot in enumerate(data["robots"]):
        if any(iou(robot["box"], other["box"]) > .7 for other in data["robots"][:i]):
            return None, "duplicate_or_ambiguous"
    boxes = [convert_robot(r, (0, 0, width, height), "robot")[0] for r in data["robots"]]
    if any(b is None for b in boxes):
        return None, "outside_crop"
    return boxes, None


def extract_frames(root, source, plan):
    import imageio_ffmpeg
    video = root / "sources" / (source["match"] + ".mp4")
    if not video.is_file():
        raise ValueError(f"Missing source video: {video}")
    frames = root / "frames" / source["match"]
    frames.mkdir(parents=True, exist_ok=True)
    marker = frames / "extraction.json"
    contract = {"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "crop": plan["crop"],
                "fps": plan["label_fps"], "start_s": source["start_s"], "end_s": source["end_s"]}
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["contract"] != contract:
            raise ValueError("Source extraction changed; use a new job root")
        if len(list(frames.glob("f_*.jpg"))) != previous["frames"]:
            raise ValueError("Extracted frames are missing; restore before resuming")
        return frames
    if list(frames.glob("f_*.jpg")):
        raise ValueError("Incomplete extraction: move partial frames aside before resuming")
    x1, y1, x2, y2 = plan["crop"]
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-stats", "-stats_period", "15", "-n",
                    "-ss", str(source["start_s"]), "-to", str(source["end_s"]), "-i", str(video),
                    "-vf", f"fps={plan['label_fps']},crop={x2-x1}:{y2-y1}:{x1}:{y1}",
                    "-q:v", "2", "-start_number", "0", str(frames / "f_%04d.jpg")], check=True)
    count = len(list(frames.glob("f_*.jpg")))
    expected = round((source["end_s"] - source["start_s"]) * plan["label_fps"])
    if count != expected:
        raise ValueError(f"Expected {expected} frames from {video}, got {count}; check source timing")
    with Image.open(frames / "f_0000.jpg") as image:
        if image.size != (1920, 610):
            raise ValueError("Unexpected video dimensions; source crop must be reviewed")
    atomic_json(marker, {"contract": contract, "frames": count})
    return frames


def build_dataset(root, plan):
    out = root / "dataset"
    if (out / "manifest.json").exists():
        return out
    if out.exists():
        raise ValueError("Partial dataset exists; inspect it before restarting")
    out.mkdir()
    records, rejected = [], Counter()
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for source in plan["sources"]:
        for label_path in sorted((root / "labels" / source["match"]).glob("f_*.json"))[::plan["export_stride"]]:
            data = json.loads(label_path.read_text())
            boxes, reason = training_boxes(data)
            if reason:
                rejected[reason] += 1
                continue
            name = source["match"] + "_" + label_path.stem + ".jpg"
            split = source["split"]
            frame = root / "frames" / source["match"] / (label_path.stem + ".jpg")
            if hashlib.sha256(frame.read_bytes()).hexdigest() != data["image_sha256"]:
                raise ValueError("Source image changed after labeling")
            shutil.copyfile(frame, out / "images" / split / name)
            (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join("0 " + " ".join(f"{v:.8f}" for v in b["xywh"]) for b in boxes) + "\n")
            records.append({"image": name, "split": split, "source_match": source["match"],
                            "source_label": str(label_path.resolve()), "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                            "boxes": boxes, "label_type": "unreviewed_teacher"})
    # Preserve the old checked test block; never add it to training or teacher context.
    checked = Path("datasets/checked_clip_v1")
    for frame in sorted((checked / "images/test").glob("*.jpg")):
        shutil.copyfile(frame, out / "images/test" / frame.name)
        shutil.copyfile(checked / "labels/test" / (frame.stem + ".txt"), out / "labels/test" / (frame.stem + ".txt"))
        records.append({"image": frame.name, "split": "test", "source_match": "2026cmptx_f1m1", "label_type": "checked_answer_key"})
    summary = {s: {"images": sum(r["split"] == s for r in records),
                   "boxes": sum(len((out / "labels" / s / (Path(r["image"]).stem + ".txt")).read_text().splitlines())
                                for r in records if r["split"] == s)} for s in ("train", "val", "test")}
    for split, minimum in plan["minimum_images"].items():
        if summary[split]["images"] < minimum:
            raise ValueError(f"Too few accepted {split} images: {summary[split]['images']}; inspect labels before training")
    if summary["test"]["images"] != 14:
        raise ValueError("The original 14-image checked test set is missing")
    for split in ("train", "val", "test"):
        make_preview(out, split, [r for r in records if r["split"] == split], ["robot"])
    (out / "data.yaml").write_text(f"path: {json.dumps(str(out.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: robot\n")
    atomic_json(out / "manifest.json", {"classes": ["robot"], "crop": plan["crop"], "sources": plan["sources"],
        "summary": summary, "frames": records, "rejected": dict(rejected),
        "warning": "Teacher labels are pseudo-labels, not human-verified ground truth. Six-visible filtering biases against occlusions. Test set has only 14 frames."})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/haiku_teacher_v2.json"))
    parser.add_argument("--root", type=Path, default=Path("datasets/teacher_v2"))
    parser.add_argument("--out", type=Path, default=Path("runs/local_robot_v2"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if plan["model"] != MODEL or not 0 < plan["budget_usd"] <= 4:
        parser.error("Only Haiku 5.5 within the approved $4 budget is authorized")
    contract = root / "job_plan.json"
    if contract.exists() and json.loads(contract.read_text()) != plan:
        parser.error("Plan changed; existing job must be reviewed rather than silently reused")
    atomic_json(contract, plan)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    stage = "preflight"
    def status(state, **extra):
        atomic_json(root / "status.json", {"state": state, "stage": stage, "updated_unix": time.time(), **extra})
    try:
        audit = json.loads((root / "tracking_audit/metrics.json").read_text())
        teacher = Teacher(root / "billing", cap=plan["budget_usd"])
        if not audit["passed"] or audit["prompt_sha256"] != teacher.prompt_hash:
            raise ValueError("A matching teacher quality audit must pass before bulk labeling")
        for source in plan["sources"]:
            stage = "extracting " + source["match"]
            status("running")
            print(stage, flush=True)
            frames = extract_frames(root, source, plan)
            prior, previous, previous_index = None, None, None
            stage = "labeling " + source["match"]
            paths = sorted(frames.glob("f_*.jpg"))
            for index, frame in enumerate(paths):
                target = root / "labels" / source["match"] / (frame.stem + ".json")
                digest = hashlib.sha256(frame.read_bytes()).hexdigest()
                if not target.exists() and any(a["image_sha256"] == digest and a["status"] == "billed"
                                               for a in teacher.ledger["attempts"]):
                    quarantine_path = root / "quarantine.json"
                    quarantine = json.loads(quarantine_path.read_text()) if quarantine_path.exists() else {}
                    quarantine[digest] = {"image": str(frame), "reason": "Previously billed response has no usable label; excluded without retry"}
                    atomic_json(quarantine_path, quarantine)
                    print(f"Excluded already-billed frame {frame.name}; no duplicate request", flush=True)
                    continue
                if not target.exists() and any(a["image_sha256"] == digest and a["status"] in ("pending", "failed_or_uncertain")
                                                for a in teacher.ledger["attempts"]):
                    raise ValueError("This frame has an uncertain earlier API attempt; reconcile it before any retry")
                gap = (index - previous_index) / plan["label_fps"] if previous_index is not None else 1 / plan["label_fps"]
                data = teacher.label(frame, target, teams=source["teams"], previous=previous, prior=prior, gap=gap)
                previous, prior, previous_index = frame, data["robots"], index
                status("running", match=source["match"], frame=index + 1, frames=len(paths),
                       spent_usd=teacher.ledger["spent_usd"], reserved_usd=teacher.ledger["reserved_usd"])
                print(f"{source['match']} {index+1}/{len(paths)}; visible={sum(r['visible'] for r in prior)}; spend=${teacher.ledger['spent_usd']:.4f}", flush=True)
        teacher.lock.close()
        stage = "building dataset"
        status("running")
        dataset = build_dataset(root, plan)
        stage = "training"
        status("running")
        training = plan["training"]
        subprocess.run([sys.executable, "-u", "train_detector.py", "--data", str(dataset / "data.yaml"), "--out", str(args.out),
                        "--epochs", str(training["epochs"]), "--imgsz", str(training["imgsz"]), "--batch", str(training["batch"])], check=True)
        stage = "evaluating"
        status("running")
        evaluation = args.out.with_name(args.out.name + "_test")
        subprocess.run([sys.executable, "-u", "evaluate_detector.py", str(args.out / "weights/best.pt"), str(dataset), str(evaluation)], check=True)
        metrics = json.loads((evaluation / "metrics.json").read_text())
        baseline = json.loads(Path("results/local_robot_v1.json").read_text())
        summary = {"run": args.out.name, "model": "YOLO26n", "teacher": MODEL,
                   "teacher_billing": json.loads((root / "billing/ledger.json").read_text())["spent_usd"],
                   "metrics": metrics["metrics"], "baseline_metrics": baseline["metrics"],
                   "dataset_summary": json.loads((dataset / "manifest.json").read_text())["summary"],
                   "checkpoint": str(args.out / "weights/best.pt"), "artifacts": str(evaluation),
                   "limitation": "14 checked test images; teacher pseudo-labels and no team identity model."}
        atomic_json(root / "result.json", summary)
        stage = "complete"
        status("complete", result=summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        # Avoid writing exception bodies that might contain request headers or secrets.
        status("stopped", error_type=type(error).__name__)
        print(f"Job stopped at {stage}: {type(error).__name__}; inspect local logs. No automatic retries.", flush=True)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
