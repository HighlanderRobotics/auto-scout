"""Build an offline detector dataset from confident regions of saved teacher labels."""
import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image
from evaluate_detector import count_matches, iou
from prepare_dataset import make_preview


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def clip(box, crop):
    return [max(box[0], crop[0]), max(box[1], crop[1]), min(box[2], crop[2]), min(box[3], crop[3])]


def overlap(box, crop):
    return area(clip(box, crop))


def select_region(data, width=1920, height=610):
    if not data.get("usable") or not isinstance(data.get("robots"), list):
        return None
    positives, uncertain = [], []
    for robot in data["robots"]:
        box = robot.get("box")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            return None
        w, h = box[2] - box[0], box[3] - box[1]
        trusted = (robot.get("visible") is True and robot.get("confidence", 0) >= .5 and
                   0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height and
                   12 <= w <= 480 and 12 <= h <= 400)
        if trusted:
            positives.append(robot)
        elif area(box) > 0:
            uncertain.append(box)
    if not positives:
        return None
    candidates = []
    for crop_width in (1920, 1280, 960, 640):
        if crop_width > width:
            continue
        for x in sorted({0, (width - crop_width) // 2, width - crop_width}):
            crop = [x, 0, x + crop_width, height]
            if any(overlap(box, crop) > 0 for box in uncertain):
                continue
            included = [r for r in positives if overlap(r["box"], crop) > 0]
            if not included or any(overlap(r["box"], crop) / area(r["box"]) < .5 for r in included):
                continue
            boxes = [clip(r["box"], crop) for r in included]
            if any(b[2] - b[0] < 8 or b[3] - b[1] < 8 for b in boxes):
                continue
            if any(iou(a, b) > .7 for i, a in enumerate(boxes) for b in boxes[:i]):
                continue
            candidates.append({"crop": crop, "robots": included, "boxes": boxes})
    return max(candidates, key=lambda c: (len(c["boxes"]), c["crop"][2] - c["crop"][0], -c["crop"][0])) if candidates else None


def calibrate(root):
    counts, frames = Counter(), 0
    for path in sorted((root / "tracking_audit/labels").glob("f_*.json")):
        selected = select_region(json.loads(path.read_text()))
        if not selected:
            continue
        crop = selected["crop"]
        references = []
        for robot in json.loads((Path("track/final") / path.name).read_text())["robots"]:
            box = [robot["box"][0], robot["box"][1] - 110, robot["box"][2], robot["box"][3] - 110]
            if robot["visible"] and overlap(box, crop) > 0:
                references.append({"class_id": 0, "box": clip(box, crop)})
        predictions = [{"class_id": 0, "box": b} for b in selected["boxes"]]
        counts.update(count_matches(predictions, references, .5))
        frames += 1
    tp, fp, fn = (counts[k] for k in ("true_positive", "false_positive", "false_negative"))
    return {"frames": frames, **dict(counts), "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "scope": "Old-match calibration frames 1-40 only, not student test images."}


def export(root, plan_path, reviews_path, out):
    if out.exists() or out.is_symlink():
        raise ValueError("Choose a new output folder")
    plan = json.loads(plan_path.read_text())
    reviews = json.loads(reviews_path.read_text())
    records, rejected = [], Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="regions-") as temp:
        staging = Path(temp) / "dataset"
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        for source in plan["sources"]:
            for path in sorted((root / "labels" / source["match"]).glob("f_*.json")):
                if int(path.stem[2:]) % 3:
                    continue
                key = source["match"] + "/" + path.stem
                review = reviews["frames"].get(key)
                if review and review["decision"] == "reject":
                    rejected["visual_review"] += 1
                    continue
                data = json.loads(path.read_text())
                selected = select_region(data)
                if selected is None:
                    rejected["no_confident_region"] += 1
                    continue
                frame = root / "frames" / source["match"] / (path.stem + ".jpg")
                if hashlib.sha256(frame.read_bytes()).hexdigest() != data["image_sha256"]:
                    raise ValueError(f"Source hash mismatch: {frame}")
                crop = selected["crop"]
                if review and crop != review["crop"]:
                    raise ValueError(f"Reviewed crop changed: {key}")
                name = source["match"] + "_" + path.stem + ".jpg"
                split = source["split"]
                with Image.open(frame) as image:
                    if image.size != (1920, 610):
                        raise ValueError("Unexpected source dimensions")
                    image.crop(crop).save(staging / "images" / split / name, quality=95)
                w, h = crop[2] - crop[0], crop[3] - crop[1]
                boxes = [[((b[0] + b[2]) / 2 - crop[0]) / w, ((b[1] + b[3]) / 2 - crop[1]) / h,
                          (b[2] - b[0]) / w, (b[3] - b[1]) / h] for b in selected["boxes"]]
                (staging / "labels" / split / (Path(name).stem + ".txt")).write_text(
                    "\n".join("0 " + " ".join(f"{v:.8f}" for v in box) for box in boxes) + "\n")
                records.append({"image": name, "split": split, "source_match": source["match"], "crop": crop,
                                "source_label": str(path.resolve()), "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "boxes": boxes, "label_type": "teacher_confident_region", "spot_review": review})
        checked = Path("datasets/checked_clip_v1")
        for frame in sorted((checked / "images/test").glob("*.jpg")):
            shutil.copyfile(frame, staging / "images/test" / frame.name)
            shutil.copyfile(checked / "labels/test" / (frame.stem + ".txt"), staging / "labels/test" / (frame.stem + ".txt"))
            records.append({"image": frame.name, "split": "test", "label_type": "checked_answer_key"})
        summary = {s: {"images": sum(r["split"] == s for r in records),
                       "boxes": sum(len((staging / "labels" / s / (Path(r["image"]).stem + ".txt")).read_text().splitlines())
                                    for r in records if r["split"] == s)} for s in ("train", "val", "test")}
        if summary["train"]["images"] < 100 or summary["val"]["images"] < 30 or summary["test"]["images"] != 14:
            raise ValueError(f"Insufficient dataset: {summary}")
        calibration = calibrate(root)
        if calibration["frames"] < 5 or calibration["precision"] < .9:
            raise ValueError(f"Region calibration failed: {calibration}")
        for split in summary:
            make_preview(staging, split, [r for r in records if r["split"] == split], ["robot"])
        (staging / "data.yaml").write_text(f"path: {json.dumps(str(out.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: robot\n")
        manifest = {"classes": ["robot"], "summary": summary, "frames": records, "rejected": dict(rejected),
                    "calibration": calibration, "reviews": reviews,
                    "policy": "Keep confidence >=0.5; exclude any crop intersecting an uncertain visible/hidden estimate. Max one crop per source frame, fixed stride 3. No API calls.",
                    "warning": "Pseudo-labels still have errors and omissions; reviewed samples do not verify every crop. Crops change object scale. Test is 14 full-field images."}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(out)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/teacher_v2"))
    parser.add_argument("--plan", type=Path, default=Path("configs/haiku_teacher_v2.json"))
    parser.add_argument("--reviews", type=Path, default=Path("configs/confident_region_reviews.json"))
    parser.add_argument("--out", type=Path, default=Path("datasets/confident_regions_v1"))
    args = parser.parse_args()
    manifest = export(args.root, args.plan, args.reviews, args.out)
    print(json.dumps({"summary": manifest["summary"], "calibration": manifest["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
