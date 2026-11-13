"""Export saved tracking labels and frames as a local robot detection dataset.

No API calls. Run with .venv/bin/python prepare_dataset.py --help.
"""
import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

FIELD_CROP = (0, 110, 1920, 720)
ALLIANCES = {"868": "blue", "2910": "blue", "2046": "blue",
             "4065": "red", "4414": "red", "1323": "red"}


def split_frames(files, gap):
    """Chronological 70/15/15 split, with `gap` excluded frames per boundary."""
    n = len(files)
    boundaries = (int(n * .7), int(n * .85))
    result = {}
    for i, path in enumerate(files):
        if any(b - gap // 2 <= i < b + (gap + 1) // 2 for b in boundaries):
            result[path.stem] = "excluded"
        else:
            result[path.stem] = "train" if i < boundaries[0] else "val" if i < boundaries[1] else "test"
    if any(s not in result.values() for s in ("train", "val", "test")):
        raise ValueError("Not enough frames for three splits and this gap; add frames or reduce --gap-frames")
    return result


def convert_robot(robot, crop, classes):
    if type(robot.get("visible")) is not bool:
        raise ValueError("Each robot needs a boolean visible field")
    if not robot["visible"]:
        return None, "hidden"
    box = robot.get("box")
    if not isinstance(box, list) or len(box) != 4 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError("Visible robot needs four finite box coordinates")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive width and height")
    left, top, right, bottom = crop
    x1, x2 = max(x1, left), min(x2, right)
    y1, y2 = max(y1, top), min(y2, bottom)
    if x2 <= x1 or y2 <= y1:
        return None, "outside_crop"
    w, h = right - left, bottom - top
    class_id = 0
    if classes == "alliance":
        team = robot.get("team")
        if team not in ALLIANCES:
            raise ValueError(f"Unknown team {team!r}; update ALLIANCES before exporting another match")
        class_id = 0 if ALLIANCES[team] == "red" else 1
    xywh = [(x1 + x2 - 2 * left) / (2 * w), (y1 + y2 - 2 * top) / (2 * h),
            (x2 - x1) / w, (y2 - y1) / h]
    return {"class_id": class_id, "xywh": xywh, "team": robot.get("team")}, None


def make_preview(root, split, records, names):
    selected = records if len(records) <= 6 else [records[round(i * (len(records) - 1) / 5)] for i in range(6)]
    sheet = Image.new("RGB", (960, 3 * 195), "#202020")
    for i, record in enumerate(selected):
        with Image.open(root / "images" / split / record["image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in (root / "labels" / split / (Path(record["image"]).stem + ".txt")).read_text().splitlines():
            c, cx, cy, w, h = map(float, line.split())
            box = [(cx - w / 2) * image.width, (cy - h / 2) * image.height,
                   (cx + w / 2) * image.width, (cy + h / 2) * image.height]
            color = "#ff5050" if names[int(c)] == "robot_red" else "#50b0ff" if names[int(c)] == "robot_blue" else "#50ff80"
            draw.rectangle(box, outline=color, width=4)
        thumb = ImageOps.contain(image, (480, 165))
        x, y = i % 2 * 480, i // 2 * 195
        sheet.paste(thumb, (x, y + 25))
        ImageDraw.Draw(sheet).text((x + 8, y + 6), f"{split}: {record['image']}", fill="white")
    sheet.save(root / f"preview_{split}.jpg", quality=92)


def export_dataset(frames_dir, labels_dir, out_dir, crop=FIELD_CROP, classes="robot", gap=10):
    frames_dir, labels_dir, out_dir = map(Path, (frames_dir, labels_dir, out_dir))
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError(f"Output already exists: {out_dir}. Choose a new folder.")
    if classes not in ("robot", "alliance") or gap < 0:
        raise ValueError("Invalid class mode or negative split gap")
    if len(crop) != 4 or crop[0] < 0 or crop[1] < 0 or crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError("Invalid crop rectangle")
    files = sorted(labels_dir.glob("f_*.json"))
    if not files:
        raise ValueError(f"No f_*.json labels in {labels_dir}")
    splits = split_frames(files, gap)
    records = []
    names = ["robot"] if classes == "robot" else ["robot_red", "robot_blue"]
    # Validate every source before creating output; never silently drop missing frames.
    for path in files:
        data = json.loads(path.read_text())
        image_name = path.stem + ".jpg"
        if data.get("frame", image_name) != image_name:
            raise ValueError(f"Frame name disagrees with filename: {path}")
        image_path = frames_dir / image_name
        if not image_path.is_file():
            raise ValueError(f"Missing image: {image_path}. Restore the frames using DETAILS.md.")
        with Image.open(image_path) as image:
            image.verify()
            if image.size != (1920, 1080):
                raise ValueError(f"Expected 1920x1080 source image: {image_path}")
        if crop[2] > 1920 or crop[3] > 1080:
            raise ValueError("Crop exceeds source image dimensions")
        boxes, skipped = [], Counter()
        for robot in data["robots"]:
            box, reason = convert_robot(robot, crop, classes)
            if reason:
                skipped[reason] += 1
            else:
                boxes.append(box)
        records.append({"image": image_name, "split": splits[path.stem], "boxes": boxes,
                        "skipped": dict(skipped), "source_label": str(path.resolve()),
                        "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset-", dir=out_dir.parent) as temp:
        root = Path(temp) / "export"
        root.mkdir()
        summary = {}
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            selected = [r for r in records if r["split"] == split]
            for record in selected:
                with Image.open(frames_dir / record["image"]) as image:
                    image.crop(crop).save(root / "images" / split / record["image"], quality=95)
                lines = [str(b["class_id"]) + " " + " ".join(f"{v:.8f}" for v in b["xywh"]) for b in record["boxes"]]
                (root / "labels" / split / (Path(record["image"]).stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""))
            summary[split] = {"images": len(selected), "boxes": sum(len(r["boxes"]) for r in selected)}
            make_preview(root, split, selected, names)
        manifest = {"source_frames": str(frames_dir.resolve()), "source_labels": str(labels_dir.resolve()),
                    "crop": list(crop), "classes": names, "gap_frames": gap, "summary": summary,
                    "excluded_frames": sum(r["split"] == "excluded" for r in records), "frames": records,
                    "warning": "Single-match time split only; not evidence of generalization to another match or venue."}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Absolute path is JSON-quoted, which is also a valid YAML scalar.
        (root / "data.yaml").write_text(f"path: {json.dumps(str(out_dir.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
                                        "".join(f"  {i}: {name}\n" for i, name in enumerate(names)))
        root.rename(out_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("out_dir", type=Path, help="New folder; existing output is never overwritten")
    parser.add_argument("--classes", choices=("robot", "alliance"), default="robot")
    parser.add_argument("--crop", nargs=4, type=int, default=FIELD_CROP, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--gap-frames", type=int, default=10, help="Excluded frames at each chronological split boundary")
    args = parser.parse_args()
    try:
        manifest = export_dataset(args.frames_dir, args.labels_dir, args.out_dir, args.crop, args.classes, args.gap_frames)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Dataset not exported: {error}\n")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Excluded {manifest['excluded_frames']} gap frames. Previews and manifest: {args.out_dir}")


if __name__ == "__main__":
    main()
