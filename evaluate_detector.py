"""Score local detections, counting missed robots and false positives explicitly."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def count_matches(predictions, references, threshold):
    """Maximum-cardinality one-to-one same-class matching above an IoU threshold."""
    edges = [[j for j, ref in sorted(enumerate(references),
              key=lambda item: iou(pred["box"], item[1]["box"]), reverse=True)
              if pred["class_id"] == ref["class_id"] and iou(pred["box"], ref["box"]) >= threshold]
             for pred in predictions]
    owners = {}

    def assign(index, visited):
        for ref in edges[index]:
            if ref in visited:
                continue
            visited.add(ref)
            if ref not in owners or assign(owners[ref], visited):
                owners[ref] = index
                return True
        return False

    matched = sum(assign(i, set()) for i in range(len(predictions)))
    return {"true_positive": matched, "false_positive": len(predictions) - matched,
            "false_negative": len(references) - matched}


def read_labels(path, width, height):
    result = []
    for line in path.read_text().splitlines():
        cls, cx, cy, w, h = map(float, line.split())
        result.append({"class_id": int(cls), "box": [(cx - w / 2) * width, (cy - h / 2) * height,
                                                      (cx + w / 2) * width, (cy + h / 2) * height]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("Choose a new output folder")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    sources = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    if not sources or not args.weights.is_file():
        parser.error("Missing weights or evaluation images")
    root = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".venv/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".venv/matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO, settings
    import imageio_ffmpeg
    settings.update({k: False for k in ("sync", "hub") if k in settings})
    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True)
    (args.out / "frames").mkdir()
    totals = {str(t): {k: 0 for k in ("true_positive", "false_positive", "false_negative")} for t in (.3, .5)}
    records = []
    started = time.monotonic()
    results = model.predict(source=[str(p) for p in sources], stream=True, imgsz=args.imgsz,
                            conf=args.confidence, device=args.device, verbose=False, save=False)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20) if Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf").exists() else ImageFont.load_default()
    for i, (source, result) in enumerate(zip(sources, results)):
        if Path(result.path).name != source.name:
            raise RuntimeError("Prediction order disagrees with source images")
        predictions = [{"box": box, "class_id": int(cls), "confidence": float(conf)}
                       for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(),
                                                 result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]
        with Image.open(source) as source_image:
            image = source_image.convert("RGB")
        references = read_labels(args.dataset / "labels" / args.split / (source.stem + ".txt"), *image.size)
        counts = {}
        for threshold in (.3, .5):
            key = str(threshold)
            counts[key] = count_matches(predictions, references, threshold)
            for name, count in counts[key].items():
                totals[key][name] += count
        draw = ImageDraw.Draw(image)
        for ref in references:
            draw.rectangle(ref["box"], outline="#40ff60", width=3)
        for prediction in predictions:
            draw.rectangle(prediction["box"], outline="#ff9a20", width=4)
            x, y = prediction["box"][:2]
            draw.text((x, max(32, y - 24)), f"robot {prediction['confidence']:.2f}", font=font,
                      fill="#ff9a20", stroke_width=1, stroke_fill="black")
        draw.rectangle((0, 0, image.width, 30), fill="black")
        draw.text((8, 3), f"HELD-OUT {args.split}: {source.name} | orange: predictions | green: answer key", font=font, fill="white")
        image.save(args.out / "frames" / f"r_{i:04d}.jpg", quality=94)
        records.append({"frame": source.name, "predictions": predictions, "references": references, "counts": counts})
        print(f"Evaluated {i + 1}/{len(sources)}: {source.name}", flush=True)
    elapsed = time.monotonic() - started
    if len(records) != len(sources):
        raise RuntimeError("Model did not return a prediction for every evaluation image")
    for counts in totals.values():
        tp, fp, fn = (counts[k] for k in ("true_positive", "false_positive", "false_negative"))
        counts.update(precision=tp / (tp + fp) if tp + fp else 0, recall=tp / (tp + fn) if tp + fn else 0)
    report = {"weights": str(args.weights.resolve()), "dataset": str(args.dataset.resolve()), "split": args.split,
              "images": len(records), "confidence_threshold": args.confidence, "imgsz": args.imgsz,
              "metrics": totals, "evaluation_wall_seconds": elapsed,
              "coordinates": "pixels in cropped dataset images", "classes": manifest["classes"],
              "limitation": "Same-match pilot, no team identity or independent venue evaluation.", "frames": records}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    selected = [round(i * (len(records) - 1) / 5) for i in range(6)]
    sheet = Image.new("RGB", (1280, 3 * 230), "#202020")
    for i, index in enumerate(selected):
        with Image.open(args.out / "frames" / f"r_{index:04d}.jpg") as image:
            thumb = ImageOps.contain(image, (640, 230))
            sheet.paste(thumb, (i % 2 * 640, i // 2 * 230))
    sheet.save(args.out / "preview.jpg", quality=95)
    # Sequential frames here follow the pilot's consecutive 5 fps test block.
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-framerate", "5", "-i",
                    str(args.out / "frames/r_%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(args.out / "predictions.mp4")], check=True)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
