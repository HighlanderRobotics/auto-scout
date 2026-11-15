"""Make an offline browser tool for humans to label fixed floor landmarks in video."""

import argparse
import html
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent


def sample_times(start_s, duration_s, every_s):
    """Return inclusive-start sample times without exceeding the requested window."""
    if start_s < 0 or duration_s <= 0 or every_s <= 0:
        raise ValueError("start must be >= 0; duration and every must be positive")
    count = int((duration_s - 1e-9) // every_s) + 1
    return [round(start_s + index * every_s, 3) for index in range(count)]


def build_meta(source, width, height, samples):
    return {
        "format": "auto-scout-landmark-labels-v1",
        "source": source.name,
        "image_size": [width, height],
        "instructions": "Use fixed points on the carpet or field boundary, never robot or hub-top positions.",
        "frames": [
            {"id": f"frame_{index:04d}", "time_s": time_s,
             "image": f"frames/frame_{index:04d}.jpg"}
            for index, time_s in enumerate(samples)
        ],
        "landmarks": [
            {"id": "blue_far", "name": "Blue field corner / far side", "field_xy": [0, 0]},
            {"id": "red_far", "name": "Red field corner / far side", "field_xy": [100, 0]},
            {"id": "red_near", "name": "Red field corner / near side", "field_xy": [100, 100]},
            {"id": "blue_near", "name": "Blue field corner / near side", "field_xy": [0, 100]},
            {"id": "center_far", "name": "Center line meets far boundary", "field_xy": [50, 0]},
            {"id": "center_near", "name": "Center line meets near boundary", "field_xy": [50, 100]},
            {"id": "blue_mid", "name": "Blue boundary midpoint", "field_xy": [0, 50]},
            {"id": "red_mid", "name": "Red boundary midpoint", "field_xy": [100, 50]},
        ],
    }


def write_labeler(out, meta):
    template = (ROOT / "landmark_labeler.html").read_text(encoding="utf-8")
    page = template.replace("__META__", html.escape(json.dumps(meta), quote=False))
    (out / "label_landmarks.html").write_text(page, encoding="utf-8")


def extract_frames(source, out, samples):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("Video has no readable dimensions or frame rate")
    frames_dir = out / "frames"
    frames_dir.mkdir()
    for index, time_s in enumerate(samples):
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read video at {time_s:.3f} seconds")
        destination = frames_dir / f"frame_{index:04d}.jpg"
        if not cv2.imwrite(str(destination), frame):
            raise ValueError(f"Could not write {destination}")
    capture.release()
    return width, height


def main():
    parser = argparse.ArgumentParser(description="Create an offline click-to-label landmark tool.")
    parser.add_argument("source", type=Path, help="Local match video; no upload or API call is made")
    parser.add_argument("--out", type=Path, required=True, help="New output folder")
    parser.add_argument("--start", type=float, default=0, help="First sampled video second")
    parser.add_argument("--duration", type=float, default=20, help="Window to sample, in seconds")
    parser.add_argument("--every", type=float, default=2, help="Seconds between sampled frames")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}. Choose a new folder.")
    try:
        samples = sample_times(args.start, args.duration, args.every)
        args.out.mkdir(parents=True)
        width, height = extract_frames(args.source, args.out, samples)
        meta = build_meta(args.source, width, height, samples)
        (args.out / "labeling_plan.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        write_labeler(args.out, meta)
    except Exception:
        if args.out.exists() and not any(args.out.iterdir()):
            args.out.rmdir()
        raise
    print(f"Prepared {len(samples)} frames. Open {args.out / 'label_landmarks.html'} in a browser.")
    print("Click at least four fixed floor points in each useful camera view, then download landmark_labels.json.")


if __name__ == "__main__":
    main()
