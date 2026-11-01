"""Draw large, bold robot labels on each frame and join the frames into a 5 fps video.

Usage: .venv/bin/python render_labels.py track/frames track/final out.mp4
Reads one JSON per frame (robots with team, box, visible, confidence) from the labels folder.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEAMS = {"868": "blue", "2910": "blue", "2046": "blue", "4065": "red", "4414": "red", "1323": "red"}
COLORS = {"blue": (30, 110, 255), "red": (235, 30, 30)}
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Black.ttf", 64)
SMALL = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
CROP = (0, 110, 1920, 720)


def draw(frame_path, labels, out_path):
    im = Image.open(frame_path).convert("RGB")
    d = ImageDraw.Draw(im)
    placed = []     # label rectangles already drawn, so big labels do not cover each other
    for r in labels["robots"]:
        color = COLORS[TEAMS[r["team"]]]
        x1, y1, x2, y2 = r["box"]
        if r["visible"]:
            d.rectangle([x1, y1, x2, y2], outline=color, width=8)
        else:   # hidden: dashed outline at the estimated place
            for x in range(x1, x2, 24):
                d.line([(x, y1), (min(x + 12, x2), y1)], fill=color, width=6)
                d.line([(x, y2), (min(x + 12, x2), y2)], fill=color, width=6)
        text = r["team"] + ("" if r["visible"] else " (hidden)")
        if r.get("confidence", 1) < 0.5:
            text += " ?"
        font = FONT if r["visible"] else SMALL
        tw, th = d.textbbox((0, 0), text, font=font)[2:]
        tx = min(max(x1, CROP[0] + 4), CROP[2] - tw - 12)
        # Try above the box, then below it, then inside its top edge.
        for ty in (max(y1 - th - 18, CROP[1] + 4), min(y2 + 6, CROP[3] - th - 14), y1 + 6):
            rect = (tx - 8, ty - 4, tx + tw + 8, ty + th + 12)
            if not any(rect[0] < p[2] and p[0] < rect[2] and rect[1] < p[3] and p[1] < rect[3] for p in placed):
                break
        placed.append(rect)
        d.rectangle(rect, fill=color)
        d.text((tx, ty), text, font=font, fill="white", stroke_width=3, stroke_fill="black")
    clock = f"t = {labels['time_s']:.1f} s"
    d.text((20, CROP[1] + 10), clock, font=SMALL, fill="white", stroke_width=4, stroke_fill="black")
    im.crop(CROP).save(out_path, quality=92)


def main():
    frames_dir, labels_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for f in sorted(frames_dir.glob("f_*.jpg")):
            lab = labels_dir / (f.stem + ".json")
            if lab.exists():
                draw(f, json.load(open(lab)), Path(tmp) / f"r_{n:04d}.jpg")
                n += 1
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "5", "-i", f"{tmp}/r_%04d.jpg",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1920:-2", out], check=True)
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
