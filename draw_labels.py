"""Draw field areas and robot boxes from a labels JSON onto its video frame."""
import json, sys
from PIL import Image, ImageDraw

labels = json.load(open(sys.argv[1]))
frame, out = sys.argv[2], sys.argv[3]
im = Image.open(frame).convert("RGBA")
layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
d = ImageDraw.Draw(layer)
colors = {"blue": (40, 120, 255), "red": (255, 50, 50), "neutral": (255, 255, 255),
          "bump": (255, 200, 0), "trench": (0, 230, 160), "hub": (200, 0, 255)}
for name, poly in labels["areas"].items():
    kind = next((k for k in ("bump", "trench", "hub") if k in name), name.split("_")[0])
    c = colors[kind]
    pts = [tuple(p) for p in poly]
    d.polygon(pts, fill=c + (45,), outline=c + (255,))
    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    d.text((cx - 30, cy), name, fill=(255, 255, 255, 255))
for r in labels["robots"]:
    c = colors[r["alliance"]]
    d.rectangle(r["box"], outline=c + (255,), width=4)
    d.text((r["box"][0], r["box"][1] - 14), f'{r["team"]} ({r["confidence"]})', fill=(255, 255, 0, 255))
Image.alpha_composite(im, layer).convert("RGB").crop((0, 110, 1920, 720)).save(out, quality=92)
