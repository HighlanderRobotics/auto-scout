"""Forward pass: ask Claude where each robot is in every frame, given the previous frame and state.

Usage: .venv/bin/python track_llm.py FRAMES_DIR OUT_DIR [--seed labels.json] [--start N] [--end N]
Writes one JSON per frame to the output folder (reruns skip frames already done).
--start/--end run one chunk of frames, so several chunks can run side by side; the backward pass joins them.
Boxes in the output are in full 1920x1080 frame pixels.
"""
import argparse, base64, io, json, sys, time
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw

FPS = 5
T0 = 5.5                      # video time of frame 0
CROP = (0, 110, 1920, 720)    # wide field camera (top part of the broadcast)
SCALE = 0.8                   # sent image is 1536 x 488
TEAMS = {"868": "blue", "2910": "blue", "2046": "blue", "4065": "red", "4414": "red", "1323": "red"}
MODEL = "claude-opus-5-5"

SYSTEM = """You track FRC robots in broadcast video of a 2026 REBUILT match (Einstein Final 1).
The image is the wide field camera. The blue alliance end is at the left, the red end at the right, and the camera looks across the field from behind the near side wall.
Teams: blue alliance (blue bumpers) 868, 2910, 2046. Red alliance (red bumpers) 4065, 4414, 1323. Team numbers are printed on the bumpers.
Things that are NOT robots: the two hubs (tall blue or red boxes with black-and-white AprilTags), bumps (blue or red ramps beside each hub), trenches (low arms near the side walls), people, yellow fuel balls, the ladders at each end, monitors and the score table at the near edge.
Each image has a light pixel grid every 100 px with the coordinates written on it. Give every box in the CURRENT image's pixels, tight around the robot including its bumpers.
Use the previous frame and the previous state: frames are 0.2 s apart, so a robot moves at most about 150 px between frames. Keep each team on the same robot unless you clearly read a different number.
If a robot is hidden, set visible=false and give your best estimate of its box from where it was last seen.
Set named_by to "bumper_number" only when you read its digits in THIS frame."""

ACTIONS_PROMPT = """
Also give each robot's action in THIS frame, using both frames to judge motion:
- shooting: fuel is leaving the robot toward a hub
- feeding: the robot launches or pushes fuel toward its own alliance end, not into a hub
- intaking: the robot drives through loose fuel and collects it
- defending: the robot blocks, pins or pushes an opposing robot
- climbing: the robot is on the ladder at its alliance end
- driving: moving, none of the above
- stationary: not moving, none of the above
- unknown: hidden or not clear"""
ACTION_FIELDS = {
    "action": {"type": "string", "enum": ["shooting", "feeding", "intaking", "defending", "climbing", "driving", "stationary", "unknown"]},
    "defending_team": {"type": "string"},
}

SCHEMA = {
    "type": "object",
    "properties": {
        "robots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "enum": list(TEAMS)},
                    "visible": {"type": "boolean"},
                    "box": {"type": "array", "items": {"type": "integer"}},
                    "named_by": {"type": "string", "enum": ["bumper_number", "previous_frames", "elimination", "guess"]},
                    "bumper_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["team", "visible", "box", "named_by", "bumper_text", "confidence"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["robots", "notes"],
    "additionalProperties": False,
}


def api_key():
    for line in Path(".env").read_text().splitlines():
        if line.startswith("CLAUDE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("CLAUDE_API_KEY not found in .env")


def to_sent(box):   # full-frame pixels -> sent-image pixels
    x1, y1, x2, y2 = box
    return [round((x1 - CROP[0]) * SCALE), round((y1 - CROP[1]) * SCALE), round((x2 - CROP[0]) * SCALE), round((y2 - CROP[1]) * SCALE)]


def to_full(box):   # sent-image pixels -> full-frame pixels
    x1, y1, x2, y2 = box
    return [round(x1 / SCALE + CROP[0]), round(y1 / SCALE + CROP[1]), round(x2 / SCALE + CROP[0]), round(y2 / SCALE + CROP[1])]


def image_block(path):
    im = Image.open(path).convert("RGB").crop(CROP)
    im = im.resize((round(im.width * SCALE), round(im.height * SCALE)), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(100, im.width, 100):
        d.line([(x, 0), (x, im.height)], fill=(255, 255, 255, 70))
        d.text((x + 2, 2), str(x), fill=(255, 255, 0, 255))
    for y in range(100, im.height, 100):
        d.line([(0, y), (im.width, y)], fill=(255, 255, 255, 70))
        d.text((2, y + 2), str(y), fill=(255, 255, 0, 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def state_for_prompt(robots):
    return [{**r, "box": to_sent(r["box"])} for r in robots]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir"); ap.add_argument("out_dir")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--actions", action="store_true", help="also ask for each robot's action")
    ap.add_argument("--robots", help="JSON file of team -> how the robot looks, added to the prompt")
    ap.add_argument("--step", type=int, default=1, help="use every Nth frame (2 = 2.5 frames per second)")
    ap.add_argument("--one-image", action="store_true", help="send only the current frame; the previous positions still go as text")
    ap.add_argument("--seed"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int)
    a = ap.parse_args()
    frames_dir, out_dir = Path(a.frames_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("f_*.jpg"))
    end = min(a.end if a.end is not None else len(frames), len(frames))
    prior = None
    if a.seed and a.start == 0:   # hand labels for the first frame, as a weak hint
        prior = [{"team": r["team"].rstrip("?"), "visible": True, "box": r["box"], "named_by": "guess",
                  "bumper_text": "", "confidence": 0.3} for r in json.load(open(a.seed))["robots"] if r["team"].rstrip("?") in TEAMS]
    system, schema = SYSTEM, json.loads(json.dumps(SCHEMA))
    if a.robots:
        looks = json.load(open(a.robots))
        system += "\nHow each robot looks (use this to keep names right when a bumper is hidden):\n" + "\n".join(f"- {t}: {d}" for t, d in looks.items())
    if a.step > 1:
        gap = a.step / FPS
        system = system.replace("frames are 0.2 s apart, so a robot moves at most about 150 px between frames",
                                f"frames are {gap:.1f} s apart, so a robot moves at most about {round(150 * a.step)} px between frames")
    if a.one_image:
        system += "\nOnly the current frame is sent as an image. Use the previous state's boxes to follow each robot."
    if a.actions:
        system += ACTIONS_PROMPT
        item = schema["properties"]["robots"]["items"]
        item["properties"].update(ACTION_FIELDS)
        item["required"] += list(ACTION_FIELDS)
    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if not a.model.startswith("claude-haiku"):   # Haiku 4.5 does not take an effort setting
        output_config["effort"] = "medium"
    client = anthropic.Anthropic(api_key=api_key())
    started = time.time()
    for i in range(a.start, end, a.step):
        f = frames[i]
        out = out_dir / (f.stem + ".json")
        t = T0 + i / FPS
        if out.exists():
            prior = json.load(open(out))["robots"]
            continue
        content = []
        if i > a.start and not a.one_image:
            content += [{"type": "text", "text": f"Previous frame (t={t - a.step / FPS:.1f} s):"}, image_block(frames[i - a.step])]
        content += [{"type": "text", "text": f"Current frame (t={t:.1f} s):"}, image_block(f)]
        hint = "Previous state (boxes in image pixels):\n" + json.dumps(state_for_prompt(prior)) if prior else "No previous state. This is the first frame."
        content.append({"type": "text", "text": hint + "\n\nReturn all six robots for the CURRENT frame."})
        response = client.messages.create(
            model=a.model, max_tokens=16000, system=system, output_config=output_config,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise SystemExit(f"refused at frame {i}")
        data = json.loads(next(b.text for b in response.content if b.type == "text"))
        for r in data["robots"]:
            r["box"] = to_full(r["box"])
        data.update(frame=f.name, time_s=round(t, 2), usage={"in": response.usage.input_tokens, "out": response.usage.output_tokens})
        json.dump(data, open(out, "w"), indent=1)
        prior = data["robots"]
        seen = sum(r["visible"] for r in prior)
        read = ",".join(r["team"] for r in prior if r["named_by"] == "bumper_number")
        el = time.time() - started
        print(f"[{a.start}-{end} step {a.step}] frame {i + 1 - a.start}/{end - a.start} t={t:.1f}s visible={seen} read=[{read}] elapsed={el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
