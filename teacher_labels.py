"""Budgeted Haiku 5.5 labels; tracking names aid continuity, not student classes."""
import base64
import copy
import fcntl
import hashlib
import io
import inspect
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw

MODEL = "claude-haiku-5-5"
INPUT_RATE, OUTPUT_RATE = .10, .50  # USD / million; prompts below 100K tokens
MAX_OUTPUT = 4096
RESERVATION = .01  # conservative per-attempt reserve, retained on unknown outcomes
SYSTEM = """Label competition robots in an FRC REBUILT broadcast field image for object-detector training.
Return only robots actually visible in this image, including partially occluded robots. Never invent boxes for fully hidden robots.
There are at most six competition robots. Do not force six detections when fewer are visible.
Boxes must tightly enclose each physical robot including its colored bumpers, but exclude flying fuel, shooting streams, field structures, and people.
Do not label hubs (large fixed boxes with AprilTags), ramps, trench arms, ladders, scoreboards, carts, or audience objects as robots.
Use the CURRENT IMAGE pixel coordinates, [left, top, right, bottom]. A faint 100-pixel grid is provided for coordinates.
If this is a replay, close-up, split-screen inset, title card, or not a usable wide field view, set usable=false and return no robots.
Only mark usable=true if the main field view is shown. Do not infer team numbers.
Systematically check all three red-bumper robots and all three blue-bumper robots. Inspect both near foreground corners and the far side of both hubs.
Do not omit a partly visible robot because a person blocks part of its chassis, or because its bumper digits cannot be read.
Return compact JSON; notes should briefly identify ambiguity, if any."""
SCHEMA = {"type": "object", "properties": {
    "usable": {"type": "boolean"},
    "robots": {"type": "array", "items": {"type": "object", "properties": {
        "box": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"}}, "required": ["box", "confidence"], "additionalProperties": False}},
    "notes": {"type": "string"}}, "required": ["usable", "robots", "notes"], "additionalProperties": False}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def load_key():
    for line in Path(".env").read_text().splitlines():
        if line.startswith("CLAUDE_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key.startswith("sk-ant-"):
                return key
    raise ValueError("No configured CLAUDE_API_KEY in .env")


def validate_labels(data, width, height):
    if type(data.get("usable")) is not bool or not isinstance(data.get("robots"), list):
        raise ValueError("Malformed teacher response")
    if len(data["robots"]) > 6 or (not data["usable"] and data["robots"]):
        raise ValueError("Invalid scene or robot count")
    for robot in data["robots"]:
        box = robot.get("box")
        confidence = robot.get("confidence")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError("Invalid box coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Teacher box is outside the image or has zero area")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
    return data


class Teacher:
    def __init__(self, directory, cap=4.0):
        if not 0 < cap <= 4:
            raise ValueError("This approved run has a maximum $4 cap")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "ledger.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.ledger_path = self.directory / "ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {
            "model": MODEL, "cap_usd": cap, "spent_usd": 0.0, "reserved_usd": 0.0, "attempts": [],
            "rates_per_million": {"input": INPUT_RATE, "output": OUTPUT_RATE}}
        if self.ledger["model"] != MODEL or self.ledger["cap_usd"] != cap:
            raise ValueError("Existing ledger model/budget differs; do not reset it")
        import anthropic
        self.client = anthropic.Anthropic(api_key=load_key(), base_url="https://api.anthropic.com", max_retries=0, timeout=120)
        # A checked example demonstrates complete annotation, including partial occlusions.
        self.example_image = Path("track/clip_frames/f_0000.jpg")
        example = json.loads(Path("track/final/f_0000.json").read_text())
        self.example_labels = {"usable": True, "robots": [
            {"box": [round(r["box"][0] * .8), round((r["box"][1] - 110) * .8),
                     round(r["box"][2] * .8), round((r["box"][3] - 110) * .8)], "confidence": 1.0}
            for r in example["robots"] if r["visible"]], "notes": "All six robots are visible, including partially occluded ones."}
        self.prompt_hash = hashlib.sha256((SYSTEM + json.dumps(SCHEMA) + json.dumps(self.example_labels) + inspect.getsource(self.label)).encode() + self.example_image.read_bytes()).hexdigest()

    def label(self, image_path, output, teams=None, previous=None, prior=None, gap=.2):
        image_path, output = Path(image_path), Path(output)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        prompt_hash = hashlib.sha256((self.prompt_hash + json.dumps(teams) + str(gap)).encode()).hexdigest()
        if output.exists():
            data = json.loads(output.read_text())
            if data["image_sha256"] != digest or data["teacher_model"] != MODEL or data.get("prompt_sha256") != prompt_hash:
                raise ValueError("Cached label does not match its source/model")
            return data
        # Prepare a single bounded image. Source labels stay in unscaled crop pixels.
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = min(1.0, 1536 / width, 864 / height)
        sent = image.resize((round(width * scale), round(height * scale)))
        drawing = ImageDraw.Draw(sent, "RGBA")
        for x in range(100, sent.width, 100):
            drawing.line([(x, 0), (x, sent.height)], fill=(255, 255, 255, 55))
            drawing.text((x + 2, 2), str(x), fill=(255, 255, 0, 255))
        for y in range(100, sent.height, 100):
            drawing.line([(0, y), (sent.width, y)], fill=(255, 255, 255, 55))
            drawing.text((2, y + 2), str(y), fill=(255, 255, 0, 255))
        buffer = io.BytesIO()
        sent.save(buffer, "JPEG", quality=90)
        with Image.open(self.example_image) as example_source:
            example_image = example_source.convert("RGB").crop((0, 110, 1920, 720)).resize((1536, 488))
        example_buffer = io.BytesIO()
        example_image.save(example_buffer, "JPEG", quality=90)
        system, schema = SYSTEM, SCHEMA
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Checked annotation example, 1536x488 pixels:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(example_buffer.getvalue()).decode()}}]},
            {"role": "assistant", "content": json.dumps(self.example_labels)}]
        content = []
        if teams is not None:
            from track_llm import SCHEMA as TRACK_SCHEMA
            schema = copy.deepcopy(TRACK_SCHEMA)
            schema["properties"]["robots"]["items"]["properties"]["team"]["enum"] = list(teams)
            system = f"""Track the six FRC REBUILT competition robots in a cropped wide field image.
Teams and bumper colors: {json.dumps(teams)}. Blue alliance end is left, red alliance end is right.
Return exactly one entry per team. Keep identity on the same physical robot unless a clear bumper read proves otherwise.
Each frame is {gap} seconds after the previous frame. Use the previous image and state to follow robots through motion and partial occlusion.
Boxes are [left, top, right, bottom] in the CURRENT SENT IMAGE pixels, tight around the physical chassis including bumpers.
Do not include shooting streams or loose fuel in boxes. Hubs, ramps, trench arms, ladders, people, and monitors are NOT robots.
Search all three red robots and all three blue robots, including near foreground corners and both far hub sides.
If a robot is fully hidden, set visible=false. Do not set visible=false just because its number is unreadable or part of the robot is blocked.
Set named_by=bumper_number only if digits are actually read in this frame. Otherwise use previous_frames, elimination, or guess.
No extra commentary. If the camera is a close-up or a replay, set every robot visible=false and explain briefly in notes."""
            messages = []
            if previous is not None:
                with Image.open(previous) as previous_source:
                    previous_image = previous_source.convert("RGB").resize(sent.size)
                previous_buffer = io.BytesIO()
                previous_image.save(previous_buffer, "JPEG", quality=90)
                content += [{"type": "text", "text": "Previous frame:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(previous_buffer.getvalue()).decode()}}]
            if prior:
                state = [{**r, "box": [round(r["box"][0] * sent.width / width), round(r["box"][1] * sent.height / height),
                         round(r["box"][2] * sent.width / width), round(r["box"][3] * sent.height / height)]} for r in prior]
                content.append({"type": "text", "text": "Previous state (sent-image pixels): " + json.dumps(state)})
        content += [{"type": "text", "text": "Current frame:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode()}},
            {"type": "text", "text": f"Current image size: {sent.width}x{sent.height}. Return the robots for the CURRENT frame."}]
        messages.append({"role": "user", "content": content})
        if self.ledger["spent_usd"] + self.ledger["reserved_usd"] + RESERVATION > self.ledger["cap_usd"]:
            raise RuntimeError("Budget exhausted; no request sent")
        attempt = {"image": str(image_path), "image_sha256": digest, "status": "pending", "reserved_usd": RESERVATION}
        self.ledger["attempts"].append(attempt)
        self.ledger["reserved_usd"] += RESERVATION
        atomic_json(self.ledger_path, self.ledger)  # Reserve before sending; never discard on timeout/restart.
        try:
            response = self.client.messages.create(model=MODEL, max_tokens=MAX_OUTPUT, system=system,
                output_config={"format": {"type": "json_schema", "schema": schema}}, messages=messages)
        except Exception as error:
            # Do not print server error bodies, headers, or the client/key.
            body = getattr(error, "body", None)
            detail = body.get("error", body) if isinstance(body, dict) else {}
            message = str(detail.get("message", "No structured error detail")) if isinstance(detail, dict) else "No structured error detail"
            message = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "[redacted]", message.replace(load_key(), "[redacted]"))[:500]
            attempt.update(status="failed_or_uncertain", error_type=type(error).__name__, http_status=getattr(error, "status_code", None))
            attempt["error_detail"] = message
            atomic_json(self.ledger_path, self.ledger)
            raise RuntimeError(f"Claude request stopped: {attempt['error_type']} (HTTP {attempt['http_status']}): {message}; no automatic retry") from None
        usage = response.usage.model_dump()
        # No caching is requested; charge unexpected cache fields conservatively as input writes.
        cost = (usage["input_tokens"] * INPUT_RATE + usage["output_tokens"] * OUTPUT_RATE +
                (usage.get("cache_creation_input_tokens") or 0) * INPUT_RATE * 2 +
                (usage.get("cache_read_input_tokens") or 0) * INPUT_RATE) / 1e6
        self.ledger["reserved_usd"] = max(0.0, self.ledger["reserved_usd"] - RESERVATION)
        self.ledger["spent_usd"] += cost
        attempt.update(status="billed", usage=usage, cost_usd=cost, response_id=response.id)
        atomic_json(self.ledger_path, self.ledger)
        if cost > RESERVATION or not response.model.startswith(MODEL):
            raise RuntimeError("Unexpected billing/model response; stopped before further requests")
        if response.stop_reason != "end_turn":
            raise ValueError(f"Teacher did not finish normally: {response.stop_reason}")
        data = json.loads(next(block.text for block in response.content if block.type == "text"))
        raw = output.parent / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        atomic_json(raw / output.name, data)
        if teams is not None:
            if len(data["robots"]) != 6 or {r.get("team") for r in data["robots"]} != set(teams):
                raise ValueError("Tracker teacher did not return each team exactly once")
            if any(type(r.get("visible")) is not bool for r in data["robots"]):
                raise ValueError("Tracker teacher returned invalid visibility")
            data["usable"] = any(r["visible"] for r in data["robots"])
            validate_labels({"usable": True, "robots": [r for r in data["robots"] if r["visible"]]}, sent.width, sent.height)
        else:
            validate_labels(data, sent.width, sent.height)
        for robot in data["robots"]:
            # Hidden estimates are not training targets. Some responses omit their box.
            box = robot.get("box")
            if teams is not None and not robot["visible"] and (
                    not isinstance(box, list) or len(box) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
                previous_robot = next((r for r in (prior or []) if r.get("team") == robot["team"]), None)
                if previous_robot:
                    b = previous_robot["box"]
                    robot["box"] = [b[0] * sent.width / width, b[1] * sent.height / height,
                                    b[2] * sent.width / width, b[3] * sent.height / height]
                else:
                    robot["box"] = [0, 0, 1, 1]
            robot["box"] = [round(robot["box"][0] * width / sent.width), round(robot["box"][1] * height / sent.height),
                            round(robot["box"][2] * width / sent.width), round(robot["box"][3] * height / sent.height)]
        data.update(image=image_path.name, image_sha256=digest, teacher_model=MODEL, response_model=response.model,
                    image_size=[width, height], usage=usage, cost_usd=cost, prompt_sha256=prompt_hash)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, data)
        return data
