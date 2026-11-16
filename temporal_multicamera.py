"""Temporal multi-camera teaching: Haiku routinely, capped Opus help on hard frames.

Preparation is offline. Paid work requires --execute; one shared, locked ledger
enforces the new user-approved $4.90 ceiling across both models and all cameras.
"""
import argparse
import base64
import copy
import fcntl
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps

from evaluate_detector import iou
from teacher_labels import atomic_json, load_key, normalize_labels, InvalidTeacherLabel
from track_llm import SCHEMA as TRACK_SCHEMA
from train_detector import heartbeat

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
RATES = {HAIKU: (.10, .50), OPUS: (4., 20.)}
RESERVES = {HAIKU: .01, OPUS: .15}
SYSTEM = """Track competition robots in a single camera crop from a 2026 FRC REBUILT broadcast.
You receive previous and current images of THE SAME CAMERA, its roster, and previous state.
Maintain identity on the same physical robot using motion, chassis appearance and bumper reads.
This may be a side/end camera: fewer than six robots can be visible. Never force visibility.
Return one record per roster team, visible=false for fully hidden/off-camera robots. Hidden boxes
are estimates only. Unreadable numbers do not imply invisibility. Inspect partial robots at edges.
Boxes [left,top,right,bottom] use CURRENT image pixels and tightly include chassis and bumpers,
excluding flying fuel, shooting streams, people, hubs, ramps, AprilTags, ladders and field walls.
named_by=bumper_number ONLY when you actually read that team's digits IN THE CURRENT image.
Use previous_frames, elimination or guess otherwise. confidence measures IDENTITY certainty;
box_confidence separately measures geometry certainty. Keep guesses low confidence.
Set usable=false for a title/replay/transition/camera cut or a view you cannot reliably annotate.
Set complete=false if any visible physical robot was missed, duplicated, or cannot be boxed.
Do not assume blue is on the left: orientation differs by camera. Notes must be brief.
For an expert review, correct the proposed labels using the surrounding images; do not assume
the proposal is true. Return CURRENT-frame records only, never boxes from another image."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_for(teams):
    schema = copy.deepcopy(TRACK_SCHEMA)
    robot = schema["properties"]["robots"]["items"]
    robot["properties"]["team"]["enum"] = list(teams)
    robot["properties"]["box_confidence"] = {"type": "number"}
    robot["required"].append("box_confidence")
    for key in ("usable", "complete"):
        schema["properties"][key] = {"type": "boolean"}
        schema["required"].append(key)
    return schema


def image_block(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((1536, 864))
    draw = ImageDraw.Draw(image, "RGBA")
    for x in range(100, image.width, 100):
        draw.line([(x, 0), (x, image.height)], fill=(255, 255, 255, 45))
        draw.text((x + 2, 2), str(x), fill="yellow")
    for y in range(100, image.height, 100):
        draw.line([(0, y), (image.width, y)], fill=(255, 255, 255, 45))
        draw.text((2, y + 2), str(y), fill="yellow")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(buffer.getvalue()).decode()}}, image.size


class Budget:
    def __init__(self, root, contract, client=None):
        self.path = root / "ledger.json"
        self.lock = (root / "job.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "contract": contract, "cap_usd": 4.90, "opus_cap_usd": 1.50,
            "spent_usd": 0., "reserved_usd": 0., "attempts": [], "rates_per_million": {k:list(v) for k,v in RATES.items()}}
        if (self.data["contract"] != contract or self.data["cap_usd"] != 4.90 or
                self.data["opus_cap_usd"] != 1.50 or self.data["rates_per_million"] != {k:list(v) for k,v in RATES.items()}):
            raise ValueError("Existing budget/contract changed; never reset a ledger")
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=load_key(), base_url="https://api.anthropic.com", max_retries=0, timeout=90)
        self.client = client
        atomic_json(self.path, self.data)

    def request(self, model, content, schema, target, fingerprint):
        if model not in RATES:
            raise ValueError("Unauthorized model")
        if target.exists():
            result = json.loads(target.read_text())
            if result["fingerprint"] != fingerprint:
                raise ValueError("Cached request context changed")
            return result
        with self.mutex:
            if self.stop.is_set():
                raise RuntimeError("Another request failed; no further calls")
            prior = [a for a in self.data["attempts"] if a["fingerprint"] == fingerprint]
            if prior:
                raise RuntimeError("Previously attempted request has no cached response; no paid retry")
            reserve = RESERVES[model]
            committed = self.data["spent_usd"] + self.data["reserved_usd"]
            opus = sum(a.get("cost_usd", 0) if a["status"] == "billed" else a["reserve_usd"]
                       for a in self.data["attempts"] if a["model"] == OPUS)
            if committed + reserve > self.data["cap_usd"] + 1e-9 or (model == OPUS and opus + reserve > 1.50 + 1e-9):
                return None
            # Preserve $1.50 for expert checks even if Haiku's cost grows.
            haiku = sum(a.get("cost_usd", 0) if a["status"] == "billed" else a["reserve_usd"]
                        for a in self.data["attempts"] if a["model"] == HAIKU)
            if model == HAIKU and haiku + reserve > 3.40 + 1e-9:
                return None
            attempt = {"fingerprint": fingerprint, "model": model, "target": str(target),
                       "status": "pending", "reserve_usd": reserve}
            self.data["attempts"].append(attempt)
            self.data["reserved_usd"] += reserve
            atomic_json(self.path, self.data)
        try:
            response = self.client.messages.create(model=model, max_tokens=4096, system=SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}])
        except Exception as error:
            with self.mutex:
                attempt.update(status="failed_or_uncertain", error_type=type(error).__name__,
                               http_status=getattr(error, "status_code", None))
                atomic_json(self.path, self.data)
                self.stop.set()
            raise RuntimeError(f"Teacher API stopped: {type(error).__name__}; reservation retained; no automatic retries") from None
        usage = response.usage.model_dump()
        rate_in, rate_out = RATES[model]
        cost = (usage["input_tokens"] * rate_in + usage["output_tokens"] * rate_out +
                (usage.get("cache_creation_input_tokens") or 0) * rate_in * 2 +
                (usage.get("cache_read_input_tokens") or 0) * rate_in) / 1e6
        with self.mutex:
            self.data["spent_usd"] += cost
            self.data["reserved_usd"] = max(0., self.data["reserved_usd"] - reserve)
            attempt.update(status="billed", usage=usage, cost_usd=cost, response_model=response.model,
                           response_id=response.id)
            atomic_json(self.path, self.data)
        text = "".join(b.text for b in response.content if b.type == "text")
        result = {"fingerprint": fingerprint, "model": response.model, "usage": usage,
                  "cost_usd": cost, "stop_reason": response.stop_reason, "text": text}
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(target, result)
        if cost > reserve or not response.model.startswith(model) or usage["input_tokens"] > 100000:
            self.stop.set()
            raise RuntimeError("Unexpected model/billing; no further requests")
        return result


def prepare(plan, root):
    contract = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "frames.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["contract"] != contract:
            raise ValueError("Plan changed; choose a new root")
        for source in saved["sources"]:
            if sha(source["video"]) != source["sha256"]:
                raise ValueError("Source video changed")
        for r in saved["frames"]:
            if sha(root / r["image"]) != r["image_sha256"]:
                raise ValueError("Extracted image changed")
        return saved
    matches = [s["match"] for s in plan["sources"]]
    if len(matches) != len(set(matches)) or "2026cmptx_f1m1" in matches:
        raise ValueError("Each match has one split; Einstein is reserved for checks")
    records, sources = [], []
    for source in plan["sources"]:
        video = Path(source["video"])
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Missing video: {video}")
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (1920, 1080):
            raise ValueError("Camera layout requires reviewed 1920x1080 source")
        sources.append({"video": str(video), "sha256": sha(video)})
        for i in range(round(plan["duration_s"] * plan["fps"])):
            t = round(plan["start_s"] + i / plan["fps"], 3)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Missing frame at {t}")
            for camera, box in source["views"].items():
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
                    raise ValueError("Invalid camera crop")
                name = f"frames/{source['match']}/{camera}/f_{i:04d}.jpg"
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise ValueError("Partial extraction exists; inspect before restarting")
                if not cv2.imwrite(str(path), frame[y1:y2, x1:x2]):
                    raise ValueError("Could not save camera crop")
                records.append({"image": name, "image_sha256": sha(path), "match": source["match"],
                                "camera": camera, "split": source["split"], "index": i,
                                "video_s": t, "crop": box, "teams": source["teams"]})
        cap.release()
        print(f"Prepared {source['match']}: {i+1} synchronized times, {len(source['views'])} cameras", flush=True)
    saved = {"contract": contract, "sources": sources, "frames": records}
    atomic_json(manifest_path, saved)
    atomic_json(root / "plan.json", plan)
    return saved


def query(budget, root, record, previous, prior, model, proposal=None, following=None):
    current = root / record["image"]
    block, sent_size = image_block(current)
    with Image.open(current) as image:
        width, height = image.size
    sx, sy = sent_size[0] / width, sent_size[1] / height
    content = [{"type": "text", "text": f"Camera={record['camera']}, roster/colors={json.dumps(record['teams'])}. Images are {sent_size[0]}x{sent_size[1]}."}]
    if previous:
        prev_block, prev_size = image_block(root / previous["image"])
        if prev_size != sent_size:
            raise ValueError("Camera dimensions changed")
        content += [{"type": "text", "text": f"PREVIOUS frame t={previous['video_s']}s"}, prev_block]
    def scaled(rows):
        return [dict(r, box=[round(r["box"][0]*sx), round(r["box"][1]*sy),
                            round(r["box"][2]*sx), round(r["box"][3]*sy)]) for r in rows]
    if prior:
        content.append({"type": "text", "text": "Previous state in sent-image pixels: " + json.dumps(scaled(prior))})
    content += [{"type": "text", "text": f"CURRENT frame t={record['video_s']}s"}, block]
    if proposal:
        content.append({"type": "text", "text": "Expert review: correct this uncertain CURRENT proposal: " + json.dumps(scaled(proposal))})
    if following:
        content += [{"type": "text", "text": f"FOLLOWING frame t={following['video_s']}s (context only)"}, image_block(root / following["image"])[0]]
    if record["index"] >= 10:
        content.append({"type":"text", "text":"Re-evaluate box geometry independently in this frame; do not inherit previous box confidence. An unreadable team number can have low ID confidence AND high box confidence. Do not mark a visible robot hidden because its team number is uncertain. Do not count robots pictured on monitors as physical robots."})
        if record["camera"] != "main":
            main = current.parent.parent / "main" / current.name
            content += [{"type":"text", "text":"SYNCHRONIZED MAIN VIEW: identity/visibility context only. Its coordinate system differs. Return boxes ONLY in the CURRENT SIDE CAMERA pixels above."}, image_block(main)[0]]
    schema = schema_for(record["teams"])
    fingerprint = hashlib.sha256((SYSTEM + json.dumps(schema, sort_keys=True) + json.dumps(content, sort_keys=True) + model).encode()).hexdigest()
    target = root / "responses" / model / record["match"] / record["camera"] / (current.stem + ".json")
    response = budget.request(model, content, schema, target, fingerprint)
    if response is None:
        return None
    if response["stop_reason"] != "end_turn":
        raise InvalidTeacherLabel("Incomplete response")
    try:
        data = json.loads(response["text"])
        usable, complete = data["usable"], data["complete"]
        if type(usable) is not bool or type(complete) is not bool:
            raise ValueError("Invalid scene flags")
        data = normalize_labels(data, record["teams"], prior, *sent_size, width, height)
        if not usable and any(r["visible"] for r in data["robots"]):
            raise ValueError("Unusable image has visible robots")
        for r in data["robots"]:
            for key in ("confidence", "box_confidence"):
                if type(r.get(key)) not in (int, float) or not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
                    raise ValueError("Invalid confidence")
            if r["named_by"] == "bumper_number" and r["bumper_text"].strip() != r["team"]:
                r["named_by"], r["confidence"] = "guess", min(.3, r["confidence"])
        data.update(usable=usable, complete=complete, record=record, teacher_model=model,
                    request_fingerprint=fingerprint, image_sha256=record["image_sha256"])
        return data
    except (ValueError, TypeError, KeyError) as error:
        raise InvalidTeacherLabel(type(error).__name__) from None


def hard_frame(data, prior, width):
    visible = [r for r in data["robots"] if r["visible"]]
    reasons = []
    if any(r["confidence"] < .65 for r in visible):
        reasons.append("uncertain_identity")
    if any(r["box_confidence"] < .8 for r in visible) or not data["complete"]:
        reasons.append("uncertain_geometry_or_omission")
    for i, r in enumerate(visible):
        if any(iou(r["box"], other["box"]) > .25 for other in visible[:i]):
            reasons.append("crossing_or_overlap")
        old = next((p for p in (prior or []) if p["team"] == r["team"] and p["visible"]), None)
        if old:
            distance = math.hypot((r["box"][0]+r["box"][2]-old["box"][0]-old["box"][2])/2,
                                  (r["box"][1]+r["box"][3]-old["box"][1]-old["box"][3])/2)
            if distance > max(140, width * .12):
                reasons.append("identity_jump")
    return sorted(set(reasons))


def teach_stream(records, budget, root, pilot):
    if not pilot:
        gate=json.loads((root/'review_gate.json').read_text())
        stream=records[0]['match']+'/'+records[0]['camera']
        if stream in gate.get('excluded_streams',[]):
            print(f'Skipped {stream}: pilot review excluded this stream',flush=True)
            return
    prior, previous, experts = None, None, 0
    invalid=0
    for index, record in enumerate(records[:11] if pilot else records):
        target = root / "labels" / record["match"] / record["camera"] / (Path(record["image"]).stem + ".json")
        if (root / "quarantine" / record["match"] / record["camera"] / target.name).exists():
            continue
        if target.exists():
            data = json.loads(target.read_text())
            if data["image_sha256"] != record["image_sha256"]:
                raise ValueError("Source label image changed")
            experts += bool(data.get("expert_review"))
            prior, previous = data["robots"], record
            continue
        try:
            data = query(budget, root, record, previous, prior, HAIKU)
            if data is None:
                print("Haiku allocation reached; preserving remaining expert budget", flush=True)
                return
            width = record["crop"][2] - record["crop"][0]
            reasons = hard_frame(data, prior, width)
            # Up to five expert frames/stream, at least two seconds between them.
            if reasons and experts < 5 and index % 10 == 0:
                expert = query(budget, root, record, previous, prior, OPUS, data["robots"],
                               records[index+1] if index+1 < len(records) else None)
                if expert is not None:
                    expert["haiku_proposal"] = data
                    expert["expert_review"] = {"trigger": reasons, "status": "proposal_not_ground_truth"}
                    data, experts = expert, experts + 1
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(target, data)
            invalid=0
            prior, previous = data["robots"], record
            print(f"{record['match']} {record['camera']} {index+1}/{len(records)}: visible={sum(r['visible'] for r in prior)}, teacher={data['teacher_model']}, shared spend=${budget.data['spent_usd']:.4f}", flush=True)
        except InvalidTeacherLabel as error:
            invalid+=1
            quarantine = root / "quarantine" / record["match"] / record["camera"]
            quarantine.mkdir(parents=True, exist_ok=True)
            atomic_json(quarantine / target.name, {"record": record, "error": str(error), "no_paid_retry": True})
            print(f"Excluded invalid {record['match']}/{record['camera']}/{index}; no retry", flush=True)
            if invalid>=3:
                budget.stop.set()
                raise RuntimeError('Three consecutive invalid labels; stopped before wasting more credits')


def teach(manifest, root, pilot=False):
    budget = Budget(root, manifest["contract"])
    groups = {}
    for record in manifest["frames"]:
        groups.setdefault((record["match"], record["camera"]), []).append(record)
    try:
        # Bounded concurrency; ledger reservations are serialized across all streams.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(teach_stream, rows, budget, root, pilot) for rows in groups.values()]
            for future in futures:
                future.result()
    finally:
        budget.lock.close()


def accepted_boxes(data):
    if not data["usable"] or not data["complete"]:
        return None
    robots = [r for r in data["robots"] if r["visible"]]
    if not robots or any(r["box_confidence"] < .8 for r in robots):
        return None
    boxes = [r["box"] for r in robots]
    if any(b[2]-b[0] < 8 or b[3]-b[1] < 8 for b in boxes):
        return None
    if any(iou(a, b) > .7 for i, a in enumerate(boxes) for b in boxes[:i]):
        return None
    return boxes


def training_region(data):
    """Keep >=0.7 geometry proposals; never crop through uncertain box estimates.

    A partial-image region is required when the teacher flags incomplete coverage.
    The spot-review gate must separately approve this policy before bulk export.
    """
    if not data['usable']:
        return None
    from prepare_confident_regions import area, clip, overlap
    record=data['record'];width=record['crop'][2]-record['crop'][0];height=record['crop'][3]-record['crop'][1]
    positives=[r['box'] for r in data['robots'] if r['visible'] and r['box_confidence']>=.7]
    uncertain=[r['box'] for r in data['robots'] if r['box'] not in positives]
    if not positives:
        return None
    candidates=[]
    for cw in sorted({width,round(width*.75),round(width*.5),round(width/3),max(128,round(width*.25))},reverse=True):
        if cw>width or (cw==width and not data['complete']):
            continue
        starts={0,(width-cw)//2,width-cw}
        for b in positives:
            starts.update(max(0,min(width-cw,round(x))) for x in (b[0],b[2]-cw,(b[0]+b[2]-cw)/2))
        for x in sorted(starts):
            region=[x,0,x+cw,height]
            if any(overlap(b,region)>0 for b in uncertain):
                continue
            included=[b for b in positives if overlap(b,region)>0]
            if not included or any(overlap(b,region)/area(b)<.5 for b in included):
                continue
            boxes=[clip(b,region) for b in included]
            if any(b[2]-b[0]<8 or b[3]-b[1]<8 for b in boxes):
                continue
            if any(iou(a,b)>.7 for i,a in enumerate(boxes) for b in boxes[:i]):
                continue
            candidates.append({'crop':region,'boxes':boxes})
    return max(candidates,key=lambda r:(len(r['boxes']),r['crop'][2]-r['crop'][0],-r['crop'][0])) if candidates else None


def review_sheet(root):
    groups = {}
    for p in sorted((root / "labels").glob("*/*/*.json")):
        data = json.loads(p.read_text())
        r = data["record"]
        groups.setdefault((r["match"], r["camera"]), []).append(data)
    out = root / "review"
    out.mkdir(exist_ok=True)
    for (match, camera), values in groups.items():
        selected = [values[i] for i in sorted({0, len(values)//2, len(values)-1})]
        sheet = Image.new("RGB", (960, 330 * len(selected)), "#222222")
        for index, data in enumerate(selected):
            with Image.open(root / data["record"]["image"]) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for r in data["robots"]:
                if r["visible"]:
                    draw.rectangle(r["box"], outline="lime", width=3)
                    draw.text(tuple(r["box"][:2]), f"{r['team']} ID={r['confidence']:.2f} box={r['box_confidence']:.2f}", fill="white", stroke_width=1, stroke_fill="black")
            image.thumbnail((960, 300))
            sheet.paste(image, (0, index*330+25))
            ImageDraw.Draw(sheet).text((5,index*330+5), f"{match}/{camera} t={data['record']['video_s']} accepted={accepted_boxes(data) is not None}", fill="white")
        sheet.save(out / f"{match}_{camera}.jpg")


def build_dataset(root):
    reviews = json.loads((root / "review_gate.json").read_text())
    if not reviews.get("passed") or reviews["contract"] != json.loads((root / "frames.json").read_text())["contract"]:
        raise ValueError("A matching pilot review is required before training")
    out = root / "dataset"
    if out.exists():
        raise ValueError("Choose a new dataset; do not overwrite")
    for split in ("train", "val", "test"):
        for folder in ("images", "labels"):
            (out / folder / split).mkdir(parents=True, exist_ok=True)
    records, rejected = [], Counter()
    for path in sorted((root / "labels").glob("*/*/*.json")):
        data = json.loads(path.read_text())
        r = data["record"]
        if r["index"] % 5:  # one training image/second, fixed before evaluating
            continue
        region = training_region(data)
        if region is None or f"{r['match']}/{r['camera']}" in reviews.get("excluded_streams", []):
            rejected["uncertain_or_excluded"] += 1
            continue
        if sha(root / r["image"]) != data["image_sha256"]:
            raise ValueError("Labeled image changed")
        name = f"{r['match']}_{r['camera']}_{Path(r['image']).name}"
        split = r["split"]
        x1,y1,x2,y2=region['crop'];w,h=x2-x1,y2-y1;boxes=region['boxes']
        with Image.open(root/r['image']) as image:
            image.crop(region['crop']).save(out/'images'/split/name,quality=95)
        rows = [f"0 {(b[0]+b[2]-2*x1)/(2*w):.8f} {(b[1]+b[3]-2*y1)/(2*h):.8f} {(b[2]-b[0])/w:.8f} {(b[3]-b[1])/h:.8f}" for b in boxes]
        (out / "labels" / split / (Path(name).stem+".txt")).write_text("\n".join(rows)+"\n")
        records.append({"image":name, "split":split, "camera":r["camera"], "source_match":r["match"],
                        "label_sha256":sha(path), "boxes":len(boxes), "crop":region['crop'], "label_type":"temporal_teacher_confident_region"})
    old = Path("datasets/checked_clip_v1")
    # Preserve old training examples to limit catastrophic forgetting; test stays test.
    for split in ("train", "test"):
        for image in sorted((old / "images" / split).glob("*.jpg")):
            label = old / "labels" / split / (image.stem+".txt")
            for folder, source in (("images",image), ("labels",label)):
                shutil.copyfile(source, out / folder / split / source.name)
            records.append({"image":image.name,"split":split,"camera":"einstein_main", "source_match":"2026cmptx_f1m1",
                            "boxes":len(label.read_text().splitlines()),"label_type":"existing_checked_split"})
    summary = {s:{"images":sum(r["split"]==s for r in records),"boxes":sum(r["boxes"] for r in records if r["split"]==s)} for s in ("train","val","test")}
    if summary["train"]["images"] < 100 or summary["val"]["images"] < 15 or summary["test"]["images"] != 14:
        raise ValueError(f"Insufficient accepted labels: {summary}")
    atomic_json(out / "manifest.json", {"classes":["robot"], "summary":summary,"frames":records,"rejected":dict(rejected),
        "warning":"Teacher labels are not independently verified truth. Detector training does not train team identity or camera fusion."})
    (out / "data.yaml").write_text(f"path: {json.dumps(str(out.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [robot]\n")
    from prepare_dataset import make_preview
    for split in summary:
        make_preview(out,split,[r for r in records if r["split"]==split],["robot"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("configs/temporal_multicamera_v1.json"))
    p.add_argument("--root", type=Path, default=Path("datasets/temporal_multicamera_v1_run"))
    p.add_argument("--out", type=Path, default=Path("runs/temporal_multicamera_v1"))
    p.add_argument("--stage", choices=("prepare","pilot","teach","review","train"), default="prepare")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["cap_usd"] != 4.90 or plan["opus_cap_usd"] != 1.50:
        p.error("This run is authorized only within $4.90 total / $1.50 Opus")
    if args.stage in ("pilot","teach") and not args.execute:
        p.error("Paid requests require --execute")
    args.root.mkdir(parents=True,exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    try:
        manifest = prepare(plan,args.root)
        atomic_json(args.root/"status.json",{"state":"running","stage":args.stage,"updated_unix":time.time()})
        if args.stage == "teach":
            gate = json.loads((args.root/"review_gate.json").read_text())
            if not gate["passed"] or gate["contract"] != manifest["contract"]:
                raise ValueError("Pilot review is required before bulk teaching")
        if args.stage in ("pilot","teach"):
            teach(manifest,args.root,args.stage=="pilot")
            review_sheet(args.root)
        elif args.stage == "review":
            review_sheet(args.root)
        elif args.stage == "train":
            dataset = build_dataset(args.root)
            subprocess.run([sys.executable,"-u","train_detector.py","--data",str(dataset/"data.yaml"),"--out",str(args.out),
                            "--model","runs/local_robot_v1/weights/best.pt","--epochs","30"],check=True)
            subprocess.run([sys.executable,"-u","evaluate_detector.py",str(args.out/"weights/best.pt"),str(dataset),str(args.out)+"_test"],check=True)
            subprocess.run([sys.executable,"-u","evaluate_detector.py",str(args.out/"weights/best.pt"),str(dataset),str(args.out)+"_val","--split","val"],check=True)
        atomic_json(args.root/"status.json",{"state":"complete","stage":args.stage,"updated_unix":time.time()})
    except BaseException as error:
        atomic_json(args.root/"status.json",{"state":"stopped","stage":args.stage,"error_type":type(error).__name__,"updated_unix":time.time()})
        print(f"Stage {args.stage} stopped: {type(error).__name__}. No automatic retries.",flush=True)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
