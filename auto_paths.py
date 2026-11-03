"""YouTube/local video -> local robot tracking -> calibrated autonomous path animation."""
import argparse
import base64
import hashlib
import json
import math
import os
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
KNOWN_VIDEO = "EjF9we707DA"
# Normalized schematic landmarks, not a surveyed field drawing.
FIELD_ELEMENTS = [
    {"kind": "hub", "alliance": side, "x": x, "y": 50, "width": 7.2, "height": 15}
    for side, x in (("blue", 33.333333), ("red", 66.666667))
] + [
    {"kind": "ramp", "alliance": side, "x": x, "y": y, "width": 6.8, "height": 21}
    for side, x in (("blue", 33.333333), ("red", 66.666667)) for y in (29, 71)
]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".venv/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def youtube_id(value):
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("https", "http") or host not in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"):
        return None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        candidate = parsed.path.split("/")[2]
    else:
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    return candidate if len(candidate) == 11 and all(c.isalnum() or c in "_-" for c in candidate) else None


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    size = np.asarray(data["image_size"], dtype=float)
    crop = np.asarray(data["crop"], dtype=float)
    image = np.asarray(data["image_points"], dtype=np.float64)
    field = np.asarray(data["field_points"], dtype=np.float64)
    if size.shape != (2,) or crop.shape != (4,) or image.shape != (4, 2) or field.shape != (4, 2):
        raise ValueError("Calibration needs image_size, crop and four matched image_points/field_points")
    if not all(np.isfinite(v).all() for v in (size, crop, image, field)):
        raise ValueError("Calibration contains non-finite coordinates")
    if any(v != round(v) for v in (*size, *crop)):
        raise ValueError("Image dimensions and crop must be whole pixels")
    if not (0 <= crop[0] < crop[2] <= size[0] and 0 <= crop[1] < crop[3] <= size[1]):
        raise ValueError("Calibration crop is outside source dimensions")
    if np.any(image < 0) or np.any(image > size) or np.any(field < 0) or np.any(field > 100):
        raise ValueError("Image points must be inside the image and field points within 0..100")
    for points in (image, field):
        contour = points.astype(np.float32)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1:
            raise ValueError("Calibration points must form a non-degenerate convex quadrilateral in corresponding order")
    matrix = cv2.getPerspectiveTransform(image.astype(np.float32), field.astype(np.float32))
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Degenerate field mapping")
    return data, matrix


def field_position(box, matrix):
    floor = np.array([[(box[0] + box[2]) / 2, box[3]]], dtype=np.float64)
    projected = cv2.perspectiveTransform(floor[None, :, :], matrix)[0, 0]
    if not np.isfinite(projected).all() or not (0 <= projected[0] <= 100 and 0 <= projected[1] <= 100):
        return None
    return [round(float(v), 4) for v in projected]


def connected(a, b, max_gap=.6):
    return 0 < b["t"] - a["t"] <= max_gap and math.hypot(b["x"] - a["x"], b["y"] - a["y"]) <= 25


def color(track):
    import colorsys
    return tuple(round(v * 255) for v in colorsys.hsv_to_rgb((track * .61803398875) % 1, .65, 1))


def heartbeat(stop):
    started = time.monotonic()
    while not stop.wait(25):
        print(f"Path job active: {time.monotonic() - started:.0f}s elapsed", flush=True)


def source_video(source, out):
    local = Path(source)
    if local.is_file():
        return local.resolve()
    identifier = youtube_id(source)
    if identifier is None:
        raise ValueError("Source must be a local video file or a valid YouTube video URL")
    import yt_dlp
    print(f"Downloading YouTube video {identifier} (no Claude calls)", flush=True)
    opts = {"format": "bv*[height<=1080][ext=mp4][vcodec^=avc1]/bv*[height<=1080][ext=mp4]",
            "outtmpl": str(out / "source.%(ext)s"), "noplaylist": True, "retries": 2,
            "quiet": True, "noprogress": True, "no_warnings": False}
    with yt_dlp.YoutubeDL(opts) as downloader:
        info = downloader.extract_info("https://www.youtube.com/watch?v=" + identifier, download=True)
        return Path(downloader.prepare_filename(info)).resolve()


def font(size):
    path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def field_frame(tracks, current, duration, approximate, max_gap=.6, elements=None):
    image = Image.new("RGB", (960, 540), "#0b1220")
    draw = ImageDraw.Draw(image)
    draw.text((30, 12), f"Autonomous paths  {current:04.1f} / {duration:.1f}s", font=font(24), fill="white")
    draw.text((30, 45), "Approximate projection | anonymous track IDs" if approximate else "Calibrated projection | anonymous track IDs", font=font(16), fill="#cbd5e1")
    left, top, right, bottom = 45, 88, 915, 480
    draw.rectangle((left, top, right, bottom), fill="#152139", outline="#94a3b8", width=2)
    for step in range(10, 100, 10):
        x = left + (right-left)*step/100; y = top+(bottom-top)*step/100
        draw.line((x, top, x, bottom), fill="#23334c")
        draw.line((left, y, right, y), fill="#23334c")
    draw.line(((left+right)/2, top, (left+right)/2, bottom), fill="#64748b", width=2)
    for element in FIELD_ELEMENTS if elements is None else elements:
        cx=left+(right-left)*element["x"]/100;cy=top+(bottom-top)*element["y"]/100
        w=(right-left)*element["width"]/100;h=(bottom-top)*element["height"]/100
        edge="#60a5fa" if element["alliance"]=="blue" else "#fb7185"
        fill="#173e68" if element["alliance"]=="blue" else "#632b3b"
        box=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
        draw.rounded_rectangle(box,radius=4,fill=fill,outline=edge,width=2)
        if element["kind"]=="hub":
            # Basket rim and a dark opening, seen from above.
            draw.ellipse((cx-w*.28,cy-h*.3,cx+w*.28,cy+h*.1),fill="#07101d",outline=edge,width=3)
            label="HUB"
            draw.text((cx,cy+h*.27),label,font=font(13),fill="#e2e8f0",anchor="mm")
        else:
            # Two slopes rise toward the center ridge of each bump/ramp.
            draw.line((cx,cy-h*.42,cx,cy+h*.42),fill=edge,width=3)
            for y_offset in (-h*.27,0,h*.27):
                draw.line((cx-w*.35,cy+y_offset-h*.08,cx-w*.08,cy+y_offset,cx-w*.35,cy+y_offset+h*.08),fill=edge,width=2)
                draw.line((cx+w*.35,cy+y_offset-h*.08,cx+w*.08,cy+y_offset,cx+w*.35,cy+y_offset+h*.08),fill=edge,width=2)
            draw.rectangle((cx-w*.4,cy+h*.2,cx+w*.4,cy+h*.46),fill=fill)
            draw.text((cx,cy+h*.34),"RAMP",font=font(10),fill=edge,anchor="mm")
    draw.text((left, bottom+12), "Blue end  |  normalized field coordinates (not meters)", font=font(16), fill="#93c5fd")
    draw.text((right-85, bottom+12), "Red end", font=font(16), fill="#fca5a5")
    def xy(point):
        return left+(right-left)*point["x"]/100, top+(bottom-top)*point["y"]/100
    for track in tracks:
        rgb=tuple(track["color"])
        points = [p for p in track["points"] if p["t"] <= current + 1e-6]
        for a,b in zip(points,points[1:]):
            if connected(a,b,max_gap): draw.line((*xy(a),*xy(b)),fill=rgb,width=3)
        if points and current-points[-1]["t"] <= min(max_gap,.35):
            x,y=xy(points[-1]);draw.ellipse((x-7,y-7,x+7,y+7),fill=rgb,outline="white",width=2)
            draw.text((x+9,y-10),str(track["id"]),font=font(16),fill=rgb)
    return image


def render_animation(data, out):
    """Render saved coordinates without re-downloading or running the detector."""
    import imageio_ffmpeg
    data["field_elements"] = [dict(e) for e in FIELD_ELEMENTS]
    data["field_elements_note"] = "Schematic hub/basket and ramp placement; no fuel rendered. Not surveyed geometry."
    (out/"paths.json").write_text(json.dumps(data,indent=2)+"\n")
    template=(ROOT/"auto_paths_viewer.html").read_text()
    (out/"animation.html").write_text(template.replace("__PATH_DATA__",json.dumps(data).replace("<","\\u003c")))
    writer=imageio_ffmpeg.write_frames(str(out/"paths.mp4"),(960,540),fps=data["fps"],codec="libx264",
                pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2,ffmpeg_log_level="error")
    next(writer)
    gif=[]
    count=math.ceil(data["duration"]*data["fps"])
    try:
        for index in range(count):
            pane=field_frame(data["tracks"],index/data["fps"],data["duration"],
                             data["calibration"].get("approximate",False),data["max_gap_s"],data["field_elements"])
            writer.send(np.asarray(pane))
            if index%max(1,data["fps"]//5)==0:
                gif.append(pane.resize((768,432)).convert("P",palette=Image.Palette.ADAPTIVE,colors=128))
    finally:
        writer.close()
    if gif:
        gif[0].save(out/"paths.gif",save_all=True,append_images=gif[1:],duration=round(data["duration"]*1000/len(gif)),loop=0,optimize=False)
    field_frame(data["tracks"],data["duration"],data["duration"],data["calibration"].get("approximate",False),
                data["max_gap_s"],data["field_elements"]).save(out/"paths_preview.png")


def prepare_calibrator(out, image, auto_start, source):
    import io
    buf=io.BytesIO();image.save(buf,format="JPEG",quality=92)
    meta={"image_size":list(image.size),"auto_start_s":auto_start,"source":source}
    template=(ROOT/"auto_paths_calibrate.html").read_text()
    template=template.replace("__REFERENCE__","data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode())
    template=template.replace("__META__",json.dumps(meta).replace("<","\\u003c"))
    (out/"calibrate.html").write_text(template)


def score_reference(frames, directory):
    from evaluate_detector import count_matches
    reference={round(float(d["time_s"])*1000):d for p in Path(directory).glob("f_*.json") for d in [json.loads(p.read_text())]}
    totals={str(t):Counter() for t in (.3,.5)};n=0
    for frame in frames:
        ref=reference.get(round(frame["video_s"]*1000))
        if ref is None:continue
        n+=1
        predictions=[{"class_id":0,"box":r["box"]} for r in frame["robots"]]
        truth=[{"class_id":0,"box":r["box"]} for r in ref["robots"] if r["visible"]]
        for threshold in (.3,.5):totals[str(threshold)].update(count_matches(predictions,truth,threshold))
    result={"frames_compared":n,"scope":"Anonymous tracked-box agreement only; no field-position ground truth or verified team identity."}
    for key,counts in totals.items():
        tp,fp,fn=(counts[k] for k in ("true_positive","false_positive","false_negative"))
        result[key]={**dict(counts),"precision":tp/(tp+fp) if tp+fp else 0,"recall":tp/(tp+fn) if tp+fn else 0}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",nargs="?",help="YouTube link or local video")
    parser.add_argument("--render-paths",type=Path,help="Re-render a saved paths.json without detection; requires a new --out")
    parser.add_argument("--out",type=Path,default=Path("runs/auto_paths"))
    parser.add_argument("--weights",type=Path,default=Path("runs/local_robot_v1/weights/best.pt"))
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--auto-start",type=float,help="Video seconds where autonomous begins")
    parser.add_argument("--duration",type=float,default=20)
    parser.add_argument("--fps",type=int,default=10)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--confidence",type=float,default=.1,help="Low detection threshold; ByteTrack filters new tracks")
    parser.add_argument("--min-track-frames",type=int,default=3)
    parser.add_argument("--max-gap",type=float,default=.6)
    parser.add_argument("--prepare",action="store_true",help="Make a click-to-calibrate HTML file instead of running inference")
    parser.add_argument("--reference",type=Path,help="Optional checked full-frame labels with matching time_s for scoring")
    args=parser.parse_args()
    if args.out.exists() or args.out.is_symlink():parser.error("Choose a new output folder; existing runs are never overwritten")
    if args.render_paths:
        if args.source:parser.error("Use --render-paths without a source argument")
        data=json.loads(args.render_paths.read_text())
        args.out.mkdir(parents=True)
        for name in ("tracking.mp4","calibration.json","calibration_preview.jpg","bytetrack.yaml"):
            source_asset=args.render_paths.parent/name
            if source_asset.exists():shutil.copyfile(source_asset,args.out/name)
        render_animation(data,args.out)
        print(f"Re-rendered saved paths: {args.out/'animation.html'}; no inference or API calls",flush=True)
        return
    if not args.source:parser.error("Provide a YouTube link/local video, or --render-paths")
    if not 1<=args.fps<=30 or not 0<args.duration<=180 or args.min_track_frames<1 or not 0<args.confidence<1 or args.max_gap<=0:
        parser.error("Invalid frame rate, duration, confidence or track settings")
    if args.calibration is None and youtube_id(args.source)==KNOWN_VIDEO and not args.prepare:
        args.calibration=ROOT/"configs/einstein_final1_camera.json"
    calibration,matrix=(load_calibration(args.calibration) if args.calibration else (None,None))
    start=args.auto_start if args.auto_start is not None else (calibration or {}).get("auto_start_s")
    if start is None or not math.isfinite(start) or start<0:
        parser.error("Pass --auto-start VIDEO_SECONDS. The tool does not automatically identify the autonomous start.")
    if calibration is None and not args.prepare:
        parser.error("This camera needs calibration. Run with --prepare first, then supply its exported camera.json.")
    if not args.prepare and not args.weights.is_file():parser.error("Local detector weights missing; pass --weights PATH to a trained checkpoint")
    args.out.mkdir(parents=True)
    stop=threading.Event();threading.Thread(target=heartbeat,args=(stop,),daemon=True).start()
    capture=None;writers=[]
    try:
        video=source_video(args.source,args.out)
        capture=cv2.VideoCapture(str(video))
        if not capture.isOpened():raise ValueError("Cannot open the source video")
        native_fps=float(capture.get(cv2.CAP_PROP_FPS));native_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if native_fps<=0 or native_frames<=0:raise ValueError("Video has no usable frame timing")
        if args.fps>native_fps+.001:raise ValueError("Requested frame rate exceeds source rate; lower --fps")
        if not args.prepare and start+args.duration>native_frames/native_fps+.05:raise ValueError("Autonomous window extends beyond the video")
        capture.set(cv2.CAP_PROP_POS_MSEC,start*1000);ok,frame=capture.read()
        if not ok:raise ValueError("Cannot read the autonomous start frame")
        height,width=frame.shape[:2]
        if args.prepare:
            prepare_calibrator(args.out,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),start,args.source)
            print(f"Open {args.out/'calibrate.html'}, export camera.json, and rerun with --calibration in a NEW output folder.",flush=True)
            return
        if [width,height]!=calibration["image_size"]:raise ValueError("Calibration resolution differs from video; recalibrate this camera")
        (args.out/"calibration.json").write_text(json.dumps(calibration,indent=2)+"\n")
        preview=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));drawing=ImageDraw.Draw(preview)
        drawing.polygon([tuple(p) for p in calibration["image_points"]],outline="#f472b6",width=5)
        for i,p in enumerate(calibration["image_points"]):drawing.text(tuple(p),str(i+1),font=font(30),fill="#f472b6")
        preview.save(args.out/"calibration_preview.jpg")
        from ultralytics import YOLO,settings
        import imageio_ffmpeg
        settings.update({k:False for k in ("sync","hub") if k in settings})
        import torch
        device = args.device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
        model=YOLO(str(args.weights))
        tracker=args.out/"bytetrack.yaml"
        tracker.write_text(f"tracker_type: bytetrack\ntrack_high_thresh: 0.3\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.4\ntrack_buffer: {args.fps}\nmatch_thresh: 0.8\nfuse_score: true\n")
        x1,y1,x2,y2=calibration["crop"]
        annotated_size=(1280,round((y2-y1)*1280/(x2-x1)/2)*2)
        annotation=imageio_ffmpeg.write_frames(str(args.out/"tracking.mp4"),annotated_size,fps=args.fps,
                    codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2,ffmpeg_log_level="error")
        next(annotation);writers.append(annotation)
        frames=[];tracks={};outside=0;last_native=-1
        count=math.ceil(args.duration*args.fps)
        for index in range(count):
            t=index/args.fps;target=round((start+t)*native_fps)
            if target<=last_native:raise ValueError("Requested sample rate exceeds usable source frame rate")
            if target!=last_native+1:capture.set(cv2.CAP_PROP_POS_FRAMES,target)
            ok,frame=capture.read();last_native=target
            if not ok:raise ValueError(f"Video ended before frame {index}")
            roi=frame[y1:y2,x1:x2].copy()
            result=model.track(roi,persist=True,tracker=str(tracker.resolve()),device=device,imgsz=960,
                               conf=args.confidence,max_det=12,verbose=False)[0]
            robots=[]
            if result.boxes.id is not None:
                for box,identifier,confidence in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.id.int().cpu().tolist(),result.boxes.conf.cpu().tolist()):
                    full=[box[0]+x1,box[1]+y1,box[2]+x1,box[3]+y1]
                    position=field_position(full,matrix)
                    robots.append({"id":identifier,"box":[round(v,2) for v in full],"confidence":round(confidence,4),"field":position})
                    rgb=color(identifier);bgr=rgb[::-1]
                    cv2.rectangle(roi,(round(box[0]),round(box[1])),(round(box[2]),round(box[3])),bgr,3)
                    cv2.putText(roi,f"Track {identifier} {confidence:.2f}",(max(0,round(box[0])),max(25,round(box[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.9,bgr,2)
                    if position is None:outside+=1;continue
                    track=tracks.setdefault(identifier,{"id":identifier,"color":list(rgb),"points":[]})
                    track["points"].append({"t":round(t,4),"x":position[0],"y":position[1],"confidence":round(confidence,4)})
            cv2.putText(roi,f"AUTO +{t:.1f}s | Anonymous IDs, not teams",(20,roi.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            resized=cv2.resize(roi,annotated_size)
            annotation.send(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB))
            frames.append({"t":round(t,4),"video_s":round(start+t,4),"robots":robots})
            if index%args.fps==0 or index+1==count:print(f"Tracking {index+1}/{count} frames; {len(tracks)} IDs observed",flush=True)
        annotation.close();writers.remove(annotation)
        included=[v for _,v in sorted(tracks.items()) if len(v["points"])>=args.min_track_frames]
        data={"title":"Autonomous robot paths","duration":args.duration,"fps":args.fps,"auto_start_s":start,
              "source":args.source,"calibration":calibration,"max_gap_s":args.max_gap,"tracks":included,
              "frames":frames,"summary":{"frames":len(frames),"raw_track_ids":len({r["id"] for f in frames for r in f["robots"]}),
                         "projected_track_ids":len(tracks),"shown_track_ids":len(included),
                         "out_of_field_observations":outside,"short_tracks_omitted":len(tracks)-len(included)},
              "limitations":["Track IDs are not verified team identities and can swap or fragment.",
                 "Bounding-box bottom-center approximates floor contact; occlusion and camera changes degrade it.",
                 "The chosen auto start and camera calibration must be checked. No automatic clock or cutaway detection.",
                 "Field coordinates are 0..100 fractions, not measured meters."],
              "model_sha256":hashlib.sha256(args.weights.read_bytes()).hexdigest(),"source_sha256":hashlib.sha256(video.read_bytes()).hexdigest()}
        if args.reference:data["reference_score"]=score_reference(frames,args.reference)
        render_animation(data,args.out)
        print(json.dumps(data["summary"],indent=2),flush=True)
        print(f"Open {args.out/'animation.html'}; path video: {args.out/'paths.mp4'}",flush=True)
    finally:
        if capture is not None:capture.release()
        for writer in writers:writer.close()
        stop.set()


if __name__=="__main__":
    main()
