"""First-run setup and launcher for the Auto-Scout Paths release (AGPL-3.0)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent


def logged(command, logfile):
    """Keep installation/inference visible and save the same output to a log."""
    done = threading.Event()
    def progress():
        while not done.wait(25):
            print("Auto-Scout is still working; see " + str(logfile), flush=True)
    threading.Thread(target=progress, daemon=True).start()
    try:
        with logfile.open("a", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace") as child:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return child.wait()
    finally:
        done.set()


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Install Python 3.11 or newer from https://www.python.org/downloads/ and retry.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="YouTube URL or local video file")
    parser.add_argument("--setup", action="store_true", help="Install dependencies only")
    parser.add_argument("--check", action="store_true", help="Check dependencies and bundled model without a video")
    parser.add_argument("--demo", action="store_true", help="Use the documented match and camera preset")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--auto-start", type=float)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prepare", action="store_true", help="Create a calibration page for a new camera")
    parser.add_argument("--device", help="cpu, mps or CUDA device number; default is automatic")
    parser.add_argument("--teams", help="Comma-separated match roster for local bumper OCR (macOS)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if args.source and args.demo:
        parser.error("Choose a source or --demo")
    envdir = ROOT / ".venv"
    python = envdir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = ROOT / "requirements.txt"
    stamp = envdir / "auto-scout-requirements.sha256"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not python.exists():
        print("Creating a private Python environment…", flush=True)
        venv.EnvBuilder(with_pip=True).create(envdir)
    if not stamp.exists() or stamp.read_text().strip() != digest:
        print("Installing dependencies. First setup downloads PyTorch and may take several minutes.", flush=True)
        if logged([str(python), "-m", "pip", "install", "-r", str(requirements)], ROOT / "setup.log"):
            raise SystemExit("Setup failed. See setup.log, fix the reported error, and rerun.")
        stamp.write_text(digest + "\n")
    if args.setup:
        print("Setup complete. Run Start Auto-Scout again to process a video.")
        return
    if args.check:
        manifest = json.loads((ROOT / "manifest.json").read_text())
        weights = ROOT / "models/robot.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise SystemExit("Bundled model checksum failed; extract a fresh release download.")
        code = ("import cv2, numpy, PIL, yt_dlp, imageio_ffmpeg, lap, torch; "
                "from ultralytics import YOLO; "
                "YOLO('models/robot.pt'); "
                "print('Dependencies and bundled model OK; device:', "
                "'mps' if torch.backends.mps.is_available() else 'cpu')")
        raise SystemExit(logged([str(python), "-c", code], ROOT / "setup.log"))
    from urllib.parse import urlparse, parse_qs
    source = "https://www.youtube.com/watch?v=EjF9we707DA" if args.demo else args.source
    if not source:
        source = input("Paste a YouTube URL or local video path (Enter for demo): ").strip().strip('"')
        source = source or "https://www.youtube.com/watch?v=EjF9we707DA"
    parsed = urlparse(source)
    identifier = (parsed.path.strip("/") if parsed.hostname == "youtu.be"
                  else parse_qs(parsed.query).get("v", [""])[0])
    if not args.teams and sys.platform == 'darwin' and not args.prepare:
        if identifier == 'EjF9we707DA':
            args.teams = '868,2910,2046,4065,4414,1323'
        elif sys.stdin.isatty():
            args.teams = input('Match team numbers, comma-separated (Enter for anonymous debug only): ').strip() or None
    if identifier != "EjF9we707DA" and args.calibration is None and not args.prepare:
        print("This camera needs calibration. Four matching floor landmarks are required.")
        if not sys.stdin.isatty():
            parser.error("Supply --calibration camera.json or --prepare --auto-start SECONDS")
        existing = input("Path to camera.json (Enter to create calibration): ").strip().strip('"')
        if existing:
            args.calibration = Path(existing).expanduser().resolve()
        else:
            args.prepare = True
    if args.prepare and args.auto_start is None:
        if not sys.stdin.isatty():
            parser.error("Calibration preparation needs --auto-start SECONDS")
        try:
            args.auto_start = float(input("Video time when autonomous begins, in seconds: "))
        except ValueError:
            parser.error("Start time must be a number")
    # Resolve user-supplied paths before running the child in the release folder.
    if Path(source).expanduser().is_file():
        source = str(Path(source).expanduser().resolve())
    out = args.out.expanduser().resolve() if args.out else ROOT / "outputs" / time.strftime("%Y%m%d-%H%M%S")
    if out.exists():
        parser.error("Output folder already exists; choose a new --out folder")
    command = [str(python), "-u", str(ROOT / "auto_paths.py"), source,
               "--weights", str(ROOT / "models/robot.pt"), "--out", str(out),
               "--duration", str(args.duration), "--device", args.device or "auto"]
    if args.calibration:
        command += ["--calibration", str(args.calibration.expanduser().resolve())]
    if args.auto_start is not None:
        command += ["--auto-start", str(args.auto_start)]
    if args.prepare:
        command += ["--prepare"]
    if args.teams:
        command += ["--teams", args.teams]
    print("Running locally. No Claude requests or API key required.", flush=True)
    result = logged(command, ROOT / "run.log")
    if result:
        raise SystemExit("Processing failed. See run.log for the cause; rerun with a new output folder.")
    page = out / ("calibrate.html" if args.prepare else ("teams/animation.html" if args.teams else "animation.html"))
    print("Open: " + str(page), flush=True)
    if args.prepare:
        print("Export camera.json from the page, then run again and provide that file.")
    if not args.no_open:
        webbrowser.open(page.as_uri())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped by user.")
