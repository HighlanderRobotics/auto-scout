"""Train a local YOLO detector; no Claude calls or dataset uploads."""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".venv" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def heartbeat(stop):
    started = time.monotonic()
    while not stop.wait(25):
        print(f"Training process active: {time.monotonic() - started:.0f}s elapsed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/checked_clip_v1/data.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/local_robot_v1"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("Choose a new output directory; existing runs are never overwritten")
    if not args.data.is_file():
        parser.error("Dataset YAML is missing; run prepare_dataset.py first")
    import torch
    from ultralytics import YOLO, settings
    # Explicitly disable optional online experiment integrations and telemetry.
    settings.update({k: False for k in ("sync", "hub", "wandb", "mlflow", "clearml", "comet", "neptune", "dvc", "raytune") if k in settings})
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; pass --device cpu if CPU training is intended")
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    try:
        model = YOLO(args.model)
        model.train(data=str(args.data.resolve()), project=str(args.out.resolve().parent),
                    name=args.out.name, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=0, seed=0, deterministic=True,
                    patience=12, cache=False, amp=False, plots=True, close_mosaic=10,
                    fliplr=0.0, flipud=0.0, exist_ok=False)
        out = Path(model.trainer.save_dir)
        manifest = args.data.parent / "manifest.json"
        metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "dataset": json.loads(manifest.read_text()) if manifest.exists() else None,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "python": os.sys.version}
        (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (out / "environment.txt").open("w") as file:
            subprocess.run([os.sys.executable, "-m", "pip", "freeze"], stdout=file, check=True)
        print(f"Training complete. Best weights: {out / 'weights/best.pt'}", flush=True)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
