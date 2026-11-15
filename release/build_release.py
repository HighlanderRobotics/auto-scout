"""Build the redistributable Auto-Scout Paths ZIP from the checked-out project."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release" / "auto-scout-paths-v0.2.0.zip"
FILES = {
    "README.txt": ROOT / "release" / "README.txt",
    "LICENSE.txt": ROOT / "release" / "LICENSE.txt",
    "requirements.txt": ROOT / "release" / "requirements.txt",
    "run.py": ROOT / "release" / "run.py",
    "Start Auto-Scout.command": ROOT / "release" / "Start Auto-Scout.command",
    "Start Auto-Scout.bat": ROOT / "release" / "Start Auto-Scout.bat",
    "auto_paths.py": ROOT / "auto_paths.py",
    "auto_paths_viewer.html": ROOT / "auto_paths_viewer.html",
    "auto_paths_calibrate.html": ROOT / "auto_paths_calibrate.html",
    "team_paths.py": ROOT / "team_paths.py",
    "path_stitching.py": ROOT / "path_stitching.py",
    "read_team_numbers.swift": ROOT / "read_team_numbers.swift",
    "configs/einstein_final1_camera.json": ROOT / "configs" / "einstein_final1_camera.json",
    "models/robot.pt": ROOT / "runs" / "local_robot_v1" / "weights" / "best.pt",
}


def main():
    for name, path in FILES.items():
        if not path.is_file():
            raise SystemExit(f"Missing release file: {name}")
    manifest = {
        "version": "v0.2.0",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": hashlib.sha256(FILES["models/robot.pt"].read_bytes()).hexdigest(),
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in FILES.items():
            info = zipfile.ZipInfo("auto-scout-paths/" + name)
            info.external_attr = (0o100755 if name.endswith(".command") else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        archive.writestr("auto-scout-paths/manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise SystemExit("ZIP integrity check failed")
    print(OUT)


if __name__ == "__main__":
    main()
