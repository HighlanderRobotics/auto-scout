Auto-Scout Paths — preview release

Install Python 3.11 or newer from python.org. Extract the entire ZIP first.
Mac: double-click Start Auto-Scout.command (or run python3 run.py).
Windows: double-click Start Auto-Scout.bat. Linux: python3 run.py.
First use installs dependencies and needs internet and several GB of free space.
Paste a YouTube URL; Enter uses the documented demonstration match.
No Claude account, API key, or training required. Model weights are included.

Commands:
  python3 run.py --demo
  python3 run.py --check
  python3 run.py VIDEO --prepare --auto-start 5.5
  python3 run.py VIDEO --calibration camera.json

For another camera, create calibration, open calibrate.html, select four matching
floor landmarks and export camera.json. Then rerun with that calibration.
Outputs appear in outputs/: animation.html, paths.mp4, paths.gif and paths.json.
Keep animation.html next to tracking.mp4. Logs: setup.log and run.log.

Prototype: anonymous IDs can swap or fragment. Field calibration and basket/ramp
footprints are approximate. No fuel is drawn. Other venues are not validated.
Mac tested; Windows/Linux launchers provided but not tested on those systems.
YouTube downloads depend on availability and yt-dlp support.

This release's Python/HTML sources and model are distributed under AGPL-3.0;
see LICENSE.txt. Includes an Ultralytics YOLO26n-derived detector.
Ultralytics licensing: https://www.ultralytics.com/license
Complete project and training source: https://github.com/HighlanderRobotics/auto-scout
