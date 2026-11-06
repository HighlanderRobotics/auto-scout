# Auto-Scout: technical details

This is the detailed readme: how the pieces work, what we measured, the game and Lovat facts the code depends on, and technical notes for each research project. Start with [README.md](README.md).

## Goal

Scout a 2026 REBUILT match automatically from the webcast video. The output must be **per robot**: one report per robot per match, in the format of the team's scouting app, Lovat (https://github.com/HighlanderRobotics/lovat). Alliance totals alone are not enough.

Test match: Einstein Final 1, 2026 Championship, https://www.youtube.com/watch?v=EjF9we707DA. Red alliance: 4065, 4414, 1323. Blue alliance: 868, 2910, 2046. Final score: Blue 518, Red 456. Official per-period data: https://www.thebluealliance.com/match/2026cmptx_f1m1

## Multi-venue match pool

`configs/landmark_match_pool_v1.json` records a four-final pool selected from TBA:
CA District Sacramento Final 1, CA District San Diego Final 1, Wisconsin State
Championship Final 1, and Pacific Northwest District Championship Final 1. Sacramento
was already stored by the teacher experiment; the other three videos are saved in
`datasets/landmark_match_pool_v1/sources/`. Each file was opened with OpenCV after
download: all are 1920×1080, with 5,135–12,871 frames. The public match links and
YouTube source IDs are preserved in the config. No Claude calls or training happened
while collecting this pool.

This gives camera and venue variety for future evaluation and annotation. The videos
are not human-labeled, so they must not enter detector or landmark training until
their labels are checked. `track/final/` remains untouched as the answer key.

## Setup

- Mac. Tools: `brew install yt-dlp ffmpeg`. Python env: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Use `.venv/bin/python` for every script.
- API key: `.env` holds `CLAUDE_API_KEY=...`. `track_llm.py` reads it. Never print it, log it or commit it.
- Match video (not in git): `yt-dlp -f 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]' --merge-output-format mp4 -o einstein_final_1.mp4 'https://www.youtube.com/watch?v=EjF9we707DA'`
- Frames are not in git. Make them from the video (838 tracking frames, the clip links, and the pre-match frame):

  ```bash
  mkdir -p track/frames track/clip_frames
  ffmpeg -v error -y -ss 5.5 -to 173.1 -i einstein_final_1.mp4 -vf fps=5 -q:v 2 -start_number 0 track/frames/f_%04d.jpg
  for i in $(seq -f %04g 0 122); do ln -sf ../frames/f_$i.jpg track/clip_frames/f_$i.jpg; done
  ffmpeg -v error -y -ss 5.5 -i einstein_final_1.mp4 -frames:v 1 -q:v 2 start/prematch_full.jpg
  ```

## Pipeline

| Step | Command | Notes |
|---|---|---|
| Score overlay crops | `mkdir -p overlay && ffmpeg -v error -y -i einstein_final_1.mp4 -vf "fps=1,crop=1160:110:380:60" -q:v 2 overlay/o_%03d.jpg` | 1 crop per second of video |
| Read the overlay | `swift ocr_overlay.swift overlay/ > overlay_timeline.csv` | Apple Vision text recognition. Columns: video_s, blue_score, clock, red_score, shift, shift_clock |
| Fuel per period | `.venv/bin/python shift_summary.py` | Moves fuel back 3 s to the period it was scored in |
| Bumper OCR experiment | `ffmpeg ... -ss 3 -to 176 ... -vf "fps=2,crop=1920:540:0:160" field/f_%04d.jpg` then `swift find_bumpers.swift field/ 868,2910,2046,4065,4414,1323 2 > bumper_hits.csv` | Shows that plain OCR cannot name robots: 868 was never read |
| Draw field areas | `.venv/bin/python draw_labels.py start/prematch_labels.json start/prematch_full.jpg start/prematch_labeled.jpg` | Areas are hand-drawn and rough |
| Forward pass | `.venv/bin/python track_llm.py track/clip_frames OUT_DIR --model MODEL --seed start/prematch_labels.json [--actions] [--start N --end N]` | One Claude call per frame. Resumable: skips frames already done |
| Backward pass | `.venv/bin/python backward_pass.py OUT_DIR OUT_DIR_final` | Links boxes into paths; bumper reads anywhere on a path name the whole path |
| Labeled video | `.venv/bin/python render_labels.py track/clip_frames OUT_DIR_final OUT.mp4` | Large bold labels, 5 fps |
| Score a run | `.venv/bin/python compare_runs.py track/final opus OUT_DIR_final sonnet` | Run names must be `opus`, `sonnet` or `haiku` (used for prices) |

## Local dataset preparation

`prepare_dataset.py` converts the existing tracking JSON into YOLO-format detection
labels and cropped JPEGs. It does not call Claude, read `.env`, install a detector,
or train a model. An offline-data environment needs only:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt
.venv/bin/python prepare_dataset.py track/clip_frames track/final datasets/checked_clip_v1
.venv/bin/python -m unittest -v test_prepare_dataset
```

Use Python 3.10 or newer for Pillow 12+. On this Mac, Homebrew Python is
`/opt/homebrew/bin/python3`; the system `python3` was Python 3.9. The data requirements
are separate because local data preparation does not need the Anthropic SDK.

Restore the frames with the Setup commands first. If system FFmpeg is unavailable,
`imageio-ffmpeg` in the data requirements supplies a binary; get its absolute path
with `.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'`
and use that path in place of `ffmpeg` in those commands. Downloading the video and
extracting frames do not use the Claude API.

Default export behavior:

- Crop full-frame 1920x1080 images to `(0, 110, 1920, 720)`, producing 1920x610 images.
  Translate and clip boxes into that crop, then normalize center/width/height to 0–1.
- Export only `visible=true` boxes. Hidden robots have estimated positions and are
  excluded as detection targets. Visible, partially occluded robots remain included.
- Use one `robot` class. Optional `--classes alliance` produces `robot_red` and
  `robot_blue`; its team-to-alliance map is specific to the documented match.
- Split sorted labeled frames into chronological 70/15/15 blocks. Exclude ten frames
  at each boundary (`--gap-frames 10`), five on each side. For consecutive 5 fps labels
  this removes two seconds per boundary. Sparse source labels produce larger time gaps.
- Require every labeled frame's image, including excluded gap frames; fail rather
  than silently shrink the dataset. Reject invalid visible boxes and image sizes.
- Require a new output folder. Never overwrite existing datasets or source labels.

Output contains `images/{train,val,test}/`, `labels/{train,val,test}/`, `data.yaml`,
and `preview_train.jpg`, `preview_val.jpg`, `preview_test.jpg`. The previews draw boxes
from the exported text labels, allowing coordinate conversion to be checked visually.
`manifest.json` records source paths, source-label SHA-256 hashes, per-frame boxes,
omitted hidden/outside-crop boxes, split membership, and image/box totals. Datasets
are ignored by Git and can be recreated. The YAML contains an absolute dataset path;
regenerate it when moving the dataset to another computer.

The 123-frame answer key is a **pilot dataset**, with small evaluation blocks. A time
gap reduces adjacent-frame leakage, but all blocks still show the same robots and
venue. Evaluate on separate matches/venues before claiming generalization. Once a
frame is used in training, do not report it as an independent answer-key test.
Do not combine other model labels for these same images across different splits.

Detection evaluation must count false positives and missed reference robots as well
as box overlap. `compare_runs.py` is for the existing named-robot tracking output;
it is not a detector precision/recall evaluator, and the dataset export is not a new
tracking run. No model accuracy or inference cost is claimed for this preparation step.

Pilot export `datasets/checked_clip_v1` (2026-11-01, existing answer-key labels,
zero new API calls):

| Split | Images | Visible robot boxes |
|---|---|---|
| Train | 81 | 472 |
| Validation | 8 | 45 |
| Test | 14 | 84 |
| Excluded boundary gaps | 20 | Not exported |

All 123 source label files contained 719 visible boxes and 19 hidden estimates;
601 visible boxes remain after the split gaps. Previews were visually inspected.
Four offline tests cover crop normalization, hidden/invalid boxes, split separation,
exported labels/previews, missing-image rejection, and overwrite prevention. The
existing answer-key video can be regenerated with `render_labels.py` using the
restored clip frames; it is not a local model prediction.

## Local detector training

`train_detector.py` fine-tunes pretrained YOLO26n on the prepared one-class dataset.
Install `requirements-train.txt` using `.venv/bin/pip`. Python 3.15.1, Ultralytics
8.4.197, PyTorch 2.15.0 and torchvision 0.29.2 were used on the 16 GB Apple M5 Mac.
Ultralytics is AGPL-3.0; review its licensing before incorporating this implementation
into a differently licensed distributed application.

```bash
set -o pipefail
.venv/bin/python -u train_detector.py --out runs/local_robot_v1 2>&1 | tee /tmp/auto-scout-train-v1.log
.venv/bin/python evaluate_detector.py runs/local_robot_v1/weights/best.pt datasets/checked_clip_v1 runs/local_robot_v1_test
```

For assistants, run the training command as a background session and keep watching
the log. The script prints a heartbeat every 25 seconds, including during model
download and startup. Initial public weights are downloaded from Ultralytics GitHub
releases. Training data stays local; optional online experiment integrations and
telemetry are disabled. No Claude API calls or key are needed.

Defaults: 40 epochs, 960-pixel square training inputs, batch 4, MPS GPU, seed 0,
workers 0, early stopping patience 12. Validation uses only the validation block.
Best checkpoint selection is based on validation fitness, never test performance.
Horizontal and vertical flips are disabled; the default mosaic augmentation closes
for the final ten epochs. PyTorch warns that some MPS operations are nondeterministic,
so the fixed seed does not guarantee bit-for-bit reproduction. Pass `--device cpu`
only if CPU training is intended. Use a new `--out` path for every run.

Outputs under `runs/local_robot_v1/` include `weights/best.pt`, `weights/last.pt`,
`results.csv`, plots, `args.yaml`, `provenance.json` (including dataset split details),
and `environment.txt` with the installed package versions. Model weights and run
artifacts are ignored by Git. Preserve them locally; a source checkout alone does
not contain the trained model.

`evaluate_detector.py` predicts on the test split at a fixed confidence threshold
of 0.25 and measures one-to-one robot matching at IoU 0.3 and 0.5. It counts every
unmatched reference as a false negative and every unmatched prediction as a false
positive. Precision = TP/(TP+FP); recall = TP/(TP+FN). These are fixed-threshold
detection metrics, not average precision or team identity accuracy. The evaluation
outputs `metrics.json`, per-frame prediction/reference boxes, `preview.jpg`, and
`predictions.mp4`. Orange boxes are predictions; green boxes are the answer key.
The pilot video uses 5 fps, matching the consecutive test frames in this dataset.
Evaluation wall time includes prediction setup, matching, and JPEG rendering, so
it is not a standalone inference speed benchmark.

`compare_runs.py` cannot score this one-class detector meaningfully: it requires
known team numbers. Use the detection evaluator until tracking and team identity
are implemented; do not assign names from the answer key to inflate an identity score.
The test block contains only 14 images from the same match as training. Its scores
do not establish performance on a new match or venue.

### First trained baseline: local_robot_v1 (2026-11-01)

Training stopped at epoch 23 after 12 epochs without improvement in validation
fitness. The selected checkpoint is from epoch 11. The epoch timer reports 235 s;
model startup and final validation add overhead. No new Claude calls were made.
The checkpoint is `runs/local_robot_v1/weights/best.pt` (about 5.2 MiB).

Held-out test results, confidence fixed at 0.25, 14 images and 84 reference boxes:

| Box overlap requirement | Matched robots | Missed robots | Extra detections | Precision | Recall |
|---|---|---|---|---|---|
| IoU >= 0.3 | 83 | 1 | 10 | 89.2% | 98.8% |
| IoU >= 0.5 | 75 | 9 | 18 | 80.6% | 89.3% |

These are robot observations across frames, not 84 different robots. At the stricter
threshold, a poorly localized prediction contributes both an extra detection and a
missed reference. Visual inspection shows extra/overlapping predictions around the
crowded central robots. The baseline detects most robots at loose overlap, but its
precision and box placement still need work. No test-driven threshold tuning was done.

Checked-in summary: `results/local_robot_v1.json`, including checkpoint/dataset hashes.
Local visual artifacts: `runs/local_robot_v1_test/preview.jpg` and `predictions.mp4`;
raw predictions and per-frame counts are in `metrics.json`. The training log is
`runs/local_robot_v1/training.log`; training and evaluation source snapshots are also
saved with the run. Eight offline tests pass for dataset export and detection matching.
Next experiments should use more varied checked labels and a separate-match test,
then add tracking and identity. This pilot does not yet meet the end-to-end scouting target.

Sources: [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26/),
[Ultralytics training/MPS documentation](https://docs.ultralytics.com/modes/train/).

## Autonomous path animation

### Haiku appearance coaching experiment (2026-11-05)

The user approved a new **$2 total cap** for identity coaching. `coach_identity.py`
uses saved public Einstein match crops and the existing `start/robot_looks.json`
descriptions, without sending the checked answer key. It asks for identities from
two separate temporal crop groups per track; round 2 selects sharper crops from
temporal bins. Requests use Haiku only, disable automatic retries, cache responses,
and reserve $0.05 before each request in a locked ledger. An interrupted request
retains its reservation. Both rounds and optional detector grading share that ledger.
The earlier documentation had reversed the Einstein alliance rosters; blue is
868/2910/2046, red is 4065/4414/1323, as recorded in `track_llm.py`.

```bash
# Paid: run only after cost/cap approval. These are the recorded pilot inputs.
.venv/bin/python coach_identity.py --out datasets/identity_coach_v1 --round 2 --base-evidence runs/team_paths_free_evidence_pixels_v1/replayed_color_evidence.json --execute
# Offline: train a small local appearance gallery from the accepted identities.
.venv/bin/python train_identity_gallery.py --named-paths runs/haiku_identity_round2/paths.json --crops runs/team_paths_free_evidence_pixels_v1/ocr_crops --out runs/local_identity_gallery_v1
```

Round 1 at a 0.9 teacher-confidence gate did not increase coverage. Round 2's
offline replay used the existing resolver's 0.8 gate and still required two
supporting crop groups. `compare_runs.py` and `evaluate_team_paths.py` measured
229/229 correct identities on matched boxes versus 224/224 before. Coverage rose
from 38.6% to 39.4% of 581 visible reference observations; four teams were resolved,
including 2910. This is a small same-match experiment, with two remaining teams
unresolved. It does not establish generalization or complete scouting accuracy.

The local HSV spatial appearance gallery used 49 training crops and 47 held-out
crops from accepted tracks (2046, 2910, 4065, 4414). It agreed with their pseudo-labels
on 46/47 held-out crops (97.9%). This is same-track teacher/OCR-label agreement,
not independent team accuracy: unseen teams, background cues, and varying views
remain untested. Its `gallery.npz` and `metrics.json` are in
`runs/local_identity_gallery_v1/`. It remains experimental and is not wired into
automatic identity assignment.

`--grade --execute` asks Haiku to review four local detector frames separately.
The first grader version mixed original and resized coordinates and produced one
truncated response, so it is excluded from numerical conclusions. The corrected
version sends matching image/box coordinates and structured responses. Teacher
opinions supplement the checked answer-key scores; they do not replace them.

Final recorded pilot spend: **$0.0311586**, plus a retained **$0.05 reservation**
for the interrupted request, within the approved $2 total cap. The corrected teacher
review flagged possible duplicates/background detections and misses in four frames;
these are qualitative diagnoses, not detection truth. See
`results/haiku_identity_coaching_v1.json`. No training or paid job remains running.

### Team-number paths

Add `--teams COMMA_SEPARATED_ROSTER` to `auto_paths.py` to run local bumper OCR
after tracking. The release launcher source also accepts this flag. This feature
requires macOS Apple Vision/Swift; it adds no paid requests. Existing v0.1.0 release
downloads predate this change.

For saved detections, run:

```bash
.venv/bin/python team_paths.py runs/auto_paths_youtube_demo/paths.json --video einstein_final_1.mp4 --teams 868,2910,2046,4065,4414,1323 --blue-teams 868,2910,2046 --red-teams 4065,4414,1323 --out runs/my_team_paths
```

The video must be the original full-frame source at the same timeline/resolution.
`team_paths.py` samples crops no more often than every 0.5 seconds per anonymous ID.
`read_team_numbers.swift` reads text locally. Exact roster numbers and unique
two-or-more-digit roster fragments (for example `13` for 1323 only when no other
roster team contains it) at confidence >=0.8 count, with at least two distinct crop
times. Supplying both three-team alliance rosters enables repeated high-confidence
bumper-color evidence, but color never names a robot alone: it can only join a close
same-color fragment to an OCR anchor, or infer the third team when exactly one
same-color unresolved track remains after the other two teams are anchored. Supported
competing numbers reject the whole track; overlapping fragments claiming one team
reject that team's grouping. This guards some errors but repeated OCR mistakes and
unobserved tracker swaps can still cause incorrect assignments. OCR reads are evidence,
not verified identities.

`animation.html` and MP4/GIF show one path per resolved team. `debug/` keeps the
original anonymous paths and video; the video retains anonymous labels. `paths.json`
records source IDs, unresolved counts and conflicts, while `evidence.json`,
`crop_manifest.json` and `ocr_crops/` expose every number read for review.
Observed paths remain solid. Gaps up to two seconds with displacement <=20 field
percentage units per second use dashed straight estimated joins. Longer gaps remain
unknown; no positions are fabricated there. The six-team roster alone does not infer
an unreadable number. `--evidence` reuses saved reads without rerunning Vision.

**Free-evidence v3 check (2026-11-05):** unique partial reads, repeated bumper-color
filtering, close same-color fragment joins, and the strict leftover-team rule were
implemented locally. A replay against the checked Einstein clip used the saved Vision
reads plus 221 newly measured color observations: it still resolved 2046, 4065 and
4414, with 224/224 matched identities correct and 38.6% named-reference coverage.
No candidate met the strict join/leftover conditions, so coverage did not increase.
This is intentional: the rules must not manufacture names merely to improve a metric.
The measured record is `results/team_paths_v3_free_evidence.json`; new API cost $0.

The same-color fragment rule predicts the named robot's short future path from its
last two field observations. It can carry a confirmed identity across up to 1.5 seconds
of tracker loss only when the reappearing fragment has repeated matching bumper-color
evidence and lands near that predicted path. Motion alone never names a robot.

Four additional offline tests check fragment merging, independent votes, conflicting
numbers and simultaneous duplicate team assignments; all 32 tests pass.

Demo result: `runs/team_paths_v2/` sampled 238 crops and resolved 2046 (IDs 1/20),
4065 (ID 8) and 4414 (ID 7). The other three teams remain unresolved. Against 100
answer-key timestamps, 224 of 226 named detections matched at IoU >=0.3 and all 224
had the correct identity. Correct named coverage was only 224/581 visible reference
boxes (38.6%); accuracy on resolved boxes alone must not be advertised as full tracking
accuracy. `evaluate_team_paths.py` exported the run and invoked `compare_runs.py`;
metrics and comparison are in `runs/team_paths_v2_score/`. New API cost: $0.
The launcher source selects the demo roster automatically on macOS and opens the team
pane; other interactive runs can enter a roster. The existing downloadable release
has not been replaced.

`auto_paths.py` accepts a YouTube video link or a local video and produces an
autonomous-window tracking video and a 2D path animation. It uses the preferred
local v1 detector plus ByteTrack, not Claude. Install `requirements-paths.txt`.
The trained weights are ignored by Git: pass `--weights` to an existing local
checkpoint, or use the default `runs/local_robot_v1/weights/best.pt`.

```bash
# Documented match: includes an approximate camera preset and start at video 5.5 s.
.venv/bin/python auto_paths.py 'https://www.youtube.com/watch?v=EjF9we707DA' --out runs/my_auto_paths

# Existing video, explicit calibration, optional answer-key comparison:
.venv/bin/python auto_paths.py einstein_final_1.mp4 --calibration configs/einstein_final1_camera.json --out runs/my_local_paths --reference track/final

# Other camera: prepare a clickable reference image, then export camera.json.
.venv/bin/python auto_paths.py 'YOUTUBE_URL' --prepare --auto-start VIDEO_SECONDS --out runs/camera_setup
# Open runs/camera_setup/calibrate.html, select four matched floor landmarks,
# set their field percentages and the camera crop, and download camera.json.
.venv/bin/python auto_paths.py runs/camera_setup/source.mp4 --auto-start VIDEO_SECONDS --calibration /path/to/camera.json --out runs/calibrated_auto
```

Every output directory must be new. Default autonomous duration is 20 s, sampled
at 10 fps; `--auto-start` and `--duration` select the video window. There is no
automatic match-clock reading or replay/camera-change detection in this utility.
For an unknown YouTube link, a start time and camera calibration are required;
the built-in preset is selected only for video ID `EjF9we707DA`.

Output files:

- `animation.html`: offline canvas pane with play/pause, scrubber, speed, trail-length,
  and track visibility controls; paired with `tracking.mp4` for video evidence.
- `paths.mp4`, `paths.gif`, `paths_preview.png`: rendered 2D animation and still.
- `paths.json`: timestamped detections, full-frame boxes, normalized positions,
  anonymous track IDs, calibration, input/model hashes and optional reference scores.
- `calibration.json`, `calibration_preview.jpg`, `bytetrack.yaml`: run configuration.
- `calibrate.html` from preparation mode: click four coplanar floor landmarks and
  enter their matching normalized top-down coordinates to export a calibration file.

The pane and exported video/GIF draw two color-coded hubs/baskets and four ramps
as static schematic landmarks, with a center line and robot paths overlaid. No fuel
is drawn in the 2D pane (the original video naturally still shows fuel). The same
`field_elements` metadata in `paths.json` supplies both renderers. These are illustrative
normalized footprints, not surveyed dimensions, obstacle collision checks, or a change
to the camera calibration. Hub/bump arrangement follows the
[FIRST manual sections 5.4 and 5.5](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm).

`--render-paths PATHS_JSON --out NEW_DIRECTORY` re-renders saved coordinates using
the current field drawing. It copies the existing tracking video and calibration
assets when present, does not load a model, and makes no download or API requests.
Existing output folders are preserved. The field-element demonstration is saved
under `runs/auto_paths_field_elements_v2/`; its tracks and video detections are identical
to `runs/auto_paths_youtube_demo/`. The 200-frame, 10-fps export and absence of fuel
elements were verified, and all 28 existing tests pass.

The homography maps each bounding box's bottom center to normalized field percentages
(0–100 on each axis), not meters. Four image/field correspondences must be in matching
convex order. The demonstration preset comes from the rough neutral-zone polygon in
`start/prematch_labels.json`; it is explicitly approximate, not surveyed geometry.
Only use it for that fixed camera. Camera motion, robot heights and occlusion can
distort projected positions. Out-of-field projections are omitted rather than clamped.

ByteTrack runs on the cropped field view at detection confidence 0.1, with new-track
threshold 0.4 and a one-second lost-track buffer. IDs are anonymous observations,
not verified team numbers. Occlusions, false detections and association errors can
create more than six IDs. Tracks with fewer than three projected observations are
omitted by default. Trails break across gaps >0.6 s or jumps >25 normalized units;
the viewer does not join those gaps into an invented continuous path.

Checkpoint test on 2026-11-02: actual YouTube download, 200 frames over video seconds
5.5–25.5, 20-second animation. There were 23 observed IDs, 19 with in-field points,
and 16 displayed tracks after filtering; 41 observations projected outside the field.
This demonstrates the end-to-end utility, not reliable six-robot identity tracking.
Against 100 matching answer-key timestamps at IoU 0.5: 515 TP, 46 FP, 66 FN,
91.8% precision and 88.6% recall. At IoU 0.3: 549 TP, 12 FP, 32 FN (97.9% precision,
94.5% recall). Those scores measure tracked boxes, not 2D-position accuracy. No field
ground truth exists here, and `compare_runs.py` requires team names, so the utility
uses the anonymous detection evaluator instead. Twenty-eight offline tests pass.

Example artifacts are in `runs/auto_paths_youtube_demo/`; run log is
`/tmp/auto-scout-youtube-path-demo.log`. A local-file test caught a PIL list-color
rendering error, which was fixed before the successful YouTube test. Calibration
preparation was exercised on the saved video. Browser checks confirmed field drawing,
play/pause, speed selection, track visibility and the pane's time slider. Video
scrubbing did not follow the slider in the local HTTP preview; synchronized seeking
still needs verification with a server that supports video byte ranges.

Sources: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/),
[OpenCV homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).

## Confident-region training

The original full-frame export accepted only four training images and one validation
image, so student training never started. `prepare_confident_regions.py` reuses the
saved labels offline without changing their confidence values or weakening the 0.5
confidence cutoff. It selects at most one image region per third source frame,
keeping train/validation matches separate and copying the original 14 checked test
images and labels byte-for-byte.

For each frame, visible, finite, in-bounds boxes with confidence >=0.5, width 12–480
pixels, and height 12–400 pixels are positive candidates. Other nonempty box estimates
are uncertainty regions, including hidden robots. Candidate crop widths are 1920,
1280, 960 and 640 pixels at left, center, and right positions; height remains 610.
Reject any crop intersecting an uncertainty region. Label every confident candidate
that intersects the crop, requiring at least half the source box to remain and at
least eight pixels per dimension after clipping. Reject overlapping duplicate boxes
above IoU 0.7. Select the crop with most positive boxes, breaking ties by width, then
leftmost position. This avoids dropping individual weak detections while leaving
their estimated image regions as unlabeled background.

Eighteen deterministic sample crops (six per match, spread across time) were inspected
visually by the assistant. Two contained visible unannotated robot fragments and were
excluded; the review records and exact crop coordinates are saved in
`configs/confident_region_reviews.json`. The other crops remain pseudo-labels, not
human-verified answer keys. The review does not establish correctness for every image.
Teacher omissions outside their estimated boxes remain possible. Crops also change
object scale relative to full-field test images.

```bash
.venv/bin/python prepare_confident_regions.py --out datasets/confident_regions_v1
.venv/bin/python -m unittest -q test_prepare_dataset test_evaluate_detector test_teacher_pipeline test_confident_regions
set -o pipefail
caffeinate -i .venv/bin/python -u train_detector.py --data datasets/confident_regions_v1/data.yaml --out runs/local_robot_v2_regions --epochs 40 2>&1 | tee /tmp/auto-scout-train-regions-v2.log
.venv/bin/python evaluate_detector.py runs/local_robot_v2_regions/weights/best.pt datasets/confident_regions_v1 runs/local_robot_v2_regions_test
```

Prepared dataset: 141 training images/268 boxes, 86 validation images/140 boxes,
and 14 full-field test images/84 boxes. Sources and byte hashes are recorded in
`datasets/confident_regions_v1/manifest.json`. The crop policy is fixed before student
test evaluation. A pre-test calibration check on the existing 40-frame teacher audit
selected 11 crops: 31 TP, 2 FP, 7 FN at IoU 0.5 (93.9% precision, 81.6% recall).
This measures annotation agreement on a small calibration sample, not student accuracy.
Calibration requires at least five crops and >=90% precision; no test images are
used in that gate. The known omissions are why the run is an experiment, not a claim
that all saved labels are correct.

The new run uses the same pretrained YOLO26n initialization, image size 960, batch 4,
seed 0, and validation-based early stopping as the first baseline. No additional
Claude requests or API charges are required. Twenty-three offline tests now cover
the existing logic plus uncertain-region exclusion, duplicate boxes, positive
retention, and confidence preservation. Output directories must be new; staging
prevents a failed export from leaving a partial dataset at its final path.

### Trained result: local_robot_v2_regions (2026-11-02)

The run completed 37 epochs in about 10.4 minutes on the epoch timer, stopping after
12 epochs without better validation fitness. The selected checkpoint is from epoch
25. Evaluation used the unchanged 14 full-field test images at confidence 0.25.

| Model | TP at IoU 0.5 | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| First baseline (`local_robot_v1`) | 75 | 18 | 9 | 80.6% | 89.3% |
| Confident regions (`local_robot_v2_regions`) | 70 | 12 | 14 | 85.4% | 83.3% |

At IoU 0.3, the new run has 73 TP, 9 FP, 11 FN (89.0% precision, 86.9% recall).
It makes fewer extra detections at the stricter overlap threshold but misses more
robots. This is not a clear overall improvement; retain v1 as the preferred baseline
for recall and keep v2 as an experimental checkpoint. Neither model is a complete
scouting system or assigns reliable team identities.

Only 14 test images are available. The first baseline trained on earlier frames of
that same match, whereas this experiment trained on different matches and venues.
The comparison is useful for the current clip but does not prove cross-event quality.
No test threshold or training setting was changed after seeing these results.
Full-field annotation quality and crowded scenes remain priorities before scaling
training. More epochs alone do not address incorrect or missing labels.

Summary and hashes: `results/local_robot_v2_regions.json`. Local checkpoint:
`runs/local_robot_v2_regions/weights/best.pt`. Inspect `preview.jpg`, `predictions.mp4`,
and raw `metrics.json` under `runs/local_robot_v2_regions_test/`. Logs, package
versions, source snapshots and training arguments are saved with the run.
This round made zero Claude calls and added $0 API cost; previous teacher labeling
remains about $1.64, within the unchanged $4 cap.

## Haiku teacher pipeline

### Active multivenue teacher run

`configs/haiku_multivenue_v1.json` labels 1,800 frames at 3 fps from four saved
2026 finals: Sacramento, San Diego, Wisconsin, and Pacific Northwest. Sacramento,
San Diego, and Wisconsin are training sources; Pacific Northwest is validation. The
old 14 checked Einstein frames remain test only. It runs in
`datasets/teacher_multivenue_v1/` with output planned at
`runs/local_robot_multivenue_v1/`.

The user approved this Haiku 5.5-only run on 2026-11-02. Its recorded historical
average was $0.000714 per frame, giving a $1.29 estimate. The run-local ledger has
a hard $2.30 cap, keeping the combined prior spend, existing reservations, and new
maximum within the earlier $4 total limit. No other Claude model is permitted. The
existing prompt audit passed on checked Einstein frames, but it does not validate
label quality at the new venues; inspect samples and the held-out PNW validation
before treating a trained checkpoint as an improvement. Progress appears in
`datasets/teacher_multivenue_v1/status.json` and `logs/teacher-multivenue-v1.log`.

The strict full-frame export stopped after labeling: only 3 of 1,793 field images
showed six visible robots with every confidence >=0.5, so it correctly refused to
train rather than treating omissions as background. This did not send retries or
spend more money. `prepare_confident_regions.py` then reused the saved labels offline
with `configs/multivenue_confident_region_reviews.json`. Twelve samples were reviewed;
one Wisconsin title-card transition was rejected. The resulting dataset has 127 train,
72 validation, and the unchanged 14 checked test images, with 225/122/84 boxes. Its
existing teacher-audit calibration has 31 TP, 2 FP and 7 FN at IoU 0.5 (93.9%
precision, 81.6% recall); this checks label quality only on the old Einstein audit.
The automatic local fallback trains `runs/local_robot_multivenue_regions_v1/` and
then evaluates it in `runs/local_robot_multivenue_regions_v1_test/`. Results pending.

`run_teacher_training.py` runs the plan in `configs/haiku_teacher_v2.json`:
label with `claude-haiku-5-5`, filter pseudo-labels, train YOLO26n, then evaluate
on the original 14 checked test images. Sonnet is disabled and its comparison is
postponed. The user set the current **$4 maximum on 2026-11-02**, lowering the earlier
$9 cap after replacing the local key. Prior recorded spending and uncertain request reserves remain in the
ledger, so the limit is conservative across the key change. Expected labeling
cost for the three-match plan is about $2; unused budget is not spent automatically.

Sources (videos discovered from public TBA match pages and visually checked):

| Match | Video ID | Use |
|---|---|---|
| [Silicon Valley Quals 71](https://www.thebluealliance.com/match/2026casnv_qm71) | `y_vBDwf4Cbc` | Train |
| [Sacramento Finals 1](https://www.thebluealliance.com/match/2026casac_f1m1) | `Ygzo0f0k0mk` | Train |
| [Silicon Valley Quals 65](https://www.thebluealliance.com/match/2026casnv_qm65) | `oSjsixPUjlY` | Validation |

The first 150 video seconds per source are extracted at 5 fps with the same
1920x610 field crop: 750 frames per match, 2,250 requests at most. Title cards
and fully hidden robots are excluded. The teacher gets the roster, previous image,
and previous positions; team assignments help continuity but are discarded for
one-class detector training. Every third labeled frame is considered for export,
reducing adjacent-frame duplication in the student dataset.

Conservative export accepts only frames with six visible robots, confidence >=0.5
for each, and no pair of boxes with IoU >0.7. This reduces incomplete labels, but
cannot guarantee correctness and biases the dataset against crowded/hidden scenes.
Labels remain **unreviewed pseudo-labels**, not new answer keys. Training requires
at least 100 accepted training images and 30 validation images. Different matches
are kept in separate splits; no old Einstein frames enter student training.

The teacher audit used a checked frame-0 seed and evaluated consecutive frames 1–40:

| IoU | True positives | False positives | False negatives | Precision | Recall |
|---|---|---|---|---|---|
| 0.3 | 216 | 3 | 18 | 98.6% | 92.3% |
| 0.5 | 212 | 7 | 22 | 96.8% | 90.6% |

The gate requires >=90% precision and recall at IoU 0.5. It passes, but visible
robots are still sometimes marked hidden. A generic single-image labeling prompt
failed this pilot and was not used for training. This is not full-match accuracy.
`results/haiku_teacher_audit.json` records the audit; an adapter restores full-frame
coordinates in `runs/haiku_teacher_audit_tracking/` for `compare_runs.py`. That
comparison reports 90% names and 97% overlapping boxes on the 40 audit frames,
about $0.03. Its name percentage is conditional on matched boxes; use the detection
audit above to see missed reference robots explicitly.

Install `requirements-teacher.txt` with `.venv/bin/pip`. Before any new paid run,
give an estimated cost and obtain authorization. After approval, a foreground
command is:

```bash
set -o pipefail
caffeinate -i .venv/bin/python -u run_teacher_training.py 2>&1 | tee datasets/teacher_v2/job.log
```

Assistants should run this in a background session. The launched job uses `nohup`,
a separate process group and `caffeinate` so it can continue after the launching
shell exits and prevent idle sleep. Keep the Mac plugged in with the lid open.
The job writes progress every frame and a heartbeat at least every 25 seconds.
Inspect `datasets/teacher_v2/status.json`, `job.log`, and `launch.json` (process group).
On failure the status changes to `stopped`; there are no automatic API retries.
An uncertain in-flight request must be reconciled before retrying that frame.
Completed labels are reused after checking image, model, and prompt hashes.

The billing ledger at `datasets/teacher_v2/billing/ledger.json` reserves $0.01 before
each request and counts reported usage at $0.10/$0.50 per million input/output tokens.
Failed/uncertain attempts retain their reserves. The client sends only bounded
images and fixed prompts, with a 4,096-output-token limit. It stops when the next
reserve would cross $4. This ledger tracks this job; it does not control other use
of the account. The key is read from `.env`, never copied into the plan or logs.

The audited label method is fingerprinted separately from budget settings. The
$5-to-$9-to-$4 cap changes preserved all usage and reserves, and the label method AST was
verified unchanged before retaining the audit. The original audited source snapshot
is under `datasets/teacher_v2/provenance/`.

The running job was paused between completed API requests to lower the cap from $9
to $4, then resumed from cached labels. No paid request was repeated during that change.

On success, the job writes `datasets/teacher_v2/result.json`, trained weights under
`runs/local_robot_v2/weights/`, and a preview/video plus precision/recall under
`runs/local_robot_v2_test/`. These are ignored local artifacts. No student-v2 result
is claimed until training and evaluation finish. Eighteen offline tests cover the
dataset/evaluation logic, filtering, cap enforcement, failed-request reserves, and
cached-label reuse; none sends API requests. Hidden robots may return an empty box:
the parser preserves a previous estimate for continuity while keeping `visible=false`.
Raw responses are retained under each label folder's `raw/` subdirectory. An already
billed response with no usable label is quarantined without another paid request;
the next frame uses the actual elapsed gap. The first such recovery excluded frame
320 of Silicon Valley Quals 71 after 320 successful labels. Audit and cached labels
were retained after verifying that API request construction was unchanged.

Visible boxes with zero area, reversed corners, non-finite coordinates, or substantial
out-of-frame positions are rejected as `InvalidTeacherLabel`. Tiny boundary overshoots
of at most two sent-image pixels are clipped and recorded in `box_adjustments`.
The job quarantines malformed paid responses and continues with the last valid
tracking state and the actual elapsed frame gap; it never retries those responses.
Ten consecutive malformed responses stop the job to avoid wasting credits. API,
authentication, budget, and uncertain-network failures still stop immediately.
Raw text is preserved even when JSON is malformed. A geometry error at frame 357
of the first match was repaired this way; 356 completed labels and all billing
records were retained. Quality requirements for training were not loosened.

## Frames and coordinates

- `track/frames/f_0000.jpg` … `f_0837.jpg`: 5 fps from video time 5.5 s (pre-match, clock 0:20) to 173 s (clock 0:00). Frame `i` is at video time `5.5 + i/5`.
- `track/clip_frames/`: symlinks to the first 123 frames (5.5–30 s). This is the test clip.
- Frames are 1920x1080. The broadcast is split: the wide field camera is the top part (`y` 110–720); the bottom is a second camera. `track_llm.py` crops to `(0, 110, 1920, 720)` and scales by 0.8 before sending; all boxes in JSON results are full-frame pixels.
- Score overlay positions (full frame): blue score x≈780–900, clock x≈900–1020, red score x≈1020–1140, all at y≈65–125. Shift counter `n/6` and shift clock at x≈900–1020, y≈128–165.

## Answer key

`track/final/` is the Opus 5.5 run on the clip after the backward pass. A mentor checked it by hand: all names are correct. Score other models and prompt changes against it with `compare_runs.py`.

## Model results on the clip (123 frames)

| Model | Boxes in the right place | Correct names | Cost |
|---|---|---|---|
| `claude-opus-5-5` (effort medium) | 100% | 100% (answer key) | $2.60 |
| `claude-sonnet-5-5` (effort medium) | 97% | 95% | $1.32 |
| `claude-haiku-5-5` + robot descriptions | 100% | 98% | $0.11 |
| `claude-haiku-5-5` | 96% | 94% | $0.10 |
| `claude-haiku-4-5` | 4% | not usable | $0.62 |

Sonnet's and Haiku 5.5's errors were all the same 868/2910 swap from about 11 s to 23 s. Adding one line per robot about how it looks (`start/robot_looks.json`) cut Haiku 5.5's errors to 11 frames (21.1–23.1 s, 2910 called 868).

Cost experiments with Haiku 5.5 + robot descriptions:

- **One image per call:** not worth it. Haiku's cost is mostly output (about 1,000 tokens per frame), so dropping the previous image saved only about 10%, and calls were not faster. Without the previous picture the boxes jump more, the backward pass split robots into 11 path pieces instead of 8, and names fell to 83%.
- **2.5 frames per second:** half the cost, but the 868/2910 swap came back from the start of the clip to 22.7 s. Names 86%, boxes 93%.
- In both runs the backward pass made names worse than the forward pass alone. Its voting needs work: it should trust clear bumper reads more and change fewer names.
- Next lever for Haiku's cost: shorter output. No model ever read 868's bumper. Cost is about 3,400 input tokens and 380 output tokens per frame, mostly the two images. Ideas to try: send only the current frame; describe each robot's look in the prompt; Opus at effort `low`.

## REBUILT rules the code depends on

- Auto 20 s, then a 3 s pause, then teleop 2:20: transition 10 s (2:20–2:10), shifts 1–4 of 25 s each, end game 30 s (0:30–0:00).
- Overlay `n/6`: 1 = transition, 2–5 = shifts 1–4, 6 = end game. Not shown in auto.
- Both hubs are active in auto, transition and end game. In shifts 1–4 one hub is active at a time. The alliance with more auto fuel has its hub off in shift 1, then they alternate. Fuel still counts for 3 s after a hub turns off.
- 1 point per fuel in an active hub. Tower: auto L1 15 points; teleop L1/L2/L3 10/20/30. Climb level per robot is in the official data, so it does not need scouting.
- The yellow arrow beside an alliance name means its hub is active. The small outer boxes show live alliance fuel count.

## Lovat report format

Source: the Lovat repo, `apps/collection/lib/collection/ScoutReport.ts` and `packages/db/prisma/schema.prisma`.

- Events: `[time_s, action, position, quantity?]`. Actions: START_MATCH, START/STOP_SCORING (fuel count on STOP), START/STOP_FEEDING, INTAKE, OUTTAKE, DISRUPT, CROSS, CLIMB, START/STOP_CAMPING, START/STOP_DEFENDING. Positions: LEFT_TRENCH, LEFT_BUMP, HUB, RIGHT_TRENCH, RIGHT_BUMP, NEUTRAL_ZONE, DEPOT, OUTPOST, NONE.
- Post-match fields: robotRoles, accuracy (0–5 bands), feederTypes, defenseEffectiveness (0–4), intakeType, fieldTraversal, autoClimb, endgameClimb, beached, scoresWhileMoving, disrupts, driverAbility (1–5), robotBrokeDescription, notes.
- Upload: `POST /v1/manager/scoutreports`. A video scout should use its own scouter ("Video Scout") so analysts can filter it.

## Research projects: technical details

The README explains each research project. This section gives the technical details. The projects are not in a fixed order. Whatever you try, score it against the answer key (`track/final/`) with `compare_runs.py` where that applies, and record the result in the results table below.

Everything in a Lovat report is built from three inputs:

1. **Robot paths:** boxes per frame, after the backward pass.
2. **Robot actions:** shooting, feeding, intaking, defending, climbing, driving or stationary, per frame (`track_llm.py --actions`).
3. **Official numbers:** alliance fuel per period (TBA `hubScore`, or `overlay_timeline.csv` second by second), climb level per robot and fouls (TBA).

### Results table (keep it up to date)

| Run | Model | Changes | Correct names | Boxes in the right place | Cost per frame | Cost per match (840 frames) |
|---|---|---|---|---|---|---|
| answer key | Opus 5.5, effort medium | two images | 100% | 100% | $0.021 | about $18 |
| baseline | Sonnet 5.5, effort medium | two images | 95% | 97% | $0.011 | about $9 |
| best so far | Haiku 5.5 | two images, robot descriptions (`--robots start/robot_looks.json`) | 98% | 100% | $0.0009 | about $0.75 |
| | Haiku 5.5 | one image (`--one-image`), robot descriptions | 83% (94% before the backward pass) | 96% | $0.0008 | about $0.70 |
| | Haiku 5.5 | 2.5 frames per second (`--step 2`), two images, robot descriptions | 86% (92% before the backward pass) | 93% | $0.0010 per frame used | about $0.40 |
| | Haiku 5.5 | two images | 94% | 96% | $0.0008 | about $0.70 |
| | Haiku 4.5 | two images | not usable | 4% | $0.005 | — |
| local_robot_v1 | YOLO26n | Local detector; 81 training images, 14 held-out test images | Not implemented | 98.8% recall at IoU 0.3; 89.2% precision | $0 API | $0 API for detector; full pipeline not measured |
| local_robot_v2_regions | YOLO26n | 141 teacher-region training images, 86 validation images, same 14 full-field test images; experimental | Not implemented | 86.9% recall at IoU 0.3; 89.0% precision | $0 new API | $0 API for detector; full pipeline not measured |
| local_robot_multivenue_regions_v1 | YOLO26n | 127 multi-venue teacher-region training images, 72 validation images, same 14 full-field test images; experimental | Not implemented | 78.6% recall at IoU 0.3; 98.5% precision | $1.4247 Haiku 5.5 | Fewer extra detections but substantially lower recall than v1; do not replace v1 |
| auto path demo | YOLO26n v1 + ByteTrack | 20 s / 200 frames; 100 checked timestamps; approximate field projection | Anonymous IDs | 88.6% recall at IoU 0.5; 91.8% precision | $0 API | $0 API; full-match run not measured |
| teacher audit | Haiku 5.5 | Sequential prompt, 40 checked frames after frame-0 seed; before identity cleanup | 90% | 97% matched overlap; see detection recall above | about $0.0007 | Not measured on full match |
| identity coaching pilot | Haiku 5.5 + existing local tracks | Two crop-selection rounds; four teams accepted | 229/229 matched names; 39.4% reference coverage | Detector unchanged | $0.03116 total recorded pilot, plus $0.05 uncertain reserve | Not measured on full match |

Target for scouting hundreds of matches: at least 98% correct names and 95% boxes in the right place, below $1 per match.

### Trained result: `local_robot_multivenue_regions_v1` (2026-11-03)

The multi-venue Haiku teaching run made 1,795 billed calls across Sacramento, San Diego, Wisconsin, and Pacific Northwest championship finals. Five invalid labels were excluded without retrying. Teacher spend was **$1.4247**, inside the $2.30 run cap. The strict full-field automatic filter only retained 3 frames, so the local training set safely fell back to reviewed confident robot-region labels: 127 train images / 225 boxes, 72 validation images / 122 boxes, and the existing independent 14-image / 84-box full-field test set.

YOLO26n selected epoch 4 and stopped early after 16 of 60 requested epochs. At IoU 0.5 it found 64 of 84 test robots with 3 false positives and 20 misses (95.5% precision, 76.2% recall). The existing `local_robot_v1` found 75 with 18 false positives and 9 misses (80.6% precision, 89.3% recall). The new experiment is very conservative—at IoU 0.3 it reaches 98.5% precision—but it misses too many robots to replace v1 for path tracking. Keep it as an evaluated experiment, not the default detector.

### Project: Teach a local model with Claude

The idea: Claude is accurate but costs money per frame. A small detector (for example YOLO or RF-DETR) runs free on a Mac once it is trained. Use Claude as the teacher:

- Training labels: the boxes from the Opus and Sonnet runs (`track/final/`, `track/sonnet_final/`). Label more frames with Sonnet from several matches and venues (about $0.01 per frame). One class ("robot") or two ("red robot", "blue robot").
- Train the detector on the labels, then follow robots between frames with a standard multi-object tracker (for example ByteTrack or BoT-SORT).
- Call Claude only for the hard moments: when the tracker loses a robot, when two same-color robots pass close, or to read a bumper from a small crop (a crop costs about 1/20 of a full frame).
- A robot looks the same all event, so a small gallery of checked crops per team can name robots by look, using a free image-similarity model.
- Watch the licenses: Ultralytics YOLO is AGPL.
- Open questions: how many labeled frames are needed, and does a detector trained at one venue work at another.

**Open-source research (November 2026).** Not yet tested on our video. Speeds on a Mac are estimates from published GPU and CPU numbers.

| Part | Best candidates | Notes |
|---|---|---|
| Robot detector (trained on our labels) | [RF-DETR](https://github.com/roboflow/rf-detr) S or M (Apache-2.0); [YOLO26](https://docs.ultralytics.com/models/yolo26/) s (AGPL) | FRC teams report usable detectors from about 300 frames, or less than one match of frames. Our Claude labels are enough to start. YOLO trains on a Mac with `device="mps"`. Suggested classes: `robot_red`, `robot_blue`, `bumper_number`. |
| Tracker | BoT-SORT with an appearance model, from [BoxMOT](https://github.com/mikel-brostrom/boxmot) (AGPL) | Hold it to exactly 3 tracks per color. FRC users report name swaps after occlusions of a few seconds; fixes include bumper-color histograms, speed prediction and "fingerprinting" each robot during auto. |
| Bumper digits | [PARSeq](https://github.com/baudm/parseq) (Apache-2.0, fine-tune on our bumper crops); Apple Vision as a no-training baseline | Key trick: the schedule gives the 6 teams, and bumper color splits them 3 per alliance, so the reader only chooses 1 of 3. Vote across a whole track, not per frame. No public FRC project reads bumper numbers reliably without this. |
| Local AI model for hard moments | [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B) 4B or 9B (Apache-2.0); [Moondream 3.1](https://huggingface.co/moondream/moondream3.1-9B-A2B) (detect and point skills); [Molmo2](https://huggingface.co/allenai/Molmo2-8B) (points and video tracks; research and education use) | Run on a Mac with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Ask only "Which of A, B or C is this robot?" with a few crops. |
| Zero-shot finders (no training) | SAM 3 / 3.1 (text prompt, tracks in video; needs a CUDA GPU), YOLOE-26, Florence-2 | Not tested on "robot". Probably weak against the field structures; useful to pre-label frames. |

Estimated time per match on an M-series Mac: detection 20–35 s, tracking under 5 s, digit reading 15–30 s, local AI for hard moments 0.5–3 min. That is about 1–4 minutes and $0 per match, so hundreds of matches fit in one overnight run.

Prior FRC projects to learn from: [NimbleValley/auto-scout](https://github.com/NimbleValley/auto-scout) (YOLOv8 on red/blue bumpers), the Iowa State project ([Chief Delphi](https://www.chiefdelphi.com/t/scouting-automation-project-iowa-state-university/520629); YOLOv8, BoT-SORT, EasyOCR), [Computer Vision Scouting](https://www.chiefdelphi.com/t/computer-vision-scouting/511642) and the [AI Scouting Retrospective](https://www.chiefdelphi.com/t/ai-scouting-retrospective/517438).

First tests:

1. Train RF-DETR-S and YOLO26s on the checked clip frames, and compare visible-robot detections on held-out frames. The first local baseline uses the smaller YOLO26n. Evaluate continuity through fully hidden frames separately at the tracking stage.
2. Measure bumper digit height in pixels at 1080p, and test Apple Vision and PARSeq with the 1-of-3 rule.
3. Run BoT-SORT on the full match, and count name swaps.
4. Only then, test whether a local AI model fixes the hard moments.

### Project: Try other AI models

Compare hosted models (OpenAI GPT, Google Gemini, others) and open models on the clip, with the same prompt and the answer key. Things to look for: price per image, built-in box output (Gemini returns boxes on a 0–1000 grid), whole-video input (one call per clip instead of one per frame), batch discounts, and how well each reads small bumper digits. Research results go in the results table and in this section.

Cheaper settings to test with Claude: one image per call instead of two, effort `low`, 2.5 frames per second, a smaller image (scale 0.6), prompt caching for the system prompt, and the Batch API (50% price; needs calls that do not depend on the previous answer).

**Hosted model research (November 2026).** Not yet tested on our video. Prices and benchmark numbers are from the linked pages as of 2026-11-01; check them again before a big run. Cost per match assumes our current design: 840 frames, two 1536x488 images per call, about 380 output tokens.

| Model | Price per million tokens (in / out) | Box output | Video input | Detection / OCR score* | Est. cost per match |
|---|---|---|---|---|---|
| OpenAI `gpt-6.1-terra` | $1 / $5 | From the prompt only | No | 84.7 / 81.9 | about $3.65 ($1.85 with Batch) |
| Google `gemini-4-argon` | $2 / $10, **doubles on 2026-12-01** | Native `box_2d` (0–1000 grid) | Yes | 82.5 / 85.7 | about $8 ($4 with Batch); about $16 after Dec 1 |
| Google `gemini-3.8-flash` | $0.75 / $3.75, **doubles on 2027-01-01** | Native `box_2d` | Yes | 68.1 / 72.8 | about $3 per frame-by-frame match; about $0.65 with video input |
| Alibaba Qwen3.8 Max | $2 / $6 | No official format | Yes | 76.7 / 74.0 | about $5.40 |
| Anthropic Claude Haiku 5.5 | $0.10 / $0.50 (Batch $0.05 / $0.25) | From the prompt | No | 65.8 / 61.2 | about $0.70 (measured on the clip) |
| Anthropic Claude Sonnet 5.5 (our baseline) | $2 / $10 | From the prompt | No | 74.3 / 66.3 | about $9 (measured) |
| Anthropic Claude Opus 5.5 (answer key) | $4 / $20 | From the prompt | No | 74.5 / 78.3 | about $18 (measured) |

\*From [Roboflow Vision Evals](https://playground.roboflow.com/evals) (2026-10-26): detection mAP@50 / OCR score on general images, not small bumper digits. Grok 4.7 (40.4) and Mistral Large 4 (59.3) scored too low for this task. Haiku-level scores failed in our own test.

Notes:

- Sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [OpenAI images](https://developers.openai.com/api/docs/guides/images-vision), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding), [Gemini video](https://ai.google.dev/gemini-api/docs/video-understanding), [Gemini media resolution](https://ai.google.dev/gemini-api/docs/media-resolution), [Alibaba pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing).
- Gemini image cost does not depend on image size. The `media_resolution` setting picks it: 280, 560, 1,120 (default) or 2,240 tokens. The highest setting may help read small digits.
- Gemini video input costs only about 70–280 tokens per frame, so one call per short clip could replace one call per frame. At that resolution it probably cannot read bumper numbers. Idea: Gemini Flash tracks positions in 10-second video chunks, and a high-resolution single-frame call reads team numbers every few seconds.
- Gemini rate limits for Batch are small on the first paid tier (about one match per batch queue).
- OpenAI says its models struggle with precise positions, but Terra still has the top detection score in the benchmark. Test it.

First tests: run the same 50–100 frames of the clip through `gpt-6.1-terra` (reasoning none and low) and `gemini-4-argon` (`media_resolution` high and ultra_high). Score names and boxes against the answer key. Then test the Gemini Flash video-chunk idea.

### Project: Who is who (robot identity)

- Sonnet swapped 868 and 2910 from about 11 s to 23 s. No model ever read 868's bumper.
- Ideas: describe each robot's look in the prompt; name robots from a gallery of checked crops; give more weight to clear bumper reads in `backward_pass.py`; ask a person to settle only the few unsure moments.

### Project: Where on the field (field map)

- The wide camera does not move, so one mapping works for the whole match. Pick at least 4 points with known field positions (hub corners, AprilTags, field corners) from the 2026 game manual drawings. Fit a homography (a fixed transform from image pixels to field meters, for example `cv2.findHomography`).
- Use each robot's bumper line (bottom middle of its box) as its floor position, not the box center.
- Define the areas as rectangles in field meters: blue zone, neutral zone, red zone, the 4 trenches, the 4 bumps, the 2 hubs, depots and outposts. Draw them back on the frame to check that they line up.
- Match clock: convert video time to match time with the clock column in `overlay_timeline.csv`, so every event has a time from the start of auto.

### Project: Points per robot

1. Make shooting windows for each robot: runs of frames with action `shooting`, joining gaps of 1–2 frames.
2. From `overlay_timeline.csv`, find each second where the alliance fuel count goes up. Move each increase back 1–3 s, because the hub counts fuel late.
3. Give each increase to the robot(s) of that alliance that were shooting just before it. Several shooters share it by shooting time. No shooter means "unassigned".
4. Scale each period so the robots' totals match the official TBA count for that period.
5. Add tower points from TBA (`autoTowerRobot1-3`, `endGameTowerRobot1-3`, in the order of the alliance's team list).
- Signs it works: the three robots add up to the official alliance fuel in every period, "unassigned" is under 10%, and a hand count of one robot for one shift agrees.

### Project: Everything Lovat records

| Lovat event or field | How it could be computed |
|---|---|
| `START_MATCH` + position | Robot's floor position in the first frame, nearest of the 5 start spots (left trench, left bump, hub, right bump, right trench). Confirm what "left" means in the Lovat app for each alliance. |
| `START/STOP_SCORING` + count | Shooting windows, with the fuel assigned to each window. Position `HUB`. |
| `START/STOP_FEEDING` | Runs of action `feeding`. The count is hard to measure. |
| `INTAKE` + position | Runs of action `intaking`. Position from the field area (neutral zone, depot, outpost). |
| `CROSS` + position | The robot's path enters and leaves a trench or bump area. |
| `START/STOP_CAMPING` | Robot moves less than about 0.3 m inside a trench for more than 3 s. |
| `START/STOP_DEFENDING` | Runs of action `defending`, or within about 1 m of an opponent on the opponent's side for more than 2 s. |
| `CLIMB` | First frame of action `climbing` near the tower. The level comes from TBA. |
| `DISRUPT` | Auto only: the robot takes or pushes fuel on the opponent's side. Ask Claude to judge these frames. |
| `robotRoles` | Share of match time in each action; cycling = intake then score, repeated; immobile = not moving for more than 30 s. |
| `fieldTraversal` | From the `CROSS` events: trench, bump, both or none. |
| `intakeType` | Ground, outpost, both or neither, from where it intakes. |
| `scoresWhileMoving` | The robot moves faster than about 0.5 m/s while shooting. |
| `beached` | Not moving for more than 2 s on a bump or fuel pile while trying to drive. Ask Claude to confirm. |
| `feederTypes` | Ask Claude, with crops of the feeding windows. |
| `autoClimb`, `endgameClimb` | From TBA. |
| `accuracy` | Made = fuel from the points project. Attempts = ask Claude to count fuel launched per shooting window. Low confidence. |

### Project: Ratings and notes (the team's whiteboard fields)

- Fields: driver ability 1–10 + notes, accuracy 1–10, defensive ability 1–10 + notes, defendability (plays well while defended) 1–10 + notes, fuel scoring ability 1–10, short auto description, pick preference (1st pick, 2nd pick, do not pick), general notes.
- One Claude call per robot per match, with the robot's timeline and numbers and a contact sheet of crops. Put a scoring guide in the prompt with an example of a 2, a 5 and a 9, so a rating means the same thing every match. Every rating gets one line of evidence.
- Defensive ability: compare the defended robot's scoring rate with and without this robot close. Defendability: this robot's own scoring rate with an opponent within about 1 m, compared with when it is free.
- Compare with ratings from 8033's human scouts in Lovat at an event where both exist.

### Project: Send reports to Lovat

- One JSON per robot in Lovat's `scoutReportSchema` format. Convert 1–10 ratings to Lovat's scales (driver ability 1–5, defense 0–4, accuracy bands 0–5). Keep defendability, pick preference and the 1–10 ratings as extra fields until Lovat adds them.
- Send with `POST /v1/manager/scoutreports` as a scouter named "Video Scout". First ask the Lovat team about that route: a research pass found it may have no login check.

### Project: Hundreds of matches

- Get match lists and video links from The Blue Alliance.
- Find the match start automatically (the overlay clock shows 0:20 and the robots are still), and skip frames where the overlay is missing (replays, cutaways).
- Test at several venues: lighting and camera angles change.
- Track the cost per match for every run.

### Project: Knowing when it is right

- Make more answer keys: check more clips and matches by hand, at different venues and in crowded moments.
- Compare video-scout reports with human scout reports in Lovat for the same matches.

## Path-animation release packaging

`release/run.py` creates an isolated environment, installs `release/requirements.txt`,
selects GPU/CPU automatically, and launches `auto_paths.py` with the bundled model.
The ZIP includes the Python/HTML sources, demo calibration, v1 weights, AGPL license,
and a model checksum manifest. It excludes credentials, videos and environments.
Mac runtime tested; Windows/Linux launchers have not been platform-tested.

## Landmark recognition pilot

### Human floor-landmark labeling

`label_landmarks.py` makes a self-contained offline browser page for creating the
ground-truth correspondences needed to calibrate a fixed broadcast camera. It reads a
local video, writes sampled JPEG frames and `label_landmarks.html` to a **new** output
directory, and makes no network or Claude request:

```bash
.venv/bin/python label_landmarks.py path/to/match.mp4 --out runs/casac_main_wide_labels --start 5.5 --duration 20 --every 2
```

Open the generated HTML page, select each sampled frame, and click four or more
points that are fixed to the carpet or the field boundary. Give each point its
top-down percentage coordinate (blue-to-red x and far-to-near y) from the game manual.
Do not label robot positions, hub tops, or a landmark hidden by a robot. Download
`landmark_labels.json` and keep it next to that tournament-camera preset. A stable
wide broadcast view normally needs one calibration; cuts, replays, and zoomed views
need separate labels or must use the rough fallback.

The labeler includes field corners, center-line boundary intersections, and boundary
midpoints. Mark every one that is visible (six to eight is preferable), then use
**Download this frame's camera.json**. `auto_paths.py` robustly fits the homography
from four or more correspondences and records the total/inlier landmark counts. This
reduces the effect of one slightly misplaced click; always inspect its calibration
preview before trusting a path.

The JSON contains all sampled image points plus their known field coordinates. It is
the human-reviewed training/validation source for a future keypoint model, and it can
also be used to check one camera homography before drawing robot paths. The tool does
not yet train or automatically apply a keypoint model; that deliberately remains a
separate measured step.

` .venv/bin/python -u train_landmarks.py` prepares `datasets/landmarks_v1/` and
launches 30 epochs in `runs/landmarks_v1/`. It reuses the six rough hub/bump polygons
in `start/prematch_labels.json` as bounding boxes. Fifty frames at 5.5–25.1 seconds
train the detector; twenty at 35.5–45 seconds provide same-camera validation.
Temporal separation does not demonstrate generalization to another camera or venue.
The labels are approximate, propagated with a fixed-camera assumption, and can include
occlusions. Inspect `datasets/landmarks_v1/labels_preview.jpg` before treating them as
reliable annotations. No Claude requests are made. Progress is in
`runs/landmark_jobs/v1.log`; the training wrapper prints a heartbeat every 25 seconds.

This is structure recognition, not a floor-point alignment model. Hub bounding-box
centers must not be used as coplanar floor landmarks. Accurate floor correspondences,
separate-match testing and rejection of uncertain fits are still required before
connecting predictions to automatic calibration. `compare_runs.py` requires robot
identities and cannot score this six-class landmark task; no robot-answer-key score
or field-position accuracy is claimed. Training results are pending.
