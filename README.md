# Auto-Scout

Scout a REBUILT match from the webcast video, with AI, and produce one report per robot that [Lovat](https://github.com/HighlanderRobotics/lovat) can use.

This repo is built to be worked on with an AI coding assistant (Claude Code). You do not need to learn every script first. Open the folder in Claude Code, and ask it to do things. The technical details are in [DETAILS.md](DETAILS.md). Claude reads it too.

## Get started

1. Open this folder in Claude Code (desktop app, or `claude` in a terminal).
2. Copy `.env.example` to `.env`, and put the team's Claude API key in it. Ask a mentor for the key. **Never commit `.env`.**
3. Ask Claude: *"Set up this project and download the match video."*
4. Then ask: *"Show me the labeled tracking video for the 30-second clip."*

## Where we are

We tested on [Einstein Final 1 of the 2026 Championship](https://www.youtube.com/watch?v=EjF9we707DA).

- **Reading the score overlay works.** Code reads the score, clock and shift counter every second. It matched the official totals.
- **Claude can track all six robots.** For each video frame, Claude gets the frame and the last known positions, and returns a box and a team number for every robot. A second pass then fixes names with bumper numbers read later in the match. On a 30-second clip, a mentor checked the Opus run by hand: every name was correct.
- **Cost per match depends on the model.** One Claude call per frame is about 840 calls per match. Opus costs about $18 per match. Sonnet costs about $9 and got 95% of names correct. The new Haiku 5.5 got 94% correct for about $0.70 per match. The cheaper runs swapped 868 and 2910 for about 12 seconds. Telling Haiku 5.5 what each robot looks like fixed most of that: 98% of names correct for about $0.75 per match, which meets our target.
- **The first local detector is trained.** YOLO26n trained on 81 images and matched 75 of 84 robot boxes on 14 held-out images at IoU 0.5, with 18 extra detections. This is a small same-match pilot; team identity and testing at other venues still need work. No new Claude calls were needed. Results are in [DETAILS.md](DETAILS.md#first-trained-baseline-local_robot_v1-2026-11-01).
- **A second local experiment reused the Haiku labels.** Training on 141 confident image regions matched 70 of 84 test boxes, with 12 extras at IoU 0.5. It reduced extra detections but missed more robots, so the first model remains the preferred baseline. This round added no API charges; see [the measured result](DETAILS.md#trained-result-local_robot_v2_regions-2026-11-02).

## Research projects

Saved Haiku teacher labels are available in
[annotations/haiku_training_v1](annotations/haiku_training_v1/README.md): 4,156
annotation records with source-video metadata, frame coordinates, and provenance.
These include unreviewed proposals, so use the documented review/filtering steps
before training. Videos and credentials are excluded.

For team-number paths on macOS, add `--teams 868,2910,2046,4065,4414,1323`
to the path command below (use the actual roster for another match). Local Apple
Vision reads robot crops and merges track fragments with repeated supporting bumper
reads. Open `teams/animation.html`; its debug link retains anonymous paths.
Unresolved robots remain in debug. Short dashed joins are estimates; long hidden
intervals remain unknown. No Claude calls are made. See DETAILS.md for limits.
On `combind-paths`, compatible fragments reconnect in both directions into one path
per resolved team. Short gaps up to two seconds are filled with explicitly estimated
positions, shown as dashed lines and hollow markers. Crowded or ambiguous joins stay
unresolved. Color evidence is collected automatically, so normal `--teams` runs can
extend an OCR-confirmed identity without a separate alliance roster.
Add `--blue-teams 868,2910,2046 --red-teams 4065,4414,1323` to enable extra free,
conservative evidence: bumper-color filtering, unique two-digit partial reads, close
same-color fragment joins, and a leftover-team rule.

An experimental Haiku appearance coach (`coach_identity.py`) can name track fragments
from multiple crops and the saved robot descriptions. Paid execution requires
`--execute` and explicit budget approval. It uses a shared $2 ledger, and exports
evidence for `team_paths.py`. `train_identity_gallery.py` trains a small offline
appearance gallery from accepted names. This experiment has four known teams and
does not yet replace the app's naming system. See DETAILS.md for measured coverage.

To turn the documented match's YouTube link into a local autonomous-path animation:

```bash
.venv/bin/pip install -r requirements-paths.txt
.venv/bin/python auto_paths.py 'https://www.youtube.com/watch?v=EjF9we707DA' --out runs/my_auto_paths
```

Open `runs/my_auto_paths/animation.html` for the 2D pane and synchronized tracking
video. The first detector checkpoint must exist locally. Other cameras require
an autonomous start time and calibration; see [autonomous path animation](DETAILS.md#autonomous-path-animation).
This is a prototype with anonymous, sometimes fragmented track IDs and an approximate
demo field mapping. No Claude calls are made.

The repaired San Francisco Quals 7 demo uses the original 1080p broadcast pixels,
a tight field crop, and a detector adapted to that camera. Its camera profile selects
the locally trained checkpoint automatically:

```bash
.venv/bin/python auto_paths.py datasets/distant_sf_v1/native_source.mp4 --calibration configs/san_francisco_qm7_native_camera.json --out runs/my_sf_distant_demo
```

The repaired demo is in `runs/san_francisco_qm7_distant_fixed/animation.html`.
See `results/distant_sf_v1.json` for the far-side detection checks and limits.

`configs/landmark_match_pool_v1.json` records four TBA-linked 2026 finals from
Sacramento, San Diego, Wisconsin, and Pacific Northwest. The videos are local,
unlabeled evaluation material; they are not automatically included in training.

The 2D pane includes schematic blue/red hubs (baskets) and four ramps, without fuel.
To redraw an existing run without downloading or running detection again:

```bash
.venv/bin/python auto_paths.py --render-paths runs/my_auto_paths/paths.json --out runs/my_auto_paths_redrawn
```

Local detector data preparation is available now, without Claude calls:

```bash
.venv/bin/python prepare_dataset.py track/clip_frames track/final datasets/checked_clip_v1
```

Restore the video frames first using [DETAILS.md](DETAILS.md#setup). The export includes
training labels, separate chronological training/validation/test blocks, and labeled
preview sheets. See
[local dataset preparation](DETAILS.md#local-dataset-preparation) for checks and limitations.

To train a first local robot detector and score its held-out test images:

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python train_detector.py --out runs/local_robot_v1
.venv/bin/python evaluate_detector.py runs/local_robot_v1/weights/best.pt datasets/checked_clip_v1 runs/local_robot_v1_test
```

Training runs locally on the Mac GPU, downloads public starting weights, and makes
no Claude requests. The detector finds robots; it does not assign team numbers.
See [local detector training](DETAILS.md#local-detector-training) for settings and results.

The Haiku teacher job can label three TBA-linked matches, filter the labels, train,
and evaluate automatically. It is Haiku 5.5 only, with an approved **$4 maximum**
and an estimated labeling cost around $2. See
[the teacher pipeline](DETAILS.md#haiku-teacher-pipeline) before running paid work.

Saved teacher labels can also be reused without API calls with
`prepare_confident_regions.py`. It keeps the 50% confidence threshold and crops
away uncertain estimates rather than requiring all six robots in every training
image. See [confident-region training](DETAILS.md#confident-region-training) for
the preparation command, review records, and limitations.

This is not a list of steps. Each project is a way the system needs to improve. Pick one, try ideas with Claude, and measure the result. The technical details for each project are in [DETAILS.md](DETAILS.md#research-projects-technical-details).

### 1. Teach a local model with Claude

This is the most important project, because it decides the cost of everything else. Claude is accurate but costs money per frame. A small, special-purpose robot detector runs free on a laptop once it is trained.

The idea: use Claude as the teacher. Claude labels robots in a few thousand frames. We train a small detector on those labels. The detector and a tracker follow the robots in every frame for free. Claude is called only for the hard moments, for example two robots that pass close or a bumper number that must be read.

- How we know it works: the local model matches the hand-checked answer key, at a cost below $1 per match.
- Open questions: how many labeled frames it needs, and whether it still works at a venue it has not seen.

### 2. Try other AI models

Claude is not the only AI that can look at images. OpenAI's GPT models, Google's Gemini models and free open models are all options. Some can find objects and return boxes directly. Some can take a whole video clip in one call instead of one call per frame. First research (in DETAILS.md) says OpenAI's GPT-6.1 Terra and Google's Gemini 4 Argon look better and cheaper than Claude Sonnet for finding objects and reading text, and Gemini can take a whole video clip in one call. None of them has been tested on our video yet. Compare every model on the same clip, against the same answer key.

### 3. Who is who

The hardest part is keeping each team number on the right robot. Sonnet mixed up 868 and 2910 for about 12 seconds, and no model ever read 868's bumper. Ideas: describe each robot's look to the AI, recognize robots by their look instead of their bumper, and ask a person to settle only the few unsure moments.

### 4. Where on the field

The tracker knows where a robot is in the image, not on the field. A map from image pixels to field positions (using the AprilTags and field drawings) would tell us start positions, trench and bump crossings, and when a robot is camping or defending.

### 5. Points per robot

The official data gives fuel per alliance, not per robot. The idea: when the alliance score goes up, give the points to the robot that was shooting just before. The three robots must add up to the official total.

### 6. Everything Lovat records

Lovat records timed events (scoring, feeding, intake, crossings, camping, defending, climbing) and a post-match form (roles, intake type, accuracy, and more). Each one needs a way to compute it from robot positions, actions and official data. Some need Claude to judge a few frames.

### 7. Ratings and notes

Our whiteboard fields: driver ability, accuracy, defensive ability, defendability (how well it plays while defended), fuel scoring ability, auto description, pick preference and notes. The idea: one AI call per robot per match, with the robot's numbers and pictures, and a scoring guide so a 7 means the same thing every match. Compare with our human scouts' ratings.

### 8. Hundreds of matches

Find each match in a video automatically, skip replays and cutaways, handle different venues and lighting, and keep track of the cost per match.

### 9. Knowing when it is right

Every project needs a way to measure it. Make more hand-checked answer keys, and compare video-scout reports with human-scout reports in Lovat for the same matches.

## Rules

- Every AI run spends real money on the team's API account. Ask Claude what a run will cost before you start it.
- Never commit `.env` or the match video.
- Make a branch for your work, and open a pull request.
- Check what the AI made. Watch the videos, and look at the numbers. You are the scout. The AI is the tool.

## Downloadable path-animation release

The GitHub preview release bundles the local model and a first-run Python launcher.
See `release/README.txt`. Python 3.11+ and internet for initial dependency setup are
required; no Claude key is needed. Other cameras still require calibration.

Landmark recognition pilot: `train_landmarks.py` prepares 50 training and 20
validation frames from the existing Einstein video and trains six hub/ramp classes.
This uses rough same-camera labels; it is not automatic camera calibration yet.

To help make accurate floor-landmark labels for a camera setup, use the offline
clicking tool. It samples frames from a local video and creates a browser page; no
video is uploaded and no Claude call is made. Click four or more fixed carpet/field
points, enter their known field coordinates, and download the labels. The tool now
offers eight useful boundary/center-line landmarks and can export a `camera.json`
directly from one marked frame; use 6–8 visible landmarks when possible.

```bash
.venv/bin/python label_landmarks.py path/to/match.mp4 --out runs/my_landmark_labels --start 5.5 --duration 20 --every 2
```

Open `runs/my_landmark_labels/label_landmarks.html`. The exported
`landmark_labels.json` is reusable labeling data for that camera setup.
