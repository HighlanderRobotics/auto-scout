# Auto-Scout: technical details

This is the detailed readme: how the pieces work, what we measured, the game and Lovat facts the code depends on, and technical notes for each research project. Start with [README.md](README.md).

## Goal

Scout a 2026 REBUILT match automatically from the webcast video. The output must be **per robot**: one report per robot per match, in the format of the team's scouting app, Lovat (https://github.com/HighlanderRobotics/lovat). Alliance totals alone are not enough.

Test match: Einstein Final 1, 2026 Championship, https://www.youtube.com/watch?v=EjF9we707DA. Red alliance: 4065, 4414, 1323. Blue alliance: 868, 2910, 2046. Final score: Blue 518, Red 456. Official per-period data: https://www.thebluealliance.com/match/2026cmptx_f1m1

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

Sonnet's and Haiku 5.5's errors were all the same 868/2910 swap from about 11 s to 23 s. Adding one line per robot about how it looks (`start/robot_looks.json`) cut Haiku 5.5's errors to 11 frames (21.1–23.1 s, 2910 called 868). No model ever read 868's bumper. Cost is about 3,400 input tokens and 380 output tokens per frame, mostly the two images. Ideas to try: send only the current frame; describe each robot's look in the prompt; Opus at effort `low`.

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
| | Haiku 5.5 | two images | 94% | 96% | $0.0008 | about $0.70 |
| | Haiku 4.5 | two images | not usable | 4% | $0.005 | — |

Target for scouting hundreds of matches: at least 98% correct names and 95% boxes in the right place, below $1 per match.

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

1. Train RF-DETR-S and YOLO26s on the checked clip frames, and compare how many robots each finds on held-out frames, including hidden ones.
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
