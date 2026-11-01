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

## Research projects

Local detector data preparation is available now, without Claude calls:

```bash
.venv/bin/python prepare_dataset.py track/clip_frames track/final datasets/checked_clip_v1
```

Restore the video frames first using [DETAILS.md](DETAILS.md#setup). The export includes
training labels, separate chronological training/validation/test blocks, and labeled
preview sheets. This prepares data; it does not train a detector yet. See
[local dataset preparation](DETAILS.md#local-dataset-preparation) for checks and limitations.

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
