"""Backward pass: fix robot names using what later frames saw.

The forward pass names robots from past frames only, so an early mistake stays until a bumper is read.
Here each alliance's boxes are linked frame to frame by movement into tracks. Every bumper read
anywhere on a track votes for that track's name, so a read at second 20 also names the robot at second 6.
Then in every frame each alliance's three tracks get the three team names that best fit the votes.

Usage: .venv/bin/python backward_pass.py track/llm track/final [MAX_STEP_PX]
Use a bigger MAX_STEP_PX (default 220) when frames are further apart, for example 440 at 2.5 frames per second.
"""
import itertools, json, sys
from collections import defaultdict
from pathlib import Path

TEAMS = {"868": "blue", "2910": "blue", "2046": "blue", "4065": "red", "4414": "red", "1323": "red"}
MAX_STEP = 220          # px a robot can move between frames 0.2 s apart before the track is split
WEIGHT = {"bumper_number": 3.0, "previous_frames": 0.5, "elimination": 0.2, "guess": 0.1}


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def dist(a, b):
    (ax, ay), (bx, by) = center(a), center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def main():
    global MAX_STEP
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3:
        MAX_STEP = float(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("f_*.json"))
    frames = [json.load(open(f)) for f in files]

    # 1. Link boxes into tracks, per alliance, by the matching with the smallest total movement.
    track_of = {}           # (frame, robot index) -> track id
    next_id = 0
    for alliance in ("blue", "red"):
        prev = None
        for fi, fr in enumerate(frames):
            cur = [i for i, r in enumerate(fr["robots"]) if TEAMS[r["team"]] == alliance]
            if prev is None or len(prev) != len(cur):
                for i in cur:
                    track_of[(fi, i)] = next_id
                    next_id += 1
            else:
                pbox = [frames[fi - 1]["robots"][j]["box"] for j in prev]
                best = min(itertools.permutations(cur),
                           key=lambda p: sum(dist(pbox[k], fr["robots"][p[k]]["box"]) for k in range(len(p))))
                for k, i in enumerate(best):
                    if dist(pbox[k], fr["robots"][i]["box"]) <= MAX_STEP:
                        track_of[(fi, i)] = track_of[(fi - 1, prev[k])]
                    else:
                        track_of[(fi, i)] = next_id
                        next_id += 1
            prev = cur

    # 2. Votes: every observation on a track votes for a team name, bumper reads count most.
    votes = defaultdict(lambda: defaultdict(float))
    for (fi, i), tid in track_of.items():
        r = frames[fi]["robots"][i]
        name = r["bumper_text"].strip() if r["named_by"] == "bumper_number" and r["bumper_text"].strip() in TEAMS else r["team"]
        votes[tid][name] += WEIGHT[r["named_by"]] * (1.0 if r["visible"] else 0.3)

    # 3. In each frame, give each alliance's tracks the team names that best fit the votes.
    changed = 0
    for fi, fr in enumerate(frames):
        out = json.loads(json.dumps(fr))
        for alliance in ("blue", "red"):
            idx = [i for i, r in enumerate(fr["robots"]) if TEAMS[r["team"]] == alliance]
            names = [t for t, a in TEAMS.items() if a == alliance]
            best = max(itertools.permutations(names, len(idx)),
                       key=lambda p: sum(votes[track_of[(fi, i)]][p[k]] for k, i in enumerate(idx)))
            for k, i in enumerate(idx):
                tid = track_of[(fi, i)]
                if out["robots"][i]["team"] != best[k]:
                    changed += 1
                out["robots"][i]["forward_team"] = fr["robots"][i]["team"]
                out["robots"][i]["team"] = best[k]
                out["robots"][i]["track"] = tid
                total = sum(votes[tid].values()) or 1
                out["robots"][i]["confidence"] = round(votes[tid][best[k]] / total, 2)
        json.dump(out, open(dst / files[fi].name, "w"), indent=1)
    print(f"{len(frames)} frames, {next_id} tracks, {changed} robot names changed by the backward pass")


if __name__ == "__main__":
    main()
