"""Compare tracking runs from different models against a reference run (Opus).

Usage: .venv/bin/python compare_runs.py track/final opus track/sonnet_final sonnet track/haiku_final haiku
The first folder is the reference. For each frame, each robot box is matched to the reference box of the
same alliance with the most overlap; the names then agree or not. Also prints bumper reads, tokens and cost.
"""
import glob, itertools, json, sys
from pathlib import Path

TEAMS = {"868": "blue", "2910": "blue", "2046": "blue", "4065": "red", "4414": "red", "1323": "red"}
PRICE = {"opus": (4, 20), "sonnet": (2, 10), "haiku": (1, 5), "haiku55": (0.10, 0.50)}   # $ per million input / output tokens


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def load(folder):
    return {Path(f).name: json.load(open(f)) for f in sorted(glob.glob(f"{folder}/f_*.json"))}


def usage(folder):
    # usage is written by the forward pass; the backward pass copies it
    tin = tout = 0
    for d in load(folder).values():
        tin += d.get("usage", {}).get("in", 0); tout += d.get("usage", {}).get("out", 0)
    return tin, tout


def main():
    args = sys.argv[1:]
    runs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    ref_dir, ref_name = runs[0]
    ref = load(ref_dir)
    print(f"{'run':8} {'frames':>6} {'same name as ' + ref_name:>18} {'boxes overlap':>14} {'bumper reads':>13} {'tokens in/out':>16} {'cost $':>7}")
    for folder, name in runs:
        run = load(folder)
        agree = total = overlap = 0
        reads = 0
        for fname, d in run.items():
            reads += sum(r["named_by"] == "bumper_number" for r in d["robots"])
            if fname not in ref:
                continue
            for alliance in ("blue", "red"):
                mine = [r for r in d["robots"] if TEAMS[r["team"]] == alliance and r["visible"]]
                theirs = [r for r in ref[fname]["robots"] if TEAMS[r["team"]] == alliance and r["visible"]]
                if not mine or not theirs:
                    continue
                k = min(len(mine), len(theirs))
                best = max(itertools.permutations(theirs, k), key=lambda p: sum(iou(m["box"], t["box"]) for m, t in zip(mine, p)))
                for m, t in zip(mine, best):
                    total += 1
                    if iou(m["box"], t["box"]) >= 0.3:
                        overlap += 1
                        agree += m["team"] == t["team"]
        tin, tout = usage(folder)
        pin, pout = PRICE[name]
        cost = tin * pin / 1e6 + tout * pout / 1e6
        pct = f"{100 * agree / overlap:.0f}%" if overlap else "-"
        print(f"{name:8} {len(run):6} {pct:>18} {f'{100 * overlap / max(total, 1):.0f}%':>14} {reads:13} {f'{tin}/{tout}':>16} {cost:7.2f}")


if __name__ == "__main__":
    main()
