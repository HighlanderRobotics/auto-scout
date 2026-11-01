"""Turn overlay_timeline.csv (OCR of the broadcast overlay) into fuel per alliance per REBUILT period.

Periods follow the 2026 manual (Table 6-2). Fuel still counts for 3 s after a hub deactivates,
so a score change at video second t belongs to the period whose clock showed 3 s earlier.
"""
import csv, re, sys

LAG_S = 3
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "overlay_timeline.csv")))

def secs(clock):
    return int(clock[0]) * 60 + int(clock[2:]) if re.fullmatch(r"\d:\d\d", clock or "") else None

def period(t, teleop):
    if not teleop: return "Auto"
    if t > 130: return "Transition"
    if t > 105: return "Shift 1"
    if t > 80:  return "Shift 2"
    if t > 55:  return "Shift 3"
    if t > 30:  return "Shift 4"
    return "End game"

# Label each video second with its period, tracking the auto -> teleop clock jump.
labels, teleop, prev = [], False, None
for r in rows:
    t = secs(r["clock"])
    if t is not None:
        if prev is not None and t > prev + 30: teleop = True
        prev = t
    labels.append(period(t, teleop) if t is not None else None)

blue = red = 0
totals = {}
for i, r in enumerate(rows):
    b = int(r["blue_score"]) if r["blue_score"].isdigit() else None
    d = int(r["red_score"]) if r["red_score"].isdigit() else None
    # Scores only go up; reject OCR values that go down or jump implausibly.
    if b is not None and blue <= b <= blue + 40: blue = b
    if d is not None and red <= d <= red + 40: red = d
    j = i - LAG_S
    while j >= 0 and labels[j] is None: j -= 1
    lab = labels[j] if j >= 0 else None
    if lab: totals[lab] = (blue, red)

pb = pr = 0
print("period,blue,red")
for p in ["Auto", "Transition", "Shift 1", "Shift 2", "Shift 3", "Shift 4", "End game"]:
    if p in totals:
        b, d = totals[p]
        print(f"{p},{b - pb},{d - pr}")
        pb, pr = b, d
print(f"Overlay final,{blue},{red}")
