# Team 334 approach: lessons for Auto-Scout

Reviewed 2026-11-15. Source: [derek-pohl/ai-scouting](https://github.com/derek-pohl/ai-scouting), revision `97184951f9646f047cb5587f4d0aebac8413a729`.
The repository URL and requested branch name are not independent verification of
the project's team affiliation. This is a source review, not a benchmark or a port.

## Branch starting point

`Team-334-Aproch` starts at `combind-paths`, commit `af79f22`.
After fetching origin, this was the most complete implementation: it contains the
earlier local detector, team naming, calibration, fragment stitching, visual memory,
Glendale/SF camera adaptations, smoothing, and team boxes on the source video.
`origin/local-tracking+2d-animation` at `4979411` merged an earlier `combind-paths`
revision; comparing the trees shows it lacks the newer memory and viewer work.
`main` is still the earlier paid frame-tracking experiment.

"Best" here means the strongest feature base, not a newly measured accuracy winner.
Keep the preferred v1 detector as the global default and camera-specific models in
their profiles. Existing Einstein checks report 229/229 correct matched names but
only 39.4% named-reference coverage. Those numbers describe a limited clip, not
complete six-robot scouting. See [DETAILS.md](../DETAILS.md#local-visual-memory-2026-11-15).

## What to learn

| Priority | Upstream implementation | Application to Auto-Scout |
|---|---|---|
| 1 | Manual robot tracks and shooter marks, alongside automatic processing | Add correction controls to our synchronized viewer. Let a reviewer name a fragment, split an identity swap, and mark shooting intervals. Save edits as separate evidence with timestamps and provenance. This gives us usable reports while automatic naming remains incomplete. |
| 2 | Score OCR deltas constrain attributed makes; unmatched deltas are counted | Connect our existing overlay timeline to shooting intervals. Preserve unassigned fuel and attribution confidence. Matching alliance totals alone does not verify the robot split. |
| 3 | Center/blue/red feeds are cropped and processed separately | Add explicit composite-layout profiles, camera-specific calibration, and synchronized timestamps before merging evidence. Side views can provide evidence when the main view is obstructed. |
| 4 | Calibration points and exclusion polygons are saved in normalized coordinates | Extend our camera JSON/landmark workflow with reusable no-scan polygons and a preview. Key profiles by event, camera/layout, and source geometry; event name alone cannot identify every camera setup. |
| 5 | Cropped robot images go to an optional local vision endpoint when tracking needs labels | Experiment with local crop-based identity proposals after OCR fails. Keep our repeated-evidence, roster and conflict gates. Measure coverage, wrong names, latency and memory use before adopting it. |
| 6 | Fuel detection has a lightweight HSV path and an optional SAM path | Prototype fuel only around reviewed shooting intervals. Start with a small labeled sample before choosing a detector. Benchmark on our Mac before selecting a heavier backend. |

Code references at the reviewed revision:

- [Manual track parsing and shooter annotations](https://github.com/derek-pohl/ai-scouting/blob/97184951f9646f047cb5587f4d0aebac8413a729/app.py#L8331-L8558).
- [Score attribution and unmatched budget](https://github.com/derek-pohl/ai-scouting/blob/97184951f9646f047cb5587f4d0aebac8413a729/app.py#L2557-L2963).
- [Composite crop layout](https://github.com/derek-pohl/ai-scouting/blob/97184951f9646f047cb5587f4d0aebac8413a729/app.py#L271-L312), [calibration persistence](https://github.com/derek-pohl/ai-scouting/blob/97184951f9646f047cb5587f4d0aebac8413a729/app.py#L841-L1054).
- [Local crop labeling](https://github.com/derek-pohl/ai-scouting/blob/97184951f9646f047cb5587f4d0aebac8413a729/app.py#L1174-L1472), [fuel detection](https://github.com/derek-pohl/ai-scouting/blob/97184951f9646f047cb5587f4d0aebac8413a729/app.py#L6500-L6998).

## What needs caution

The upstream robot detector uses bumper-color masks with field/person exclusions
(`detect_robots_by_bumper_color`); the YOLO segmentation weight is used for people.
It is not evidence that its robot detection outperforms our trained YOLO model.

Its identity tracker uses nearest-center matching. Confidence increments when a
new named label arrives, even when that label differs from the previous one. That
is weaker identity evidence than repeated agreement. Keep our ambiguity checks.

Its OCR-only allocation can divide a delta equally between marked shooters. That
is a useful estimate, but two simultaneous shooters need not contribute equally.
It also raises attempt totals to at least attributed makes; attempts then are not
independently observed and cannot establish shooting accuracy by themselves.

`persist_recent_robot_bboxes` carries the last box forward without an explicit
estimated marker in that returned box. Preserve our observed/estimated distinction
when adding persistence or side-camera evidence. A hidden-zone inference must not
become a measured field coordinate or count as a detected robot.

The crop rectangles assume one broadcast composition, and calibration reuse does
not solve camera cuts or zooms. Multi-camera shot deduplication groups same-robot
events by a time window; test rapid real shots so they are not merged accidentally.

The reviewed root has no test suite, reported answer-key benchmark, or explicit
license file. Do not infer accuracy from the feature list or copy source/weights
until reuse terms are established. The upstream README describes a CUDA-oriented
SAM workflow; it has not been validated on our Apple Silicon setup.

## Recommended first experiment

Build manual fragment naming and shooting-window annotation on our existing viewer,
then replay saved paths locally into a new run folder. Keep raw predictions and
reviewer edits separate. Score named paths with `evaluate_team_paths.py` and its
`compare_runs.py` export against untouched `track/final/`; report coverage and wrong
names together, and distinguish human-assisted results from automatic results.

Next, use the saved overlay timeline to allocate fuel to those reviewed windows.
Show a table of robot allocations, human-player contributions where relevant, and
unassigned fuel against alliance fuel counters. Check one robot/shift by hand and
measure attribution error, not just conservation of the total. Preserve unknowns
when launch timing or identity is unclear.

Only after that pilot should we add multi-camera fusion or heavier fuel models.
This review made no Claude requests, trained no models, changed no runtime behavior,
and produced no new run score. Any paid experiment still requires a cost estimate
and explicit approval under AGENTS.md.
