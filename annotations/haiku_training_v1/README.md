# Saved Haiku training annotations

This folder publishes **4,156 saved annotation records** from the local Haiku
experiments. No new Claude requests were made for this export. The 12 JSONL files
total approximately 5.5 MB; each line is a standalone JSON object.

These are **teacher proposals, not a human-verified answer key**. Low-confidence,
hidden, and unusable examples are retained so the training history is inspectable.
Not every exported record was accepted into detector training. Do not combine
proposed boxes into a training dataset without the filtering and review described
in the repository's DETAILS.md.

## Contents

- `teacher_v2/`: 2,245 frame records from three matches, sampled at 5 fps.
- `teacher_multivenue_v1/`: 1,793 frame records from four championship matches,
  sampled at 3 fps. Sacramento appears in both runs; these are separate teacher
  versions, not unique extra video frames.
- `audit/`: 40 Einstein teacher-tracking audit records; distinct from `track/final/`.
- `distant_sf_v1/` and `distant_sf_native_labels_v1/`: 14 SF frame proposals across
  low/native resolution. These unreviewed proposals were **not** directly used for
  training. Their reviewed replacements are in `configs/sf_distant_reviewed_boxes.json`.
- `identity_coaching/`: 64 appearance-classification proposals across two rounds,
  including null/uncertain responses. These describe anonymous track crop groups,
  not individual-frame bounding boxes. Track IDs refer to the Einstein demo's local
  `runs/auto_paths_youtube_demo/paths.json` output, which is not part of this export.

`manifest.json` supplies the public video IDs, match URLs where available,
train/validation split, timestamps/cadence, image crop, hashes, per-file counts and
annotation status. Training-run labels contain source image hashes and prompt
fingerprints; appearance proposals did not record per-crop hashes and therefore
cannot be verified as precisely.

## Coordinates and use

`box` is `[left, top, right, bottom]` in **cropped image pixels**, with dimensions
given by `image_size`. Add the bundle's crop offset to obtain original video pixel
coordinates. For example `[0,110,1920,720]` means add 110 to each y coordinate.
`video_time_s` is computed from the original extraction contract and frame index.
SF filenames instead use seconds within their downloaded local video sections;
see the timing caveat in the manifest.

`visible: false` means a robot was hidden: its box is an old estimate, not an
observed detection. `named_by` and confidence preserve the teacher's stated evidence;
they are not independent verification. Identity-coaching confidence is also an
unvalidated model self-assessment.

Example offline reading:

```python
import json
from pathlib import Path

path = Path("annotations/haiku_training_v1/teacher_multivenue_v1/2026wicmp_f1m1.jsonl")
for line in path.open():
    frame = json.loads(line)
    visible = [r for r in frame["robots"] if r.get("visible", True)]
```

Images, videos, model weights, raw response transcripts, billing ledgers, API keys
and machine-specific filesystem paths are excluded. Restore matching public video
frames using the manifest metadata and verify their hashes before reusing labels.
The allowlisted export is produced by `export_haiku_annotations.py`; it refuses to
overwrite an existing export and checks for credential-like strings before writing.
