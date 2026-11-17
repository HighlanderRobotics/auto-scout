# NorCal identity teaching, larger-data pilot

This follow-up keeps the V1 detector, feature extractor, head, score threshold,
margin and augmentations fixed. It adds early same-event robot views instead of
searching parameters on a test set.

- **1,695 raw crop proposals** from 299 source frames: QM3 100, QM11 100, QM13 99,
  all at 45–65 video seconds. Existing matching cached responses were reused.
- One QM13 frame at **48.4 seconds timed out**. It was not retried or labeled. Its
  $0.01 reservation remains in the original shared ledger. The remaining frames
  were requested independently in a continuation folder.
- `reviewed_v2.jsonl` contains **179 unique agent-reviewed crops**, including V1's
  reviewed data. This is **not independent human ground truth**.
- The local head trains on **180 source crops / 1,080 deterministic augmentations**:
  179 reviewed crops, minus five observations at the two excluded QM3 development
  instants, plus six enrollment cards. There are 15 classes including unknown.

`proposals_*.jsonl` are unreviewed teacher output, including mistakes. Training must
use the reviewed file, not all proposals. `select_identity_examples.py` selects one
high-confidence full bumper read per team/two-second bucket, plus person/field
negatives, for source-image review. It does not approve labels automatically.
The four `review_decisions_*.json` files record the visual review. One confident
QM3 teacher label at 58.2 seconds was corrected from **5104 to 10339** after checking
the actual five-digit bumper on the low flat robot. Both the teacher proposal and
the corrected `label` remain in the reviewed record; training uses `label`.

Each proposal records the official source URL, native video hash/time/box, current
roster/colors, teacher model and request fingerprint. Exported `full_frame_box` is
the detector's original crop used by the student/reviewer; the teacher panel had
an extra eight pixels of surrounding context. Videos, images, raw API responses
and models are local ignored artifacts, not committed.

Training is restricted in code to the configured 45–65-second windows, requires
agent review, validates roster/box bounds, and checks the actual source video hashes.
The two QM3 development instants at 47.6/61.4 seconds are excluded. Adjacent training
frames are correlated; they are not independent samples. QM3 90–110 seconds and
QM13 90–110 seconds are **regression checks** now that V1 outcomes are known. The
fresh nine-observation QM3 120–140-second reference was frozen before V2 training
or identity-head inference on that window. It remains a sparse agent check, not a
full-match or exhaustive ID-switch score.

```bash
.venv/bin/python -u identity_student.py train --config configs/norcal_identity_student_v2.json --out runs/norcal_identity_student_v2
.venv/bin/python -u identity_student.py infer --model runs/norcal_identity_student_v2 --tracks runs/mqz4_identity_late_baseline_v2/tracks.json --out runs/mqz4_identity_student_v2_late_check_v2
.venv/bin/python evaluate_identity_checkpoints.py runs/mqz4_identity_student_v2_late_check_v2/predictions.json --reference configs/mqz4_identity_late_checkpoints_v2.json --out runs/mqz4_identity_student_v2_late_score_v2
```

Use new output folders for new experiments. Local training and inference make no
Claude calls. See DETAILS.md and the saved results for the verdict; defaults remain
unchanged. Source videos are listed in `configs/norcal_identity_pool_v1.json` and
`configs/focus_MQZ4Yrw86uQ.json`.
