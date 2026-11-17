# NorCal identity teaching pilot

`proposals.jsonl` contains 100 independent Haiku crop proposals from nine development
frames each in NorCal Championship 2026 QM3 and QM11. `opus_proposals.jsonl` adds 12
crop proposals from two difficult QM11 panels. They are **unreviewed proposals**,
including unknowns and nonrobots, not an answer key.

`reviewed_v1.jsonl` contains 40 unique source crops visually checked by Codex against
native footage. Positive candidates required a full matching bumper read and teacher
confidence >=0.8; nonrobot candidates required confidence >=0.85. Review confirmed
these selected crops and added the two Opus-checked blue-bumper 5104 examples. This
is agent review, not independent human ground truth. It does not validate whole
tracks; the same tracker ID can switch physical robots.

Each record includes the official source URL, video SHA256, video time, native
full-frame crop box, current match roster/colors, teacher model, request fingerprint,
reference-gallery hash and review status. Recreate an image by downloading the
linked source video at native 1080p, seeking to `video_s`, and cropping
`full_frame_box`. Source footage, images, model weights, API responses and credentials
are excluded from Git. Never infer team identity from a monitor's displayed number.

QM11 provides extra views of shared teams; 5104 changes from red in QM3 to blue in
QM11. QM13 is a separate transfer source and contributes no training examples here.
No QM3 90–110-second evaluation images contribute to these files.

`identity_student.py train --out runs/norcal_identity_student_v1` (use
`.venv/bin/python`) fits one experimental local identity head from reviewed crops
plus the six enrollment cards in `configs/mqz4_gallery_v2.json`. The config excludes
QM3 47.6 and 61.4 seconds from training. It uses 41 source crops / 246 deterministic
augmentations, frozen ResNet18 + HSV features and a ridge head. It does not retrain
the robot detector. Scores are regression support, not calibrated probabilities.
Unknown, weak, absent-roster and duplicate team claims stay unresolved. The fixed
settings are in `configs/norcal_identity_student_v1.json`; do not tune them on the
later seven-checkpoint results. This small dataset cannot establish general tracking
accuracy or unseen-team recognition.
