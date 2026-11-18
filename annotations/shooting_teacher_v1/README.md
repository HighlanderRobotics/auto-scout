# Hybrid shooting teaching pilot

These are **unreviewed Claude proposals**, not a shooting/count answer key.
`proposals.jsonl` contains 63 Haiku sequence labels and 8 Opus reviews of selected
sequences from NorCal QM3. Identity-tracking labels are not used as shooting truth.
Each record preserves source URL/hash, timestamps, focal box, crop, frame sequence,
features, split, model, uncertainty and request fingerprint. Videos/images stay local.

The first 45 detector-selected six-frame examples were largely ambiguous. The next
18 examples use explicitly highlighted subjects and 12 frames per one-second window, with more
space above the robot for outgoing trajectories. Opus reviews eight training cases.
It recognizes a shooting stream in a clear red-hub example but cannot count distinct
balls reliably in that dense stream. Counts stay null; no made totals are invented.

The initial local shooting-support model fits **8 weakly labeled examples, 3 shooting**.
Teacher confidence is preserved as a soft target and sample weight. These examples
have **zero human-confirmed labels**. The two usable validation teacher labels are
both nonshooting: their agreement is not a useful shooting-accuracy or recall score.
The model is a review proposal only, not an autonomous scoring system.

The separate hub-fuel experiment trains on dedicated ALLIANCE fuel-counter increases
and visible hub-area motion, with a fixed nominal two-second delay and an active-hub
arrow feature. It is neither an attempted-ball count nor a per-robot made count.
Its numerical targets are unreviewed OCR, so errors against those targets do not
establish human-verified accuracy. See results/hybrid_shooting_qm3_v1.json.

Open the generated review desk to correct action/team/attempt counts, adjust times,
edit fuel-counter OCR, or draw a new main-view robot example. Save the reviewed JSON;
then retrain using --reviews and a new --model-out folder. Human action labels override
teacher labels. At least six reviewed attempt counts with variation are required to
fit the optional attempt-rate head. No real attempt-count model has been trained yet.
Reviewed numerical counts are never inferred from total match points or alliance
counter increases. The latter can be attributed only as a separate reviewed subtotal
when the reviewer confirms counter readings and exclusive contribution; conflicts,
overlaps and missing data remain unassigned. Per-robot full-match totals remain null.

No video, credentials, raw API transcripts or weights are committed. Rebuild images
from the official linked video and recorded native crops/timestamps. Current teaching
budget uses the existing shared ledger; it does not reset or duplicate its allowance.

`hub_counter_targets.jsonl` preserves 569 weak numerical examples: 375 training
(QM3/QM11), 194 separate-match validation (QM13). The numerical manifest records
the fixed crops, nominal delay, source hashes and model results. These targets
are OCR observations, not reviewed robot shot counts.
