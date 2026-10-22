# Handoff: sole-region supervision training closure

## Current state, 2026-10-05

Codex completed a new, bounded reconstruction repair after the user's request to
fix unnatural motion. Current outcome: `out/sole_region_repair_v3/review_all/REPORT.md`.
This supersedes the small A/B below for the current quality decision.

- Warm start from the mature support-repair weights, not fresh smoke weights.
- Full-length sole labels for all 152 training clips; no held-out clips in training.
- Point and Region each completed 10,000 updates from identical loaded learned/EMA
  state (verified hash), same seed, data/statistics and core architecture.
- Official LBS runs only the selected vertices, once for both feet. Values and
  gradients match the full-mesh reference; no manual bind alignment was added.
- Region adds 4/8/16-frame stance displacement on uninterrupted evidence for the
  same material vertex, not an inference foot lock.
- Variable-length batches pad labels as undecided/unmeasured. CUDA visibility now
  precedes project imports; actual UUID is checked. Failed/interrupted launch
  attempts were preserved, not reported as completed runs.
- All 57 fixed held-out windows use identical source scoring masks, after context.

Results (initial / Point / Region): MPJPE 36.26 / 33.15 / 35.69 mm; measured
sole slip 79.08 / 68.64 / 56.54 mm/s; mean region penetration 3.29 / 2.83 / 0.97 mm.
Region peak penetration is still 23.58 mm and peak measured slip 1.08 m/s.
Neither candidate passes the no-skating/penetration screen. Point is retained as
the pose reconstruction reference; no model was promoted to accepted UE locomotion.
Region is a better contact candidate, not proof of natural gameplay motion.

Remote runs: `${PRIVATE_STORAGE}/motionbricks-preparation/runs/sole-region-repair-v3-20261005-r4/{point,region}`.
Full-length labels: remote `motionbricks/out/sole_region_repair_v3/labels_train`.
Actual times/memory: Point 462.76 s / 570.89 MiB; Region 645.99 s / 865.10 MiB.
These are training measurements, not runtime timing.

Local regression suite: 190 passed. Native token decoding matches reconstruction
on all 57 windows: maximum normalized-feature error 1.91e-6, maximum FK error
6.11e-7 m. This is decoding parity, not a motion-quality approval.

Local artifacts include all-window metrics/outcome, training contracts and costs,
five diagnostic showcase videos and official-skinned walk/backward comparisons.
`skinned/run_worst_skinned_source_Initial_Point_Region.mp4` shows the Region arm's
highest measured-slip validation window (`jog_ff_stop_360_R_very_fast_002__A452`,
start 9). It is retained as failure evidence, not a successful locomotion example.
The skinned videos have 32 frames at 15 Hz, preserving 2.133333 s of recorded time;
the failure video was decoded and all 32 frame checksums differ.

Playback is offline
reconstruction; the encoder still sees future source frames. Root and Pose
generation are still untrained/unvalidated. No UE changes or IK were made.
Do not describe the shown human reconstruction as WASD-driven inference.

The next model-quality target remains measured stance slip and rare penetration
without sacrificing the Point arm's pose accuracy. Do not restart skeleton/contact
investigation or declare the Region arm usable based on its improved averages.

## Earlier 1,000-step closure

The training/evaluation closure is complete. DS completed A0-v2 and B-v2 with
1,000 optimizer updates each, including random-window label alignment. Codex fixed
the evaluator's missing A0 rendering color, added rendering regressions, and evaluated
the existing final checkpoints without additional training.

Current artifacts: `out/sole_region_ab/review-v2/REPORT.md`, corrected metrics and
run-summary JSON, rendering status, and five source/A0/B/historical-A comparisons.
Every GIF has 128 frames; every MP4 has 128 frames at 60 Hz (2.133333 seconds).
All metrics exclude context frames [0,4), including intervals touching that range.
Local suite: 178 passed.

B improves sole placement on all five reviewed windows. Sliding remains substantial
and reconstruction is worse on run, turn and backward. This is a small tokenizer
pilot, not a quality-approved locomotion model. The archived run contracts do not
contain initial-weight hashes or sampling-order evidence, which is explicitly
recorded as unavailable in the report rather than added retroactively.

Continue from these results. The preparation tasks below are historical and should
not be repeated. Do not use the old `review/` table as the corrected v2 result.

## Historical preparation record

State as of this record. Nothing here is a claim that the new supervision improves
motion quality; that has not been tested.

## Where things stand

Done and verified:

* Sole-region contact geometry and candidate labels
  (`motionbricks/data/sole_region_contact.py`), with the two label defects fixed:
  an unmeasured interval is undecided rather than stable, and a landing no longer
  vetoes the resting stretch that follows it.
* Official differentiable skinning of the region
  (`motionbricks/data/sole_geometry_losses.py`), with the four integration defects
  fixed: per-vertex measured mask used for slip, official LBS instead of binding the
  whole sole to one joint, contact height gated on measured evidence, and shard paths
  compared by basename.
* Local test suite: 161 passed.
* Remote access and a versioned training-split subset (129 global frames to 128 local
  frames, no padding, validation records excluded).

Not started:

* Labels carried through dataset, collate and random crop on one sampling index.
* `BonesTokenizer.loss` wiring, config switch, version contract, old geometry loss
  switched off.
* Integration tests on the real training path.
* The 1000-step A/B and its metrics and videos.

## Remote environment

    ssh -i "${SSH_KEY_FILE}" -p 12303 private-user@remote-host.example

Host key verification stays on. Data `${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7`,
python `${PRIVATE_STORAGE}/motionbricks-preparation/envs/mb-bones/bin/python`, repo
`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl`, runs
`${PRIVATE_STORAGE}/motionbricks-preparation/runs`. Import kimodo from
`${PRIVATE_STORAGE}/motionbricks-preparation/kimodo`. Check GPU occupancy fresh before
launching; an earlier "GPU 5 idle" reading is not valid later.

The subset is at
`motionbricks/out/sole_region_labels_v1/train_subset/` on the remote, 12 items,
6 behaviors by 2, generated with
`scripts/build_sole_region_training_subset.py --per-behavior 2`.

## Order for the next round

1. Carry frame labels, interval labels and the per-vertex mask through the data
   loader, the collate function and the random crop, reusing one sampling index
   rather than cropping separately.
2. Wire the supervision into `BonesTokenizer.loss` behind a config switch with a
   version record, and switch the replaced geometry loss off so nothing is penalised
   twice.
3. Integration tests on the real training path: the weighted term is in the total
   loss, gradients reach network parameters through the skinning while labels receive
   none, random window starts, batch resampling and both interval endpoints align,
   no valid contact gives a finite loss with zero counts, and a missing label raises.
4. Re-check GPU occupancy, then run A and B for 1000 optimizer updates each with the
   same 12 items, the same initial state, a fresh optimizer, the same sampling order
   and the same budget. Report slip, penetration, contact height, reconstruction
   error and valid sample counts on the fixed validation windows, with source, A and B
   overlaid videos and the real step count, config, checkpoints, wall time and memory.

Do not re-investigate the skeleton, redesign the contact scheme or reconvert the
existing subset. Revisit a module only when an integration test shows a concrete
defect.

## Boundaries kept so far

No training has been run against the new supervision. No existing model output,
training configuration or shipped data label was modified. The four modified tracked
files in `git status` predate this work.
