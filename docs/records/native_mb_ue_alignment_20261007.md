# Native MB UE conditioning alignment

## Confirmed integration defect

The previous UE adapter passed four context and four target keyframes to Root
and Pose. Both 50,000-update checkpoints only covered a maximum of one keyframe
at each end under their actual upstream training curriculum. The configured
maximum of four is not the number learned at this checkpoint. The official
Root/Pose scheduler ramps that maximum over 200,000 updates.

Dense conditions produced excessive root velocities before packets reached UE.
The root stage already diverged; native decoder refinement changed those
velocities further. The rate/units/root-authority audit did not identify a
second root application or a metre/centimetre conversion error in UE.

The runtime now uses one context keyframe and one target keyframe. Coverage is
checked by calling the actual upstream Root/Pose `_construct_keyframe_prob`
methods with the checkpoint update counts and configurations, rather than
duplicating their schedule. The runtime also preserves the ending target
condition, matching the released interactive/navigation demo arguments.

This follows the paper's sparse navigation design. It does not claim to recover
NVIDIA's unreleased UE implementation. The existing eight-frame committed
playback remains a UE integration choice: discard decoded context, publish
frames 4:12, then use the last played frames as subsequent history. The separate
released-G1 cursor comparison below shows that changing the cursor alone does
not repair this failure. No motion output was clamped, replaced or filtered.

Official sources:

- [MotionBricks paper](https://arxiv.org/html/2604.24833v1): sparse navigation
  keyframes, native root/pose/decoder refinement, and Appendix G's played-pose
  context updates.
- `motionbricks/motion_backbone/models/root_model.py` and `pose_model.py`:
  actual training keyframe curriculum.
- `motionbricks/scripts/interactive_demo_g1.py` and
  `motionbricks/motion_backbone/demo/full_agent.py`: endpoint
  condition arguments; the constructor default and demo invocation differ.

## Frozen-weight comparison

Same portable bundle, FP32, TF32 disabled, RTX 3080, no retraining:
`a17027e3cf8fc4958437329fa91a0889324949f37458a87af36c8d9c824e963c`.

Each case runs 45 eight-frame plans at 30 Hz: idle 3.2 s, forward 6.4 s at
1.2 m/s, then idle 2.4 s. The path statistic includes initial drift, startup,
stopping and block boundaries; it is not steady-state command accuracy.

| Conditions and cursor | Path (m) | Maximum planar speed (m/s) |
| --- | ---: | ---: |
| Previous 4 context / 4 target, endpoint omitted | 154.778 | 28.703 |
| 4 / 4, endpoint preserved | 151.865 | 24.518 |
| 4 / 4, endpoint preserved, released G1 cursor | 150.006 | 18.909 |
| Corrected 1 / 1, endpoint preserved, played context | 6.834 | 1.790 |
| 4 context / 1 target | 24.328 | 14.392 |
| 1 context / 4 target | 57.217 | 9.721 |

Evidence: `out/native_mb_lafan_paper_20261006/ue-alignment-review-v2/report.json`
and `traces.npz`; reproducible with `scripts/diagnose_native_soma_alignment.py`.
These comparisons isolate the dense conditioning defect without relying on
the UE result or altering the checkpoint. They do not approve all locomotion
quality or all possible commands.

## Corrected UE trial

Trial: `native-lafan50k-aligned-20261007`; same bundle SHA256 as the failed trial.

| Measurement | Previous trial | Corrected trial |
| --- | ---: | ---: |
| Forward command (m/s) | 1.2 | 1.2 |
| Forward measurement duration (s) | about 3 | 3.001678 |
| Forward displacement (m) | 40.807 | 2.358333 |
| Measured forward average (m/s) | about 13.6 | 0.785672 |
| Maximum applied planar frame step (cm) | 107.2 | 5.869748 |
| Game-loop FPS | 25.374 | 29.996201 |
| Frame-time P95 (ms) | 47.894 | 35.687901 |
| Steady buffer underruns | 0 | 0 |
| Maximum raw retarget pose switch (degrees) | 177.809 | 100.890854 |

The corrected session applied 391 frames from 50 plans at native 30 Hz, with
zero skipped frames, zero pose blending, zero warmup/steady underruns and camera
lag disabled. The screenshot shows the character on the finite test floor,
not outside it as in the failed trial. The forward average includes command
startup and is not evidence of exact 1.2 m/s steady-state tracking.

Verification: 143 focused Python regressions passed; the UE editor build
succeeded; five `MotionBricks.SomaControl` tests passed, including the real
Enhanced Input PIE sequence and 30/60 Hz chunked-root displacement invariance.
The required gross-overspeed test passed. Its permissive limit is explicitly
an engineering heuristic, not an official MB bound or quality approval.
The corrected continuous playback performance gate also passed.

Evidence:

- `F:/dev/GASP_57/Saved/MotionBricks/native-lafan50k-aligned-20261007/`
  contains server/runtime hashes, logs and automation results.
- `F:/dev/GASP_57/Saved/MotionBricks/soma_control_pie_causal_native-lafan50k-aligned-20261007.json`
  contains the actual motion, frame-time, mask-free playback and gate metrics.
- The corresponding `.png` is the unmodified PIE screenshot.

## Remaining limits

Gross command overshoot is fixed for this frozen-weight reproduction and PIE
sequence. Natural full-body motion is NOT approved: the maximum unfiltered
retargeted pose change remains 100.9 degrees. The snapshot also shows arms held
high; that requires a separate source-pose/retarget trace, not a root speed
clamp. Foot sliding and long-duration/general-command quality are not validated
by this trial. Tokenizer/Root/Pose remain 50k checkpoints, not the full paper
training budget. No training, loss changes, IK or foot locking occurred.

## Manual trial

The corrected visible Editor and its hidden inference server were launched
after all required regression checks passed. Open the SomaControl map and use
PIE/WASD. To launch a later fresh trial after closing the owned Editor/server:

```powershell
& F:\dev\GASP_57\Scripts\RunNativeSomaLocomotion.ps1 `
  -Bundle ${PROJECT_ROOT}\out\native_mb_lafan_paper_20261006\ue-trial50k-v1\native-soma-50k.pt `
  -TrialName native-lafan50k-aligned-next -RequireGrossOverspeed
```

The launcher never kills an unrelated process on an occupied port. With the
required regression flag, a failed test prevents launching the manual Editor;
all evidence remains retained. Passing this regression is not motion-quality
approval.
