# Native MB Planar Online V1

## Basis and Scope

MotionBricks paper sections 3, 5 and 7 describe a smart primitive generating
keyframe goals, followed by Root -> masked-token Pose -> conditional codec.
We retain that pipeline and the shipped Root/Pose objectives. This is an explicit
terrain/partial-component extension, not unmodified paper reproduction.

Reference: <https://arxiv.org/html/2604.24833v1>.

The existing PARC terrain v4 training snapshot is immutable; its three stages
have now completed their bounded 50k updates.
Its full endpoint XYZ/body-pose conditioning is not an online WASD contract.
New online runs use separate configuration, code snapshot, output and bindings.

## One Contract

| Input | Availability |
| --- | --- |
| Four played Root XYZ/heading and body poses | Observed |
| Three forward Root velocity intervals within history | Observed |
| Fourth velocity interval crossing into future | Unknown |
| One endpoint XZ and cos/sin heading goal | Commanded |
| Future Root height | Unknown; Root predicts it |
| Future body pose and velocity | Unknown; Pose predicts motion |
| Terrain near support and toward the planar goal | Observed at history frame 3 only |

`sparse_planar_conditions` is shared by training and runtime. Root has an explicit
boolean `[B,8,5]` component mask. Unknown components are sanitized before every
projection, including the Root convolutional decoder's target embeddings. A
masked height sentinel is not a measured zero or an instruction to stay level.
Missing-component embeddings are learned; legacy full-frame behavior is preserved
when this extension is disabled or the optional mask is omitted.

Only current bounded `[B,2,9,9,3]` points and validity enter the networks. The static
heightfield remains in the data/query layer. Training reconstructs current world
Root state from the dataset's recorded canonical transform, then re-queries along
the same root-local goal direction as runtime. It does not reuse the cached
current-velocity direction when the intended direction differs.

## What the Dataset Teaches

PARC does not provide captured WASD for every recording. Offline endpoint XZ and
heading are hindsight goal labels from expert motion. They are not presented as
observed future motion or logged user commands. Future Y/body are used only as
supervised targets, never goal inputs.

At runtime, desired planar velocity and heading define an endpoint over the known
planning horizon. This endpoint is an input request, not a generated trajectory;
the model must infer how to approach it and whether to change height/posture.
Command time is measured from current history frame 3 to horizon frame T-1.
The paper-profile training keeps 3..16 tokens (12..64 frames); the online command
wrapper initially uses the conservative 6..16 subset. Root's original duration
classification loss is retained. Off-range training windows are disabled only for
this online profile.

This goal formulation does not prove that the model can safely execute every
arbitrary command. Held-out source-derived goals test the input contract and
motion reconstruction; closed-loop command changes and unsupported goals remain
separate behavior acceptance tests. Heightfields cannot represent overhangs,
ceilings, or arbitrary climbing handholds.

## Root -> Pose -> Codec

Root training keeps native global/local Root regression and duration classification.
Pose keeps native masked/perturbed code prediction and cross-entropy, but its
Root trajectory and relative height come from a frozen, verified online Root
checkpoint rather than the ground-truth future trajectory. Future pose conditions
are absent; the four real history poses remain available.

The existing 50k tokenizer covers one/two arbitrary decoder pose keyframes. Online
codec decoding therefore constrains only the last observed pose (frame 3), not
four untrained keyframes. Its dense planar velocity condition comes from predicted
Root, and its generated Root/height remains the native codec refinement.

No IK, foot lock, smoothing, terrain height projection, decoded Root overwrite,
ground-truth Root override or ground-truth pose-token override is permitted by the
online inference gate. Existing flat live/replay paths are not silently replaced.

## Verification and Deployment

Automatic tests exercise actual Lightning training_step and complete native
inference, not just helper functions. They check component-mask propagation,
future-Y/body perturbation invariance, masked-NaN invariance, source-row/window
alignment, current-only query time, directional scene changes, frozen Root/codec,
prediction gradients and rejection of incompatible/debug substitution inputs.

Weights-only transfer is same PARC dataset and same stage, with exact source SHA,
config, normalization and tokenizer dependency checks. Only Root's new mask
parameters may be absent in a v4 checkpoint. They start at zero; Adam/scheduler
state is explicitly reset for the new training contract. A frozen Root used by
Pose is separately bound and is neither optimized nor duplicated in Pose state.

Run a bounded 1,000-update Root/Pose adaptation preflight before extending the
budget. Preserve source/conditioning audits and compare held-out planar-goal
inference before deployment. Tests or lower loss alone are not evidence of
successful stair traversal or robust WASD control.

## Execution Receipt

The v6 online code snapshot passed 60 focused tests locally and 60 in the remote
training environment. The bounded workflow is `online_contract_preflight_v3` on
the 4090D host. It adapts Root first, waits for the existing baseline Pose final
checkpoint, then adapts Pose with the frozen new Root and runs held-out inference.
The old v4 training and browser live backend remain unchanged.

Two startup issues were preserved as separate failed outputs: the paper-schedule
argument was one step too large (now uses `PAPER_STEPS` and a validation test),
and the process file-descriptor soft limit was 1024. The fresh workflow inherits
65536; neither the global system limit nor model/data objectives were changed.
See `out/parc_training_20261009/execution.json` for hashes and a time-specific
progress receipt. This launch is not a motion-quality acceptance result.

## Completed Preflight and Online Playback

Both v6 online stages completed 1,000 real optimizer updates. Root's three new
component-mask parameters received nonzero gradients on every update; Pose used
the frozen new Root on every update without changing its tensor digest.

Matched eight-window reviews use only the four observed poses and a hindsight
planar goal. Native four-iteration generation improves final Root XYZ error from
0.691 m to 0.510 m versus one iteration. Root-stage Y error is still 0.266 m,
and decoded-versus-Root Y disagreement is 0.288 m (one iteration: 0.508 m).
The latter is disagreement between generated stages, not a ground-truth error.
All-invalid terrain increases final Root XYZ error to 0.787 m. These are small
fixed-window diagnostics, not successful traversal or robust command evidence.

The separate experimental frontend uses a verified PARC-only portable bundle.
Its seed contains exactly four observed frames, not a reference future. The
frontend receives unchanged generated frames; it commits history only after
playback acknowledgement. Reset discards unplayed predictions. Existing flat
live and offline source/codec replay remain separate modes.

Following paper Algorithm 1, the online backend retains its generated future
buffer and generates again when the command changes or the buffer is low.
Reusing future predictions never makes them observed history before playback.
The initial implementation publishes eight frames per block and excludes the
last four generated endpoint frames. Its exact replanning frequency is an
engineering setting, not the paper's 3..9-frame frequency experiment.
Desktop/mobile render and input checks are operational checks, not quality or
30 FPS certification. The model currently queries bounded geometric points
from the measured heightfield; the UI depth preview is not the network input.

## Bounded Continuation

The v7 continuation changes checkpoint/command plumbing and evaluation only;
network, motion representation, original losses, normalization and data remain
unchanged. Root resumes the verified full Lightning state at update 1,000,
including learned masks, Adam and the original scheduler, to a total of 10,000.
Automatic startup checks compare the actual restored state digests. The sampler
is reseeded, so this is not bitwise uninterrupted training.

Pose copies all matching prior online backbone weights but resets Adam and the
scheduler because its frozen Root dependency changes. It performs 10,000 new
updates using the new verified Root. Old/new Root dependencies are both recorded;
the exact 50k codec dependency is preserved. Both stages retain paper Adam 5e-5,
batch 256, warmup 10k and schedule 2M; the bounded stop is not full paper training.
The continuation automatically runs matching four-iteration observed/invalid
terrain reviews with per-stage diagnostics. No quality approval is inferred from
completion, test counts or a lower training loss.
