# Matched PARC Vertical-Frame Experiment

## Question and decision

Does removing absolute world altitude from every stage improve continuous
terrain motion, or does the remaining failure primarily arise from body-token
prediction and feedback? World-translation equivariance alone is not motion
quality. The experiment must measure both, with no foot locking, IK, output
projection, future truth at runtime, or model replacement before review.

The two arms start from the same completed PARC 50k tokenizer and matched 10k
online Root/Pose pair. Each arm adapts the full tokenizer, then Root, then Pose,
2,000 optimizer updates per stage. Encoder and codebooks can change during the
tokenizer stage. Pose's input embeddings are initialized from its own new
codebooks through the official method; the whole Pose network is then trained
on token labels from that frozen tokenizer and trajectories from that arm's
frozen Root. Old and new token vocabularies cannot be interchanged.

| Arm | Model vertical coordinates | Shared changes |
| --- | --- | --- |
| absolute | source world Y | four observed decoder context frames and actual-context velocity boundary |
| observed | Y relative to last observed pelvis | same context and boundary contract |

The native MB classes, architectures and losses run through their existing
`training_step`. New logic changes only coordinate preparation and the declared
context contract. Both arms use original loss coefficients, Adam 5e-5, batch256,
warmup10k/cosine2M, clip0.5, identical source weights and seed2333. These are
bounded adaptations with fresh optimizer states, not uninterrupted resumes or
completed paper-budget training. Training uses only the verified PARC dataset
`16401790c84c21cc7e4f52e6f75ec2cb88f587b3934caf481dfd05e3b58f3af3`.

## Coordinate contract

Let the last observed pelvis Y be `y3` and let `g` be the existing PARC train-only
Root-Y normalization mean (1.2381402254 m). Set `origin = y3 - g`. Subtract this
one constant from Root Y and every RIC joint Y throughout the window; velocities,
rotations and contacts remain unchanged. Dense terrain points already use the
current pelvis as their origin, so they remain unchanged. Decode in this frame,
then add `origin` back to Root Y and all RIC Y. Do not recenter each future frame.
Uphill motion, jumps and crouches retain their full relative height changes.

Keeping the same immutable affine statistics is deliberate in this paired
warmstart: changing statistics would add a second intervention and invalidate
the meaning of copied input/output weights. The train mean is a fixed gauge,
not a floor estimate or a generated hip-height target. Both arms retain the same
scales. A future statistics revision requires a new artifact and retraining.

Training centers after selecting the actual random window and yaw augmentation,
before global/local conversion and token encoding. Runtime derives the reference
only from frame3 of four actual observations. Future height never selects the
origin. Every stage's config declares `observed-pelvis-y-v1` for the new arm;
the portable loader rejects mixed frames, stale token dependencies and unbound
context settings. The live preview remains on its old artifact.

## Experiments

1. Lossless representation and invariance: preserve FK, SO(3), velocities and
   future vertical differences; translating the scene/history together must not
   change relative output. Changing future data must not change the origin.
2. Native training and artifact integration: execute all three original training
   methods, confirm gradients, save/reload the complete pipeline, verify frozen
   Pose dependencies. The real one-update pipeline completed on PARC data.
3. Eight fixed held-out windows: same inputs and scoring frames, decomposition
   using model/source tokens and model/source Root conditions offline. Source
   futures are diagnostic oracles only, not controller inputs.
4. Teacher-history versus played-history: four consecutive eight-frame blocks
   per held-out window, same explicit planar goals. Only the teacher comparator
   resets to expert history; the other branch feeds back actual generated FK.
   Compare per-block signed height bias, final error, FK error and sole geometry.
5. Longer 24-block idle/forward online rollouts, plus the whole-world translation
   test. Include per-scene results; the large-drop seed is not an upstairs test.

Foot geometry uses official SOMA skinning and fixed material-vertex indices.
The read-only evidence gate selects source vertices within 2 cm of the local
box top and intervals below 0.1 m/s; it is not measured physical contact truth.
Predictions never choose the source scoring mask. Invalid terrain counts remain
explicit. Vertical hand contacts and overhangs are outside this heightfield-top
diagnostic. Source and prediction statistics must be reported together.

Improvement requires reduced closed-loop drift without worsened contact and
body quality on the held-out cases; lower reconstruction loss or exact
equivariance alone is insufficient. Failure redirects the next experiment to
the responsible stage, not automatically to a longer copy of this run.

## Execution

Remote immutable training snapshot:
`${PRIVATE_STORAGE}/projects/motionbricks-parc-training-20261009/native_terrain_code_v13_vertical_frame`.
Outputs on the same host NAS:
`${PRIVATE_STORAGE}/motionbricks-parc-training-20261009/vertical_frame_ab_v1/{absolute,observed}`.
Local receipts:
`out/parc_training_20261009/vertical_frame_ab_v1`.
The first smoke run hit the OS shared-memory file-descriptor limit before Root
completed; its owned processes were stopped and the failed output preserved.
The second run raised the per-process soft limit within the existing hard limit,
closed completed data-loader workers, and completed all three stages and export.
No model/loss workaround was used for this infrastructure failure.

Both arms completed all three stages at 2,000 updates each. Their original
input-batch hashes match exactly at every stage. All five automatic review jobs
completed with exit code 0 for both arms. Neither candidate is quality approved.
B was subsequently enabled for the user-authorized interactive preview described
below. No frame-ablation trainer or automatic reviewer remains running at
the 2026-10-10 status check.

## Primary references

- MotionBricks section4: conditional multi-head tokenizer, projected Root/body
  separation and decoder refinement; Algorithm1/AppendixG: played-state feedback.
  https://arxiv.org/html/2604.24833v1
- PARC's actual `PARC/motion_generator/gen_util.py`: common motion/heightfield
  relative-Z frame with inverse restoration; its motion sampler implements the
  corresponding training coordinate treatment.
  https://github.com/mshoe/PARC

These references support the mechanisms. The new online terrain task and
coordinate adaptation are explicitly our extension, not claimed to be the
unchanged original UE demo.

## Additional Hypothesis: Initial Pose Masking

The shipped Pose sampler uses `masked_token_ratio=0.8`, so training never masks
every token, while an online first pass starts with all unknown tokens. The
new task also lacks authored ending pose constraints. This is a real task
distribution difference, but it does not by itself establish a dominant cause.

The read-only source-10k probe compares exactly the same 527 masked future-token
positions in eight held-out windows, with identical Root predictions, observed
poses and terrain. Native partial masking has cross-entropy 2.8296 / accuracy
0.1935; fully unknown input has 2.9275 / 0.1822. Partial masking has privileged
future token information and is not a runtime candidate. The small measured
difference does not justify a separate masking modification now. No such change
was added to the two running arms. This result prevents a plausible explanation
from becoming another unverified intervention.

## Completed Results

Same eight held-out windows, same planar goals and scoring, metres:

| Measurement | Absolute A | Observed frame B |
| --- | --- | --- |
| Root Y error with expert history | 0.07583 | 0.03838 |
| Sole gap on source-near vertices with expert history | 0.09025 | 0.04956 |
| Root Y error with generated FK history | 0.11156 | 0.14237 |
| Sole gap on source-near vertices with generated FK history | 0.12554 | 0.16889 |
| FK MPJPE with generated history | 0.24363 | 0.26370 |
| Same-material-point speed on source-slow intervals (m/s) | 1.38364 | 1.41374 |

The vertical-frame intervention improves teacher-history predictions but does
not improve the paired closed-loop validation aggregate. The result does not
justify deployment or claiming that more steps of the same adaptation solve
drift. Its whole-world translation deviation is about 2.4e-7..2.9e-7 m in FK,
confirming that the coordinate dependency itself is removed.

The separate 24-block source-scene test ends at pelvis Y=2.4050 m for A and
1.7035 m for B, from 0.9627 m initially. This one-path improvement does not
override the worse multi-window closed-loop result. Native learned duration in
B produced final Y=0.8816 m and maximum Y=0.9995 m in an additional probe, but
the maximum known pelvis/terrain gap was still 10.10 m over the drop. Motion
duration, travel distance and number of replans differ; this is not a matched
traversal approval, and the timing option remains off by default.

The nine reserved skill-test windows cover stairs, jumping and mid_climbing.
B improves height MAE from the pre-adaptation source model's 0.18658 m to
0.07641 m, but penetration worsens from 0.00177 m to 0.01550 m and footpoint
speed remains 0.66725 m/s. A's corresponding reserved skill test was not run
before interruption. Therefore these figures are a source/B comparison, not an
A/B representation attribution. Three fixed validation examples (mixed terrain,
platform and climbing, windows 0/1/2) have now been rendered with official SOMA
skinning at 30 FPS, 32 frames each, in `vertical_frame_ab_v1/visualization_v2`.
The gallery supports 0.25x/0.5x/1x playback. All panels share the same source-XZ
camera motion and fixed vertical framing; model motion has not been aligned,
smoothed or corrected. MP4/GIF frame counts and browser playback were verified.
The first `visualization` directory is invalidated because separate terrain/body
collection sorting falsely occluded limbs; v2 sorts their triangles together.

## Data and Control Coverage

The validation split contains no dedicated stairs-category record. The ordinary
eight-window aggregate therefore cannot certify staircase traversal. Training
contains four stairs-category records plus one generated stairs record; category
counts do not establish how many actual stair actions occur in mixed terrains.

An unweighted stride-four scan of training windows found:

| Window size | Windows | Near-zero planar goal | Of those, Y range >15 cm | Quiet root windows |
| --- | --- | --- | --- | --- |
| 32 frames | 10,916 | 13 | 9 | 0 |
| 64 frames | 7,388 | 5 | 5 | 0 |

Near-zero means average endpoint XZ displacement/time <=0.05 m/s; quiet means
maximum three-dimensional root speed <=0.05 m/s. These explicit thresholds do
not prove the absolute absence of all standing-like frames. They do show that
the sampled dataset provides extremely weak evidence for the intended
"release WASD -> stand still" behavior. A nearly unchanged XZ endpoint can
instead label substantial vertical movement. Do not fix that semantic gap by
freezing the output, duplicating a frame as training data, or importing excluded
non-PARC sources. Genuine stop/start/standing and recovery sequences need to be
generated/recorded through the authorized PARC workflow and admitted as a new
versioned dataset before further controller training.

Local final receipts and reports are under
`out/parc_training_20261009/vertical_frame_ab_v1`. Bundle SHA256 values:

- A: `dc1555d73efd806f687c3a84883e5257c200f4185d2b6e2ec0f3bcec9083bbd2`
- B: `957680feced38ff0f496c89f7ca5292e4ce8a11aaf63da37cba7fee85fb3b954`

The full bundles remain on the remote NAS; B also has a hash-verified local copy
for the interactive trial. Local reports include both state
receipts, both feedback summaries, both long source/flat rollouts, B's coordinate
and duration probes, source/B skill-test summaries, masking diagnosis and intent
coverage.

## WASD Preview: 2026-10-10

The user explicitly requested interactive control of the relative-height model.
The existing third-person browser now runs B at `http://127.0.0.1:28880/?mode=terrain`.
Local checkpoint `portable/relative-height.pt` matches B's SHA256 above. Its
Root/Pose/tokenizer parameters, four decoder iterations and fixed 8-token planning
horizon are the same as the A/B visualization; the separate native-timing probe
has not been enabled by this deployment.

Scene `out/soma_lab_relative_live_20261010/live-climbing-scene.npz` contains the
actual paired terrain and exactly four source frames from validation window2,
start35. It contains no future poses or trajectory goals. The scene is hash-bound
and dataset-bound; the loader rejects extended histories and additional arrays.
The camera starts along the observed body heading. WASD commands now determine
the planar goals; future Y and body poses remain model predictions. Camera
collision remains disabled and all existing trajectory overlays are available.

Actual keyboard tests covered all four directions with 32 generated frames per
case. Displacement along each camera-relative input was positive: W1.686 m,
A1.603 m, S1.646 m, D1.858 m, for 1.6 m/s intent. These are short input-response
checks, not motion/contact acceptance. Real browser tests also passed held-W
camera rotation, mouse lock, Y inversion, zoom, reset, desktop/mobile rendering,
and playback acknowledgement order. See the `tps`, `keyboard`, and `playback`
reports in `out/soma_lab_relative_live_20261010`.

Playback measured 23.89 FPS idle and 24.10 FPS moving; fresh-inference waits
remain visible. Stable 30 FPS is not yet established. Initial state, pause/reset,
or air phases must not be interpreted as proof of a supported-contact failure.
Standing/stop behavior and longer terrain traversal remain experimental.
