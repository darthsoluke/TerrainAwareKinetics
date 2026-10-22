# Local Control and Mixed Training, 2026-10-10

## Scope

The browser, rendering, and online inference stay on the local RTX 3080 PC.
Servers perform preparation, training, and offline numerical evaluation only.
The user now permits BONES-SEED together with official and generated PARC.
Large transfers use a private Google Drive task folder or internal storage,
never public-IP SSH/SCP data transfers. Existing source datasets are unchanged.

## What the Experiments Established

- The previous 24-frame forecast lasted 0.8 seconds. Its exhaustion exposed
  about 90 ms inference, repeated history reconstruction, and a 1.7 MB geometry
  packet. Recorded browser frame gaps reached 233-237 ms.
- Fixed-shape CUDA graphs preserve four native argmax passes. Five changed
  history/command cases retained every active token, with maximum FK discrepancy
  below 0.000002 m. Eager prediction was about 72 ms, graph replay about 13 ms.
  Final complete prediction including diagnostics measured median 17.03 ms,
  P95 22.72 ms; observed-history reconstruction 0.96 ms; rig geometry 1.38 ms.
- Browser transport now carries 77 world rig matrices, approximately 39.5 KB
  per eight frames. GPU skinning retains all eight influences actually present
  in the asset. The upstream comment saying five is stale. CPU reference checks
  on 774 vertices, including the strongest extra influences, differed from
  official LBS by at most 0.0000004 m. Desktop/mobile canvas and TPS checks pass.
- The source pose is not corrected. There is no IK, root replacement, foot lock,
  terrain projection, smoothing, skipped frame, or incompatible midpoint splice.
  A fresh forecast still starts after the full played-history acknowledgement.
- Before the GPU warm-session change, short browser timing runs improved to
  around29.4-29.8 FPS, but other runs retained100-230 ms spikes. The final warm
  session run `playback-warm-v2/checks.json` measured29.48 FPS idle, P95 39.27 ms,
  max47.5 ms (zero intervals>50 ms), and29.35 FPS moving, P95 39.35 ms,
  max71.2 ms (one interval>50 ms). No catch-up bursts occurred. The severe
  periodic stalls are substantially reduced; strict zero-spike30 FPS remains
  unapproved. Preserve individual reports; do not present the best run alone.
- Live NVML sampling identified repeated transitions between P2 at9251 MHz
  memory and P5/P8 at810/405 MHz. An active-session maintenance replay avoids
  cold GPU wakeups. It consumes no output, updates no observed
  history, and expires400 ms after the last terrain request. No global driver
  clock or power setting is changed; this trades some active-session GPU work
  for predictable response time.
- A fixed world-XZ stop-goal ablation reduced horizontal drift but worsened
  penetration on some sequences. It is not enabled in the live backend.

Evidence: `out/soma_lab_relative_live_20261010/`, including
`hitch_execution-v3/benchmark.json`, `hitch-profile-final/profile.json`,
`gpu-skin-v1/checks.json`, `tps-final/checks.json`, `stop-intent-ab.json`, and the
versioned `playback-*` reports. The live bundle remains relative-height.pt,
SHA256 `957680feced38ff0f496c89f7ca5292e4ce8a11aaf63da37cba7fee85fb3b954`.

## Control Semantics

The existing v1 protocol turns velocity into a planar endpoint. Zero input is
therefore a return-to-current-position task, not an explicit desired final
velocity. Original MB already embeds sparse local Root velocities and applies
known terminal components in `_predict_root_trajectories`. The new v2 extension
uses this mechanism with explicit component masks:

- Four observed poses and three observed forward velocity intervals.
- Desired terminal VX/VZ and heading. Future XYZ, angular velocity, and body
  pose remain unknown. Root predicts braking displacement and vertical motion.
- LocalRootGlobalJoints uses world VX/VZ because heading removal is disabled.
- A zero-initialized local-mask projection distinguishes unknown height and
  angular velocity from known zero. Every old Root parameter loads strictly;
  this projection is the only newly initialized parameter.
- The same masks and terminal condition are used by Root training, frozen Root
  predictions for Pose training, and inference. Original objectives remain.
- Terminal velocities from source clips are hindsight goal labels. They are
  not represented as captured keyboard commands or external future observations.

This is an explicit MB conditioning extension, not an unmodified-paper claim.
Tests cover unknown NaNs, no future XYZ/Y/pose leakage, training gradients, and
preserving the Root-predicted height when supplying terminal planar velocity.
References: official MB `root_backbone.py`, `motion_inference.py`, and
[MotionBricks](https://arxiv.org/html/2604.24833v1#S4.SS2).

## Intent-Dependent Observation

`intent-horizon-range-v1` uses the shared training/runtime sampler. Near support
stays +/-0.4 m at 0.1 m spacing. The directional extent is desired speed times
the explicit prediction horizon plus 0.4 m, bounded to 0.8..3.4 m. Lateral
extent remains +/-0.6 m. Both patches remain 9x9, with metric XYZ and validity.
This preserves resolution near the feet and a fixed query/encoder cost.

Old checkpoints retain fixed-v2 observations. A new range policy requires
matching Root/Pose configs and a trained artifact binding. Static maps are
queried only at the last observed frame; no future heights or future poses are
supplied. Batched ragged-map queries equal individual queries, including NaN
padding, duplicated rows, different origins/spacings, and out-of-map cells.

## Data and Training

A100 base: `${PRIVATE_STORAGE}/motionbricks-20261010`.
Final dataset: `mixed_control_full_v3`, manifest SHA256
`88f7023556e8f32334c5b7a8c5ed18d8bf176f062b31390e9891915eda94574a`.

| Split | PARC Official | PARC Generated | BONES-SEED |
| --- | ---: | ---: | ---: |
| Train | 437 | 4 | 2003 |
| Validation | 48 | 0 | 19 |
| Test | 49 | 0 | 20 |

BONES train categories: idle348, start557, stop448, walk154, run64, turn136,
backward149, lateral147. Source recording/augmentation families and existing
splits are retained. BONES motion shard bytes are unchanged. Its plane is an
explicit constant source-plane assumption at Y=0, not measured terrain or
force truth. Official skinned sole regions screen source penetration with
fixed bounds P99<=30 mm and max<=80 mm; all selected 2042 clips passed. This is
a source admission screen, not generated-motion approval. Stats remain the
original PARC train-only affine scales for checkpoint-compatible warmstart.

The earlier bounded subset was `mixed_control_v1` (192 BONES train clips).
The partial `mixed_control_full_v2` conversion was superseded, not admitted.
The complete v3 avoids generating unused per-frame observation arrays. Its
loader samples current patches from static scenes after selecting the actual
native motion window. All three data sources retain explicit provenance.

Run: `mixed_velocity_50k_v1`, launcher PID3032672 at launch. Fresh-check GPUs
0 and1 are an NVLink pair. Tokenizer and Root train independently on one GPU
each, then Pose uses the pair through original Lightning DDP. Initial four-card
planning was revised when other users occupied cards5-7; no unrelated jobs
were stopped. All stages run50000 optimizer updates with native losses,
Adam5e-5, warmup10000/cosine2000000, gradient clip0.5. Global input batch256 is
held fixed, explicitly differing from paper batch256 per GPU. Sampling assigns
half to PARC and half to BONES, balancing BONES control categories. Normalized
real windows and standard native augmentations are used; no still-frame tiling.

Before the long run, native 20-update Tokenizer/Root/Pose training, DDP, frozen
dependency checks, strict bundle reconstruction, loader workers and fixed
validation all ran. The smoke bundle is not a quality result. The long run
records fixed validation loss every5000 updates, checkpoints, frozen dependencies,
and later a60-second rest/start-stop review from held-out BONES seeds. The
workflow exports reviewed artifacts privately to Drive. Local gameplay remains
a separate acceptance step; no automatic replacement of the live model occurs.

## Hardware Evidence

Same bundle, PARC dataset, frozen CPU batches, Torch2.5.1+cu124, FP32/high matmul,
batch256, ten warmup and fifty measured updates:

| Stage | A100 80GB PCIe Median | RTX4090D Median |
| --- | ---: | ---: |
| Root | 112.97 ms | 125.65 ms |
| Pose | 191.30 ms | 215.16 ms |

The common batch digest is
`037c310233d8728593defe05eaf36d9a58304f408dca5dd3475b3d0f31a27c26`.
These are short cached-input single-GPU measurements, not a multi-GPU speedup
or convergence guarantee. The dominant earlier CPU bottleneck was per-source
terrain query synchronization: on A100, unbatched Root median was472.53 ms.
The original root benchmarks before freezing batches had different input hashes
and should not be cited as strict paired device comparisons.

## Remote Access and Transfers

- A100: private-user at176.122.136.76 port12304, RSA from ${SSH_CONFIG_DIR}.
- g3: private-user at176.122.136.76 port12303; saved old-IP host key is checked through
  HostKeyAlias `[remote-host.example]:12303`.
- 4090D: private-user atremote-host.example port32297.
- Private Drive remote name: mb-transfer, folder `MotionBricks Transfer 20261010`.
  No public sharing links were created. OAuth tokens are never in repository
  artifacts or logs. Server configs live in mode0600 private task directories.
  Local rclone is under `%LOCALAPPDATA%/MotionBricks/tools`; its config is under
  `%LOCALAPPDATA%/MotionBricks/drive` with access restricted to the current user.
  The intermediate OAuth output was removed after verified configuration.
- 4090D's existing proxy17890 failed; task rclone commands use working localhost
  HTTP proxy27890. Global proxy configuration was not changed.
- A100 system partition was nearly full; environment/data/cache/checkpoints
  were placed on data5. Existing unrelated data/environments were untouched.
- A100 common user data paths had MotionMillion/Vicon and vision datasets; no
  matching BONES/PARC manifest was found there. Project inputs arrived via Drive
  and SHA256 identities were checked. This was not an exhaustive whole-server
  permission-bypassing search.

Remaining acceptance: eliminate residual local timing spikes; demonstrate
long stationary and stop/restart stability with the new trained pipeline; then
validate terrain traversal and bring accepted weights to the local browser.
