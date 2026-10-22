# SOMA MotionBricks to UE: End-to-End Route

Decision date: 2026-10-04. This is the implementation route, not a claim that human inference already works in PIE.

Method review: `m1_paper_review.md` records paper/source mismatches and the corrected diagnostic protocol. MB is the current reference, not an exclusive final backend requirement. Other open methods may replace individual modules or serve as independent baselines after passing the same SOMA and UE evaluation contract.

User clarification: prioritize MB + SOMA flat-ground locomotion before terrain-aware extensions. Preserve editor-authored skeleton/pose keyframes for real-time in-betweening. Root-only control is one evaluation mode: authored future pose constraints are legitimate runtime inputs, unlike privileged future poses from a held-out recording. Other open methods remain references for bounded improvements.

## Endpoint and Scope

Success is a controllable human character in the existing UE 5.7 project at `F:/dev/GASP_57`, running trained human MotionBricks inference on the local RTX 3080 10 GB. Server GPUs train models; they are not a remote inference dependency.

- V1: upright planar idle, walking, running, starts/stops, curves/turns, backward movement and strafing where data coverage is verified.
- Human representation: official BONES-SEED SOMA77 converted through official Kimodo to official SOMA30, standard reference pose, meters, Y up, Z forward, four heel/toe contact channels.
- UE displays the SOMA source alongside a Manny target during diagnosis. Manny is a retargeting destination, not the model's training skeleton.
- Pure kinematics: no MuJoCo, robot tracking policy, dynamics training or G1 weights in the human path. UE collision queries and capsule movement still prevent world-geometry overlap.
- No heightmap, slopes, stairs, jumps, crouching, combat, interactions or multiplayer in the first acceptance scope. Do not expose unsupported actions as working features.
- All ordinary body locomotion is produced by MB. UE supplies input, camera, collision, retargeting and bounded geometric corrections. A blend-space or procedural walk cycle is not an acceptable substitute.

## Present Implementation and Gaps

Inspected implementation: `F:/dev/GASP_57/Plugins/MotionBricksGroot/Source`.

| Area | Already present | Required for SOMA V1 |
| --- | --- | --- |
| Input/framework | GameMode, possessed Pawn, PlayerController, Enhanced Input assets and release/cancel bindings; camera Y inversion is configurable | Human-specific capability/configuration; validate possession, focus loss and input cancellation |
| Source identity | Checkpoint-bound hierarchy manifests and identity checks | SOMA30 hierarchy, standard pose, normalization and trained checkpoint bindings; separate identity from quality approval |
| Retargeting | Native `FIKRetargetProcessor`, retarget hierarchy DataAsset and editor authoring tools | SOMA-to-Manny source assets/reference alignment, explicit unmapped joints and sole calibration |
| Animation | SkeletalMesh, AnimInstance proxy pose snapshots and inertialization | Atomic human pose/contact snapshots and correction metrics; no socket or UObject access in worker-thread evaluation |
| Movement | Custom PawnMovementComponent sweeps and slides model root deltas | Ground-aware CharacterMovement integration and actual-collision-result feedback; no duplicate root movement |
| Transport/control | Nonblocking loopback TCP JSON, future-context chunks and prefetch | Versioned human request/result schema, four contacts, current-history feedback, command-age and sequence rejection, low-latency fixed-shape requests |
| Models | Human tokenizer/root 1,000-step pipeline pilots | Clean-data tokenizer quality, root control quality, trained Pose Transformer and closed-loop acceptance |
| Performance | Untrained network-only screening on RTX 3080 | Trained complete backend plus the actual UE scene on the same GPU |

The current default Pawn hierarchy is `g1skel34`. Its 0.15-second command-request spacing, future playback context and long-chunk interpolation do not establish 60 current-input inference updates per second. The legacy shared-memory schema also assumes 34 bones/qpos36. Do not silently relabel any of these as a trained SOMA backend.

Current corrected data: V7 has 191 accepted clips / 64 recording groups (152 train, 19 validation, 20 test). Structural, finite-feature, automatic semantic-policy and group/hash isolation checks pass. Manual semantics/contacts and actor-held-out checks remain unapproved; validation/test lack lateral examples and test also lacks idle. A contact-aware 1,000-step tokenizer diagnostic completed but failed the reconstruction screen; a distinct bounded convergence protocol preserves its geometry weights and architecture. See `bones_seed_training_execution_20261004.md` for actual run status.

M0 implementation now exists in the separate `/MotionBricksGroot/SomaReplay/L_SomaSourceReplay` map, using a new Kimodo-standard hierarchy rather than the older ARDY contracts. Four real V7 fixtures contain 1,358 frames. Native full-frame source FK agrees with Python within 0.000073 cm, target bone lengths remain unchanged, and the reference validation passes facing/chain/basis/rejection checks. A SkeletalMesh/AnimInstance proxy renders Manny alongside the source bones in PIE. This is recorded source replay, not generated locomotion. See `soma_source_replay.md` for launcher, reports and the remaining review boundary.

## Runtime Responsibilities

1. `GameMode` spawns the human `ACharacter`; `PlayerController` owns possession, camera and LocalPlayer input-context lifecycle. Enhanced Input collects current movement/facing/gait intent and clears it on release, cancellation or lost possession.
2. A character-owned MB control component combines current intent with the actually applied SOMA history to construct a feasible planar root trajectory. Desired velocity is intent, not a second independently applied capsule movement.
3. A persistent backend owns model weights, per-character recurrent/history state and warmed fixed-shape tensors. Root Transformer refines the intended trajectory; Pose Transformer predicts tokens; the frozen official tokenizer decoder and FK produce SOMA motion and contacts.
4. The backend returns motion root deltas, SOMA local/global rotations as explicitly declared, contacts, sequence identity and timing. The UE source skeleton consumes the same rotations used by the reference evaluator.
5. Native IK Rig/IK Retargeter maps SOMA chains to Manny. An AnimInstance proxy snapshots the accepted pose on the game thread; the AnimGraph consumes value buffers and applies bounded transition inertialization and contact-aware foot correction.
6. The human CharacterMovement adapter consumes the MB root delta through an engine root-motion integration point and performs capsule sweep/floor handling. It is the only actor-transform authority. No simultaneous `SetActorLocation`, ordinary input acceleration and second animation-root extraction are allowed.
7. The actual capsule displacement/rotation and bounded skeletal corrections are fed back into the next history. A blocked wall must not leave the model integrating an imaginary unblocked trajectory indefinitely.

The new human Character path will coexist with the old debug Pawn until it passes the same source-replay/collision tests. Integrate with the installed engine root-motion pipeline in a focused test before attempting a broad controller rewrite. Do not add physics-based body control under the name of kinematics.

## Skeleton and Artifact Contract

Freeze and validate a complete human bundle, not just a checkpoint:

- Official SOMA joint names/order/parents and standard local reference transforms; canonical skeleton fingerprint and Kimodo revision.
- Source units, axis basis, root/heading convention, rotation space and quaternion storage order.
- Tokenizer, Root and Pose architecture/configuration and checkpoint SHA256 values; 60 Hz sampling; four frames per token; fixed 64/128-frame runtime shapes as actually supported and benchmarked.
- Train-only mean/std, their digest, preprocessing revision, corpus/split manifest digests, contact channels/sole offsets and condition-mask definition.
- UE source IK Rig, target IK Rig, retarget pose/profile, Manny asset identity and target sole calibration.
- Explicit states: source replay, reconstruction review, experimental generation, accepted generation. A verified checkpoint identity does not mean its motion quality passed.

Use the official SOMA hierarchy as the source. Do not make a second invented 27-joint layout, reinterpret G1 hinge angles, or silently change reference poses. SOMA77 export/skin visualization remains available through official Kimodo; hand/facial detail not predicted by the SOMA30 backend remains reference/default, not a claimed generated capability.

The existing coordinate candidate is position `UE_cm = (source_z, -source_x, source_y) * 100`. The reflected basis matrix must conjugate rotations, including reference transforms. Validate it with standard pose, left/right limbs, three axis translations, a 90-degree yaw and isolated joint rotations before accepting it. Converting quaternion storage order alone is insufficient.

## Data and Learning Route

1. Correct the planar corpus before expanding training. V5 reconstruction review exposed `turn_handstand_270_R_002__A305` and `walk_forward_grab_relax_002__A006`; these are not ordinary upright locomotion. V5 is retained as a failed diagnostic baseline, not an approved dataset.
2. V6 added interaction/acrobatic exclusions and upright geometry checks, but its manifest still included non-basic gestures/styles. V7 adds a conservative filename-token allowlist, retaining only ordinary locomotion names. Geometry checks compare hip height to inferred fixed support and head-above-hips orientation. These are rejection heuristics, not certified semantic labels. Keep ambiguous supports/actions rejected or separately reviewed.
3. Freeze recording-group and duplicate isolation, train-only window/yaw statistics, behavior-duration/speed/turn coverage and manually reviewed contact examples. Inspect starts/stops and turns, not just the first frame. Actor-held-out evaluation remains an independent required check.
4. Train the official VQ-VAE. Split errors into absolute joint error, root-relative body error, root XZ drift, hip height, local rotations, FK-versus-position features, trusted contact skating and codebook usage. Report raw output before corrections.
5. Train Root on the same history and trajectory constraints available at runtime. In V1, future hip height comes from observed history rather than future ground-truth height; future body keyframes are unavailable.
6. Freeze an accepted tokenizer, then train the compact official Pose Transformer with observed history plus root control. No future target body poses in the root-only experiment. Evaluate using predicted root trajectories as well as clean target roots; report them separately.
7. Evaluate generated-history rollouts under start/stop, acceleration, feasible sharp turns and alternating commands. Any trajectory perturbation must have coherent motion supervision; do not retain an unchanged pose target for a substantially changed path.

Tokenizer oracle-root reconstruction is an error diagnostic, not a control score. Low penetration alone can hide floating feet or a biased pelvis; additionally measure contact clearance, stance velocity, lift timing and correction magnitude. Automatic contact F1 is not manual ground-truth accuracy.

Learning permission and deployment acceptance are separate. Content-bound source geometry/statistics/split/gradient/token-decode checks permit a frozen tokenizer-only convergence pilot on the existing small corpus. Missing manual or generalization evidence is recorded, not declared passed. End-to-end trained UE/3080 acceptance must not be a circular prerequisite for learning the tokenizer. Full-corpus quality training and later Root/Pose stages still require their own approved data and model gates.

## Milestones: UE Evidence at Every Stage

| Milestone | Deliverable | Exit Evidence |
| --- | --- | --- |
| M0: source replay in PIE | Versioned SOMA source hierarchy, recorded real idle/walk/run/turn fixtures and SOMA-to-Manny retarget assets | Standard pose/axis/left-right/FK tests; same-time source motion matches Python; source and retargeted poses visible; no G1 backend |
| M1: reconstruction in PIE | Same fixtures reconstructed by the tokenizer, with raw/retargeted/corrected modes | Numerical reconstruction report and PIE video; contacts, stance drift and target bone lengths; tokenizer gate before Pose training |
| M2: trajectory/control route | Root prediction with current-input request identity, observed-history snapshots and movement/collision feedback | Trajectory tracking, no double movement, no stale results after rapid changes, wall-stop and resume behavior |
| M3: generated human in PIE | Trained Pose + Root + tokenizer, no ground-truth future poses | Sustained generated-history rollouts; all V1 behaviors and transitions; actual Enhanced Input control rather than clip replay |
| M4: deployable runtime | Warmed local human bundle with validated optimized inference and bounded queues | Complete current-request-to-result timing, reference/export parity, no blocking game-thread inference and lifecycle tests |
| M5: acceptance | Fixed 1080p UE flat test map, single character, same RTX 3080 for rendering and inference | Agreed frame/update latency and contact-quality checks for sustained scripted/manual control; logs, recordings and bundle identity saved |

M0 can proceed with real recorded motion while tokenizer training continues. Replay assets must be labeled as replay and must not fake missing Root/Pose checkpoint hashes to satisfy the old verified-generation importer. Add a replay-only source contract or separate loader with a distinct state.

M1 and M2 isolate geometry and control problems before combining all learned modules. Do not wait until a large training run finishes to discover UE coordinate, movement or reference-pose mismatches.

## Deployment Choice and Scheduling

First establish numerical reference parity with a local persistent PyTorch process, controlled from UE. It loads/warmups once and never creates a new process or reloads weights per frame. This is UE-integrated inference, not an engine-native model yet.

For live control, define a separate SOMA binary protocol rather than modifying the G1 qpos schema in place. Include session/model/skeleton IDs, command sequence, echoed UE request timestamp, observed-history sequence, shape, contacts and valid motion timestamps. Stage slow JSON fixtures only for offline debugging. Scope shared objects to the PIE instance; keep bounded double-buffered snapshots and a latest-command mailbox.

Maintain one GPU inference in flight per character. Coalesce pending intent, reject stale sessions/results, and report command age. Plan 64/128 frames but commit a short prefix and replan against the latest applied history. Long-horizon generation and interpolated display are not new inference updates. Evaluate the timing of the first affected rendered pose, not just packet arrival.

Evaluate a fixed-shape export early, once a candidate can be reconstructed correctly. Prefer the project's existing NNE integration if its installed GPU runtime can create and execute all required operators. Benchmark actual model creation/inference and parity; importing an ONNX asset or enabling a plugin is not proof of support. If NNE cannot support the model/export, retain the working local reference backend and evaluate an isolated native runtime behind the same interface. Do not assume NNE, ONNX, TensorRT, FP16 or a successful trace is automatically usable in production.

Network I/O, parsing and model execution are off the game thread. UObject access, ground queries, component lifecycle and publication of accepted results stay on the game thread. Animation worker threads consume immutable value snapshots. Shutdown/PIE stop cancels outstanding work and closes only this instance's resources.

## Acceptance Protocol

Numerical candidate thresholds are to be locked before comparing the next quality experiment; the failed V5 pilot does not meet a motion-quality threshold. Review representative and worst cases, and keep recording-held-out and actor-held-out results distinct.

For final timing, freeze scene, 1920x1080 viewport, rendering quality, hardware, export precision, model IDs, horizon and character count. Warm up separately, then record at least 10,000 updates and a sustained 10-minute control run.

- Required target: at least 60 complete updates/second consuming current input, not 60 playback samples of an old plan.
- Backend command-to-usable-result budget: 16.67 ms. Report p50/p95/p99/max and every deadline miss. A p99 pass with misses is not strict sustained-deadline acceptance.
- Record UE game/render/GPU frame times and inference under concurrent rendering; backend-only screening does not certify scene FPS.
- Provisional input-to-first-visible-body-response target: p99 <= 50 ms, measured with sequence/timestamp instrumentation. Camera response is measured separately.
- Motion checks: stable idle; natural starts/stops and feasible turns; no persistent model/retarget ground penetration, floating support feet or slide; no accumulating drift; bounded IK/pelvis corrections.
- Fault checks: backend unavailable/restart, focus loss, unpossession, repeated PIE start/stop, stale packets, invalid skeleton/stats identity, collision blocked/unblocked and dropped deadlines. Hold a valid pose or stop gracefully; fallback is not counted as successful MB generation.

Preserve three measurements separately: raw SOMA output, retargeted Manny output and final corrected UE pose. Save correction size so that IK cannot conceal a poor model.

## Cleanup Boundary

User authorized cleanup of obsolete experiments. Delete only verified generated/unreferenced material after inspecting paths and dependencies. Keep the official MB/Kimodo source, raw dataset, active SOMA pipeline, current contracts/checkpoints, focused tests and reproducible failure evidence.

Cleanup performed: seven untracked root-level `.obj` scratch files were removed after resolved-path and tracked-file checks: `CkptProbe`, `G1QposConverter`, `MotionBricksBackbones`, `MotionBricksGrootNative`, `MotionBricksRealtimeInference`, `MotionRepG1` and `VqvaePoseDecoder`. No UE assets or source files were deleted.

Do not delete UE assets only because their names contain G1: current defaults/scripts/maps still reference some of them. After M0-M3 replace the active route, use UE Asset Registry/reference inspection and a tested replacement map before removing legacy G1/MuJoCo/manual-pose prototypes. Record deletions and replacements; do not migrate obsolete code into the human path merely to preserve it.

## Primary References

- [Official MB inference](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/motionbricks/motionbricks/motion_backbone/inference/motion_inference.py): root and pose separation and selectable final root origin.
- [Official Kimodo SOMA skeletons](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/skeleton.html): SOMA30 internal and SOMA77 output/reference tooling.
- [Epic UE 5.7 IK Retargeting](https://dev.epicgames.com/documentation/en-us/unreal-engine/ik-rig-animation-retargeting-in-unreal-engine?application_version=5.7): chain/reference-pose retargeting rather than manual bone-name substitution.
- [Epic UE 5.7 NNE Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/neural-network-engine-overview-in-unreal-engine?application_version=5.7): runtime capabilities, tensor lifetime, model creation checks and GPU/rendering contention.
