# BONES-SEED Locomotion Training Plan

Planning date: 2026-10-04. Status: implementation started; bounded SOMA tokenizer and root pilots completed. Quality and deployment gates remain open. See `bones_seed_training_execution_20261004.md` for measured results.

Final endpoint: trained SOMA inference controlled from UE 5.7 PIE on the local RTX 3080. The implementation/deployment milestones and audited existing UE gaps are defined in `soma_ue_end_to_end_route.md`. V5 semantic review exposed acrobatic/interaction contamination; use corrected V7 for the next pilot, not V5 as an approved training corpus.

## V1 Contract

- Dataset: BONES-SEED SOMA Uniform, starting with verified planar locomotion clips from the 71,132 original BVHs already inventoried on taku-g3.
- Control: desired root trajectory and heading, together with the character's observed motion history. Optional pose/style constraints remain compatible with the official architecture but are not required for the root-only control evaluation.
- Output: human root/body motion and four supervised foot-contact channels at a 60 fps motion sampling rate.
- Runtime: purely kinematic animation, retargeted to UE's standard human skeleton; ground queries, foot locking and IK provide final geometric correction.
- No heightmap input for V1. PARC, PFNN, LAFAN1 and MotionPersonaX are optional later additions, not prerequisites for this BONES-only experiment.
- Performance target: single character, batch size 1, at least 60 complete backend updates per second on the local RTX 3080 10 GB. Sequence-generation throughput and interpolated playback are separate metrics.

V1 prioritizes idle, starts/stops, forward walk/run, curved paths, turns, backward motion and strafing where source coverage supports them. Crouching and jumps follow only after ordinary locomotion passes validation. Clips involving stairs, climbing, seating, terrain changes or unknown supports are excluded from planar contact supervision unless their geometry is independently established.

## Evidence and Reuse Boundary

The local official preview exposes `train_vqvae.py`, `train_root.py` and `train_pose.py`. They use synthetic tensors and G1-derived configuration, disable checkpointing/logging in the example trainers, and do not provide a complete SOMA training run. The dataset guide explicitly recommends adapting the data/representation layer for non-G1 motion.

Reuse the official multi-head VQ-VAE, Root Transformer, Pose Transformer, feature math where applicable, masking/sampling and inference structure. Implement only the required SOMA representation/data adapter, production training configuration, contact/geometric validation and deployment interface. G1 checkpoints, input dimensions, normalization statistics and foot offsets are not assumed compatible with the human model.

Verified local configuration: Pose Transformer width 1024, 16 layers, 16 attention heads; Root Transformer width 512 with 3 shared and 3 root-token layers. The released configs use 30 fps and four motion frames per token with up to 16 tokens. At 60 fps, the same frame count covers half as much time. Temporal windows must therefore be specified and checked in seconds as well as frames.

The existing VQ-VAE loss includes motion reconstruction, commitment, joint velocity and contact-weighted skating terms. Contact channels already exist. Explicitly validate their reconstruction and avoid allowing penetration/skating penalties to disappear just because the model predicts no contact. The current predicted-contact skating weighting requires particular scrutiny.

Sources: [official dataset integration guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/motionbricks/docs/adding_your_own_dataset.md), [MotionBricks paper](https://arxiv.org/html/2604.24833v1), [BONES-SEED dataset card](https://huggingface.co/datasets/bones-studio/seed).

## Execution Sequence

| Stage | Work | Required evidence before proceeding |
| --- | --- | --- |
| 1. Dataset and skeleton contract | Obtain metadata/temporal labels and SOMA rig companions; identify usable locomotion; freeze skeleton and split definitions | Reproducible manifest, explicit joint/coordinate contract, correct replay and contact inspection |
| 2. Real-data adapter and speed screening | Connect real motion features to the official networks; benchmark representative full inference shapes on RTX 3080 | Finite tensors, working gradients, validated motion conversion and a candidate meeting the latency budget |
| 3. Small pilot | Debug on 128-256 balanced training clips, then pilot on roughly 1,000-3,000 eligible clips if available | Reconstruction, contacts and closed-loop behavior improve on held-out data; measured training cost |
| 4. BONES locomotion training | Train tokenizer and root model; freeze tokenizer and train pose model on the accepted locomotion corpus | Versioned best checkpoints, held-out metrics and reproducible evaluation videos |
| 5. Interactive evaluation | Run the trained human backend with scripted commands and UE retargeting/ground correction | Sustained 60 Hz updates, responsive commands, stable contacts and no persistent visible foot penetration in the acceptance scene |

Speed screening is not a motion-quality result. Re-run it on the trained/exported model and actual deployment path before final acceptance.

## Stage 1: Data Preparation

1. Inventory every candidate file against metadata, record hashes and provenance, validate all frame counts, channel counts, finite values, hierarchy and declared fps. Keep original data immutable and write derived shards under our own project directory.
2. Resolve `Root` versus `Hips`, joint names/order/parents, rest offsets/orientations, rotation order, length units, heading and up axis from the SOMA rig and actual BVHs. Preserve the full original rig; declare a body-only training subset and its mapping explicitly. Do not assume a previously mentioned 27-joint layout or a G1 layout.
3. Validate FK and frame transforms on representative clips, including turns and mirrored motions. Verify left/right correspondence, angular wraparound and sole offsets. Replay the converted human motion before attempting neural reconstruction.
4. Resample using actual source timestamps to 60 fps. Handle rotations as rotations, apply appropriate downsampling treatment, recompute velocities in physical units and retain root vertical motion. Specify all thresholds in meters and seconds.
5. Estimate one fixed support plane per accepted planar segment from reliable support frames. Reject or flag uncertain segments. Do not subtract a per-frame/window minimum foot height: that can erase jumps and fabricate ground beneath airborne feet.
6. Build heel/toe or equivalent calibrated sole sampling points. Derive contact labels from height and speed with temporal consistency and confidence. Manually inspect a small balanced annotation set for contact validation; leave ambiguous frames out of contact-specific losses.
7. Split original recordings before making windows or augmentation, targeting approximately 90/5/5 train/validation/test by eligible data while preserving group boundaries. Keep takes, mirrors and duplicates together; include an actor-held-out evaluation where the grouping permits. Compute normalization from training data only.
8. Record behavior coverage by duration, speed and turning rate. Balance sampling so idle/straight walking do not overwhelm starts, stops, turns, backward and lateral movement. Generate mirrored training augmentation only after skeleton and metadata checks.

Deliverables: frozen skeleton/representation specification, provenance and split manifests, 60 fps motion shards, normalization statistics, contact confidence labels, quality report and replay examples.

## Stage 2: Model and Runtime Budget

Keep the official architecture as the reference. Screen a compact Pose Transformer (width 512, 8 layers, 8 attention heads) and, if needed for quality, a medium candidate (width 768, 12 layers, 12 heads). These are proposed configurations, not measured winners. Start with the official root/tokenizer structure and reduce components only when profiling identifies a bottleneck.

Benchmark 64- and 128-frame shapes as appropriate to the selected planning horizon. Confirm the official past/future constraint slots and masks before changing history length; preserve physical time coverage explicitly. Prefer the official single-pass pose generation path, fixed deployment shapes and validated FP16 inference. Compilation/export optimizations must preserve output quality and contacts.

Record warm and cold latency separately. Warm tests use batch size 1, synchronized GPU measurements plus CPU wall-clock timing, and at least 10,000 updates after warm-up. Report p50/p95/p99/max latency, missed deadlines, CPU overhead, peak incremental GPU memory and numerical differences against the reference precision.

Provisional budget: model computation around 8-10 ms, full backend request-to-available-result p99 at or below 16.67 ms. Report deadline misses explicitly rather than hiding them in averages. The complete UE scene must also sustain the agreed 60 fps workload on the same GPU; backend-only results do not certify rendering performance. Freeze the scene, viewport resolution and quality settings when establishing that benchmark.

Measure two distinct control delays: backend request-to-result and user-input-to-first-visible-response. Record any rollout/buffer delay. Each accepted update must consume current input and provide a new usable prediction; decoding a long sequence and reporting its frame count as throughput does not establish 60 Hz control.

## Stages 3-4: Training Order

### Human Tokenizer

Train the official VQ-VAE on the accepted human representation. Retain body rotations/positions, root-related features and contact channels with validated physical-unit losses. Evaluate pose/FK reconstruction, velocity, contact reconstruction, skating and codebook usage. Add an explicit sole-to-ground penetration penalty for all valid-plane frames, including swing frames; use trusted contact masks for support constraints so contact suppression cannot evade the loss.

Tokenizer reconstruction sets an upper bound on subsequent generation quality. Do not proceed to expensive pose training if it already damages foot contacts, turns, or body orientation. After the pilot passes, the tokenizer may include a broader clean BONES body-motion subset if this improves reconstruction without weakening locomotion/contact quality.

### Root Module

Train the official Root Transformer independently of the tokenizer, using history and root constraints consistent with runtime. Preserve root timing/trajectory supervision. Compare prescribed root trajectories with the official root-refinement path to determine which best follows interactive commands while staying plausible. Keep trajectory deviation visible in evaluation rather than silently correcting it away.

### Pose Transformer

Freeze the validated tokenizer and train masked pose-token prediction conditioned on trajectory and observed history. Audit every conditioning mask: the root-only control experiment must not receive future ground-truth body poses during training/evaluation when those poses will not be available in deployment. Retain optional keyframe conditioning as a separately evaluated mode.

Start with clean trajectory conditioning, then introduce plausible command changes/trajectory perturbations and evaluate autoregressive rollouts using generated history. Do not blindly keep an unchanged body target under a substantially altered root trajectory. Use coherent augmentation or separate consistency objectives and compare against the clean baseline.

The first 1,000-step timing/learning run estimates throughput, memory and time-to-checkpoint. Choose the full step budget and validation interval from those measurements and learning curves rather than copying the original multi-node, two-million-step configuration. Enable local logging, validation, resumable checkpoints, optimizer/scheduler state and best-model selection before any long run.

## Evaluation and Acceptance

Use held-out real motion for reconstruction and contact quality; use scripted interactive commands for control quality. Include stationary starts, gradual and abrupt stops, acceleration/deceleration, 90/180-degree turns within feasible limits, circles, alternating turns, strafing, backward movement and repeated rapid input changes. Test sustained rollouts rather than isolated short clips.

Report root tracking and heading error, joint/FK reconstruction, speed/acceleration continuity, contact precision/recall against the inspected labels, contact-transition flicker, support-foot speed, stance drift, sole penetration depth/duration and airborne clearance. A small manually checked set is needed because agreement with automatically generated contact labels alone is insufficient.

Report model outputs and post-correction outputs separately. In UE, measure foot placement after retargeting using target-character sole geometry, with foot locking, ground queries and bounded pelvis/IK correction. Record correction magnitude so a poor model cannot appear successful solely because IK performs large repairs. Reconcile root authority between the controller and backend to avoid double movement.

Final pass requires the RTX 3080 timing target in the fixed UE scene, no systematic foot sliding or persistent visible ground penetration, and consistent control response across the accepted behavior set. Establish numerical tolerances from source/rig accuracy during the pilot and lock them before comparing full training runs.

## Compute and Next Deliverable

At the planning check, taku-g3 exposes six RTX 4090 24 GB GPUs; GPU 5 was near idle and the others were busy. Availability is a snapshot, not a reservation. Use an available single GPU for the pilot, rechecking allocation at launch, and scale only after the pipeline and data are correct. The local RTX 3080 has 10 GB VRAM and is the deployment acceptance device.

Use a dedicated conda environment and pinned source/dependency versions for training. Store manifests/configs/checkpoints in our own server project directory and stage derived data on verified writable storage with adequate capacity. Preserve source dataset references to avoid duplicating the 138 GiB BVH corpus unnecessarily.

The immediate deliverable is a reviewed SOMA locomotion dataset plus a working real-data training adapter and RTX 3080 latency report. Long training starts only after those gates and the small pilot pass.
