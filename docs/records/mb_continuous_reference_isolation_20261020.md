# Continuous MB Reference Isolation

## Result

Follow-up intervention: [Retarget Repair Comparison](mb_g1_retarget_repair_20261020.md).
The original source/reference/physics artifacts below remain unchanged. Three
new candidates reuse this source and initial state. Geometry is improved, but
remaining gait/drift, solve latency and a stress failure block live deployment.

The requested three-stage zero-training experiment is complete. One continuous
16.8second MB reference was generated, retargeted offline into one fixed G1
reference, and tracked for16seconds by the frozen ProtoMotions policy. No online
MB requests, reference switching, actuator residual, filter or automatic reset
occurred during physical tracking.

The clearest reproducible defect is in the anatomical retargeting layer near a
straight leg. The later constrained reference also penetrates the ground at
some poses. Physical tracking adds motion deviations and world-Root drift.
Fixing reference feasibility is the next intervention; MB and tracker weights
remain frozen.

## Protocol

Case: `out/mb_sonic_e0_e3_20261018/continuous_reference_isolation_v1`.

- Existing64m fixed flat source, shared by MB queries, G1 geometry and MuJoCo.
- Requested source velocity XZ=[0,1.2]m/s, native heading0, RNG seed20261020.
- Native float32 CPU inference;21real AR calls,8tokens/4Posepasses each.
- Initial four-frame factual idle seed, then generated context. No executed or
  future true state was supplied to open-loop generation.
- All32raw decoder frames per call retained. The original4:28playback/context
  contract yields16.8seconds of real future; no smoothing, padding or tail hold.
- Source model tensor SHA256 remains
  `a6366a7e8522371be01d632efa6a952839a690a1749e14d6fb65819ff584b50c`.
- Current0.76sreference constraint windows were processed offline, with0.48s
  commits, into one continuous841sample50HzG1 track. The actor never switches
  between these windows: it reads the completed immutable array.
- Original G1 model, joint/motor bounds, implicit PD gains,1kHzphysics and50Hz
  policy. One explicit initial reset and100real warmup steps.

Generation took3.318s on CPU; offline retargeting2.877s; tracking9.574s. These are
offline throughput measurements, not browserFPS or interactive latency.

## Stage 1: MB Source

Root advances19.321m in16.8seconds: average1.150m/s against1.2m/s requested.
Maximum native Root frame step is4.90cm at30Hz. Height range is5.51cm; lateral
range is0.598m, so the path is continuous but not perfectly straight.

The source pose and Root remain finite and continuous across the native AR
boundaries. The videos show the original motion; no new smoothing was introduced.
This source was generated with native CPU inference, not the online CUDA/B2 loop.

## Stage 2: Retargeting

The raw anatomical mapping passes original joint/velocity bounds but amplifies
a small native rotation near a straight right leg:

| At3.3333->3.3667s | Per-frame change |
| --- | ---: |
| Native SOMA RightLeg global rotation | 1.280deg |
| Inferred anatomical thigh frame | 34.558deg |
| Raw G1 right hip yaw | 33.057deg |

The hip-knee-ankle bending-plane sine is0.02956 (about1.69deg away from a
collinear leg). `anatomical_frames` derives thigh twist from that plane; its
secondary direction becomes sensitive near extension. This is an actual frame
rotation amplification, not only an Euler-coordinate wrap.

Code locations: `motionbricks/mb_sonic/retarget.py:55` and`:65`.

The offline trajectory optimizer reduces this raw jump before the actor sees
the track. Its complete G1 reference passes all original joint and velocity
bounds; minimum joint margin is0.1628rad, maximum joint speed4.943rad/s.
It still has ground penetration: maximum2.328cm at14.2s/right foot, and34of841
poses exceed2cm. This check uses the original G1 collision geometry in the same
flat scene and performs no integration or pose correction.

Thus joint legality and smooth curves have not produced a contact-feasible G1
reference. Adding a motor-output filter would not correct either defect.

## Stage 3: Physical Tracking

The frozen tracker completes16seconds without a fall, unknown ground, motor
limit violation or actual joint overtravel. All795tick NPZ files and15900controlled
physical substeps pass the independent audit;100warmup steps are also retained.
Peak actual scene penetration is4.672mm. No external or Root-driving forces occur.

Actual forward speed averages0.956m/s over the controlled interval. At16seconds,
actual minus reference Root is[-2.155,+3.178,+0.006]m: planar error3.840m.
The actor's pinned input contract has no direct Root translation/linear-velocity
target, which is relevant to this drift.

Physical joints have acceleration RMS25.70rad/s2 versus8.59rad/s2 in the fixed
reference, with extra stance-phase knee flexion visible in the trace. Root-relative
G1 body-marker8-14Hz energy increases from4.68e-8m2in the reference to9.05e-6m2
in execution. The actor setpoints themselves encode motor feedback rather than
literal kinematic poses; actual joint/body motion is the important comparison.

The primary experiment actuates the constrained track, not the raw33degmapping
jump. The data identify an unstable mapping and incompatible reference contacts;
they do not isolate every physical deviation to a single mechanism.

## Next Fix

Replace the near-collinear marker-plane twist inference with rest-frame-aware
source rotation constraints and a trajectory IK formulation that handles G1's
DOFs and foot-ground geometry together. Preserve Root/joint continuity and the
original mechanical bounds. Test the replacement on this exact saved source,
same initial G1 state and same frozen tracker before returning to live WASD.
No MB regeneration, controller-output filtering or additional training is needed
for that controlled comparison.

## Inspect

Primary replay: `review/full_model_three_stage_lit.mp4` (480frames,30fps,16s).
Left: native SOMA30 skeleton. Middle: G1 reference rendered without physics.
Right: actual recorded G1 physics. Cameras follow each Root independently;
`review/root_paths.png` preserves the unmodified world-space paths.

Additional evidence:

- `generation/proposal.npz`,21raw block NPZ files,`generation/generation.json`
- `retarget_complete.json`,`reference_artifacts.jsonl`,`reference_numeric/`
- `tracking_v2/tracking.jsonl`,`tracking_v2/numeric/`,`tracking_v2/audit.json`
- `review/review.json`,`review/mapping_conditioning.json`,`review/reference_geometry.json`
- `review/root_paths.png`,`review/tracking_signals.png`,`review/mapping_conditioning.png`

The initial tracking harness failed to serialize its reset snapshot before any
physics steps; `track_failed.json` and the empty first trace remain preserved.
The corrected tracking_v2 reused the exact source and retarget artifacts.
Earlier skeleton-only renderings and the unlit model rendering are diagnostic
drafts; the named primary replay includes actual reset/warmup frames and proper
lighting. No physical results were changed during rendering.

All optimizer updates and new formal training exposures are0. This is a completed
isolation experiment, not stable-WASD or final terrain-quality approval. Existing
live services were not changed by this CPU-only generation/tracking experiment.
