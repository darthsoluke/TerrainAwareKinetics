# 3D Perception And Framewise Kinematic Locomotion

Priority revised on2026-10-21 after the user's comparison with PFNN:
complete3D environment input first, then a genuinely reactive framewise motion
controller, with PFNN-level locomotion quality as the initial benchmark.
Keep the trained SOMA viewer on28944 available. This is a design plan; no new
training or runtime model replacement is launched by this review.

## Findings From The Implementations

Current scene input already samples three3D regions, each8x8x8, and encodes
distance, normal, location and validity. However, production `scene_field`
constructs those observations from a single-valued heightfield, represented as
vertical solid prisms. The analytic box implementation is currently for geometry
tests. Sampling in3D does not supply missing ceiling/overhang/multilevel geometry.

`PredictedRootObservation` already resamples the corridor along predicted Root
positions for Pose/decoder. This useful connection should survive the redesign.
The current dual-horizon deployment fixes execution to8tokens,4Pose passes and
4-frame refresh. Thirty displayed frames per second is distinct from receiving
new geometry and responding to input on each motion update.

PFNN's published demo calls its phase-conditioned regression in the frame loop,
using previous joint position/velocity and trajectory terrain samples. It also
enables IK by default, reads predicted contact weights, retains foot contact
positions, raises heel/toe targets to sampled terrain and solves the leg chain.
The reference's final visual contact quality therefore includes those stages.

Source locations:

- `motionbricks/data/unified_scene_field.py`: field_from_heightfield,
  VolumetricEgoSampler and its three regional lattices.
- `motionbricks/data/joint_sdf_observation.py`: HeightfieldPrismSDF.
- `motionbricks/data/unified_terrain.py`: SharedTerrainEncoder and scene_field.
- `motionbricks/data/root_path_terrain.py`: predicted-path resampling.
- `motionbricks/data/dual_horizon_execution.py`: four-frame refresh contract.
- [PFNN published source mirror](https://github.com/sreyafrancis/PFNN/blob/master/demo/pfnn.cpp):
  default IK line78, trajectory samples line1649, regression line1664,
  IK line1726, foot locking line1907, frame loop line2674.
- [PFNN author project](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control).

## 1. Complete The Geometry Input

Start with known scene geometry. A single scene description supplies rendering,
neural queries and independent intersection measurements. Include ground,
vertical faces, ceilings, overhangs and stacked traversable surfaces.

Expose distance/nearest surface, normal, validity and surface identity. Define
inside/outside semantics explicitly for open meshes. Keep a bounded query budget
with broad body/path coverage and denser queries near feet, hands, head and
prospective contacts. Query future geometry along predicted Root/body motion.

The first distinguishing examples are a low ceiling over uneven ground, a side
wall beside a step, and a bridge with usable space both above and below. Display
the actual neural inputs so geometry at both levels can be inspected. Then
measure query and encoder cost on the deployment machine.

## 2. Make One-Frame Motion Advancement The Controller Interface

Each update consumes the latest joystick intent, the last displayed pose and
velocity, and current3D observations. It predicts the next Root/pose/contact
state while retaining a future trajectory for anticipation. Contact/gait state
must remain continuous across changes in direction and planning horizon.

Profile whether the current MB execution can meet this interface within the
frame budget. If its multi-pass sequence generation remains too expensive,
retain MB as a longer-horizon prior or training teacher and train a compact
framewise policy. The fast policy must generate the actual next-frame motion;
changing playback cadence alone does not provide that feedback loop.

Target at least30Hz motion/control and30fps presentation, with60Hz control as
the next performance target. Measure query, encoder, motion generation and
presentation separately; retain input-to-visible-response P95<=100ms as the
interaction goal. Avoid committing an unnecessary long prefix to execution.

## 3. Train Geometry And Contact Use Together

After zero-training geometry/interface work, prepare an explicit training plan
and budget for the new inputs/controller. Use scene-paired motion, contact timing,
swing clearance, Root/pose continuity and geometric intersection measurements.
Evaluate contact release and establishment during direction changes, as well as
steady locomotion. A changed encoder/codebook requires regenerated Pose labels
after freezing the new codec.

First reach the walking/running/stair/slope benchmark. Then extend the same3D
interface to crouching, crawling, supported traversal and climbing, including
transitions that cannot return to standing. Seed motions and environment
pairings must cover these capabilities.

## 4. Compare Against The Actual PFNN System

Use matched maps, speed intents and joystick traces. Report penetration,
stance sliding, foot-clearance timing, turning/stopping response, frame cost and
dropped playback frames. Keep PFNN IK-enabled and IK-disabled results separately
so the comparison exposes the contribution of learned motion and contact handling.

The current raw-output MB baseline remains available. Any future contact-solving
component is an explicit method choice and ablation.

Immediate scope:3D geometry/query prototype and PFNN baseline accounting.
The live viewer and trained weights remain unchanged during this planning review.
