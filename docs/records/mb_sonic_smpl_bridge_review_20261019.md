# Original SONIC SMPL-Mode Bridge Review

Date: 2026-10-19. Read-only decision review; no implementation or policy execution.
HF revision: `6733128a3d8a523b1418b06bca3cdf61c8b0987f`.

## Decision

**Technically credible as a separate, explicitly named SMPL-modality B1
candidate.** The original matched encoder already declares mode 2 with ten
step1 samples, offsets 0..0.18 s at 50 Hz. One genuine eight-token MB call has
0.8 s trusted future. The 0.28 s sum of lookahead and 0.10 s replan period alone
does not reserve processing/ready-age time. Applying the mainline's explicit
reserve arithmetic would require 0.40 s = 0.18 + 0.10 + 0.10 + 0.02, still within
one call's coverage. At source-state age below 0.10 s that needs at most 0.50 s
of the 0.8 s forecast. This can remove the second MB call and
the full 29-DOF future G1 reference solve from the candidate's necessary path.
The arithmetic gives 21 samples at 50 Hz including both endpoints; no candidate
runtime configuration or acceptance contract is created by this review.

It still needs calibrated human-rig transfer, correct root/marker conventions,
and **six meaningful G1 wrist references**. The original 1762-dimensional input
packing, 64-dimensional token, 994-dimensional decoder input and original
40,900,688-byte decoder remain; the separate low-latency decoder is 149,763,762
bytes. No switch to low-latency weights is required for this proposal.
[Pinned original observation config](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/observation_config.yaml).

Original G1-mode B0/B1, existing failures and the parent's CPU-history v3 work
remain the main line. Parent reports v2 reached 12 ticks/48 physical steps then
failed coverage after a discarded over-age ready plan, not a joint-limit error.
Changing modality or duration does not fix that publication/freshness path.
Actual decoder history and MB conditioning-history work also remain necessary.
Bridge calibration, modality inference, physics,
latency and motion quality: **NOT_RUN**. No real-time result is inferred here.

## Exact Consumer Semantics

| Original Mode-2 Input | Width / Original Encoder Slice |
|---|---|
| scalar mode followed by three zero fillers | 4 / [0:4], value `[2,0,0,0]` |
| ten 24-marker XYZ samples | 720 / [922:1642] |
| ten root-orientation 6D samples | 60 / [1642:1702] |
| ten six-wrist-angle samples | 60 / [1702:1762] |

Other modality sections are masked zero, including root-height and G1 full-body
reference fields. That is the declared encoder mask, not zero-filled human
poses or invented G1 leg angles. Actual decoder proprioception stays unchanged:
ten real 50 Hz samples of gyro, relative joint positions, velocities, previous
actions and body-frame gravity. The physical executor still drives the original
29 G1 body motors through the original PD/torque-limited MuJoCo path.

### Coordinates, Root And Heading

`GatherMotionSmplJointsMultiFrame` copies supplied XYZ values directly; it does
not rotate, center, scale or add root height. The upstream sender must provide
the already canonicalized features. Training's `smpl_joints_multi_future_local`
rotates each future frame's translation-free joint array by that frame's own
inverse corrected SMPL root rotation, not the first frame's rotation.

The official live producer follows:

```text
C = SOMA/SMPL Y-up -> Z-up rotation, (X,Y,Z)->(X,-Z,Y)
Qz = C * Q_smpl_pose_root
Qcorrected = Qz * conjugate([0.5,0.5,0.5,0.5])
marker_input(t) = inverse(Qcorrected(t)) * raw_translation_free_FK_joints(t)
```

Its raw FK retains the fixed target template origin. The inspected public sample
has pelvis point `(0.0031232606,-0.3514074683,0.0120365508)` m, identical to the
existing target rig's J0. It is **not a pelvis-zero joint array**. An adapter must
reproduce this feature convention; blindly subtracting SOMA Hips and sending 24
renamed points loses the target origin/bind convention.

`smpl_anchor_orientation_10frame_step1` is the normal body-orientation gatherer:
inverse actual G1 root quaternion times heading-aligned corrected human root
quaternion. Heading alignment uses initial actual yaw and inverse initial
reference yaw, plus declared operator yaw. Each matrix contributes its first two
columns, flattened row-wise. This is full-body-relative orientation, not the
v1.1 yaw-only variant. C++ transports the reference in `body_quat`/`body_quat_w`
with wxyz order; it must contain the corrected human reference orientation for
this modality, not an unrelated G1 retarget root.

Absolute human world translation and root height are not active mode-2 fields.
Local marker geometry encodes shape and articulation, including hip-to-foot
distances, but does not provide an absolute terrain-height command. Actual G1
Root still centers scene queries and remains separately logged. No reference or
physical Root reanchoring is introduced.

### Wrist And Transport Requirements

The active six angles use IsaacLab order `[23,24,25,26,27,28]`:
left roll, right roll, left pitch, right pitch, left yaw, right yaw. Their MuJoCo
indices are `[19,26,20,27,21,28]`. They are G1 hinge angles in radians, not SMPL
axis-angle elements. The official live sender redistributes human elbow swing
into wrist roll/yaw and applies side-dependent signs. A calibrated forearm/hand
rotation transfer and wrist-limit check remain necessary, even without leg,
waist or shoulder G1 future IK. Required wrist channels must not be defaulted to
zero merely because a transport version calls joint data optional.

Specifically, let `eL/eR` be elbow swing after twist decomposition about G1 Y,
and `wL/wR` the wrist rotation, both intrinsic XYZ Euler angles. The sender's
final six entries are `(eL.x+wL.x, -(eR.x+wR.x), wL.y, -wR.y,
eL.z+wL.z, eR.z+wR.z)` in the interleaved order above. These are the inspected
live producer's conventions, not a completed SOMA wrist calibration.

Official ZMQ v2/v3 selects mode 2; v3 provides meaningful wrist entries in its
29-column joint array. Other joint entries are inactive for this modality.
The transport additionally requires 21 SMPL body axis-angle rotations and body
quaternion/frame indices, although the selected encoder does not directly read
the 21-pose field. A direct Python modality packet can instead represent the
actual required marker/root/wrist signals without fabricating a full G1 qpos
reference. Existing mode-0-only builders cannot be reused unchanged.
[Official transport contract](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/zmq.md).

## Exact 24-Point Correspondence

The verified producer is **22 SMPL-X body joints plus indices 39 and 54**,
left/right thumb3. These are not classic SMPL24's final `left_hand/right_hand`
palm joints. The public sample and already bundled static rig reproduce this
selection at four common timestamps with max error 2.26835e-6 m and RMS
1.03585e-6 m. Producer identification: **PASS** for this bounded check.
[Official SMPL-X joint names](https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py).

The following are anatomical donors, not permission to copy positions without
target bind-frame/proportion calibration:

| Input Index | Verified Target Landmark | SOMA30 Donor |
|---:|---|---|
| 0 | pelvis | Hips |
| 1 | left_hip | LeftLeg |
| 2 | right_hip | RightLeg |
| 3 | spine1 | Spine1 |
| 4 | left_knee | LeftShin |
| 5 | right_knee | RightShin |
| 6 | spine2 | Spine2 |
| 7 | left_ankle | LeftFoot |
| 8 | right_ankle | RightFoot |
| 9 | spine3 | Chest |
| 10 | left_foot / forefoot | LeftToeBase |
| 11 | right_foot / forefoot | RightToeBase |
| 12 | neck | calibrated Neck1/Neck2 chain collapse |
| 13 | left_collar | LeftShoulder |
| 14 | right_collar | RightShoulder |
| 15 | head | Head, preserving the intervening neck chain |
| 16 | left_shoulder | LeftArm |
| 17 | right_shoulder | RightArm |
| 18 | left_elbow | LeftForeArm |
| 19 | right_elbow | RightForeArm |
| 20 | left_wrist | LeftHand |
| 21 | right_wrist | RightHand |
| 22 | left_thumb3, original rig index 39 | LeftHand orientation plus target thumb-chain offset |
| 23 | right_thumb3, original rig index 54 | RightHand orientation plus target thumb-chain offset |

SOMA `Hand` is the wrist, not a palm center. `HandMiddleEnd` and `HandThumbEnd`
are not interchangeable palm offsets. Here the extra markers are thumb3: the
observed fixed wrist-to-marker lengths are approximately 0.104753/0.104120 m.
Their locations need target hand-frame offsets; replacing them with SOMA's
farther finger endpoints changes learned input geometry. A conventional
SMPL24-palm producer would be a different contract and needs its own palm
anchors/proof, not a silent substitution.

### Genuine SOMA30 Transfer, Not Renamed Points

The genuine saved MB proposal contains 52 timestamped samples with both
`local_rot_mats[52,30,3,3]` and `global_rot_mats[52,30,3,3]`, plus actual decoded
root positions and native FK landmarks. These are the available pose inputs,
not angles reconstructed by zero-padding a 26-point public sample.

A calibrated bridge would transfer those native joint rotations using explicit
source/target bind axes, evaluate the target skeleton's own fixed metric bone
offsets, and run target FK. Source-global rotations plus calibrated frame
offsets define target global rotations; target locals follow from
`R_local_child = R_global_parent.T @ R_global_child`. Source and target frames
must be in the same coordinate system during calibration, with the Y-up/Z-up
conversion applied once at the declared producer boundary. Neutral-pose
reconstruction and independent walk residuals must verify the chosen offsets.

The three spine targets have direct native donors. The two native neck links
need an explicit chain collapse that retains the final head orientation; simply
dropping Neck2 is not equivalent. Target bone lengths/proportions come from the
identified static rig, not a time-varying position rescale. Mapping rotations
and evaluating that rig avoids G1's full future IK, but is still real human-rig
retargeting with measured calibration residuals required.

For target markers 22/23, the official body-only producer keeps its remaining
finger/face rotations at the defined rest pose and obtains thumb3 through the
target wrist/hand FK chain. The fixed target offsets are known geometry, not
measurements of independent thumb articulation. Native ThumbEnd/MiddleEnd can
orient a hand calibration frame; they are not copied into the target slots.
The public `pose_aa[...,72]` shape does not mean its last two axis-angle entries
are these thumb3 markers: the verified helper uses root plus 21 body rotations
and selects [0..21,39,54] from its 55-joint FK.

The public SOMA record is weaker than genuine MB output: it contains
`soma_joints[1202,26,3]`, `soma_root_quat[1202,4]`, `soma_transl[1202,3]`, fps and
joint names, with no full native local/global bone rotations. Together with the
SMPL pair it supports positional, root and some anatomical-frame calibration,
but cannot uniquely identify every axial twist or the native neck split.
Those quantities require declared bind-frame conventions or authorized full
SOMA30/77 rotation pairs. The mirrored walk can test a frozen calibration;
it does not supply the missing observations. No SOMA-to-SMPL calibration or
paired bridge error has been measured in this review.

## Public Data And Available Static Rig

The actual pinned HF tree, not a guessed folder, contains:

| Path Under `sample_data/` | Bytes | Declared SHA256 |
|---|---:|---|
| `smpl_filtered/walk_forward_amateur_001__A001.pkl` | 1,332,088 | `f31a00cd23cedb9b6cc50805d912276234a35a40678529d726df3b1dec3682d8` |
| `smpl_filtered/walk_forward_amateur_001__A001_M.pkl` | 1,334,472 | `49cbf3c604f78952474d3bcecb6bbc0b4a136eab78dc3ab8580869594383bb4f` |
| `soma_filtered/210531/walk_forward_amateur_001__A001.pkl` | 409,470 | `c51764c1c0d1b7acfd6768d53f848f892d1340fdcab4b53cbaf281bdc104ff62` |
| `soma_filtered/210531/walk_forward_amateur_001__A001_M.pkl` | 409,470 | `7c9f05ce2ead2f1fe6d7a2ecdb69081279a794ded659613104390c039721495d` |

[Pinned public sample tree](https://huggingface.co/nvidia/GEAR-SONIC/tree/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sample_data).
The non-mirrored SMPL motion record was read in memory and its actual SHA256
verified; it was not saved to the workspace. No parametric model was downloaded.
The mirror's payload was not read; its table entry is HF metadata.
Anonymous tree/payload reads succeeded without a login or gated-model request.
The sample tree was checked again on 2026-10-19; no per-sample license/README
entry was present. These public motion PKLs are not full SMPL model packages.

Actual SMPL record schema is a **flat dictionary**, unlike the clip-keyed SOMA
record: `pose_aa[2002,72]`, `transl[2002,3]`, `smpl_joints[2002,24,3]`, fps 50,
`original_pose_aa[1202,72]`, original_fps 30; arrays are float32. The existing
SOMA record has 1202 frames at 30 Hz. Pair by physical timestamps, not equal
array indices; the SMPL sampled interval ends at 40.02 s. No joint names or
shape betas are stored in this motion record.

Already present `gear_sonic/data/human/human_joints_info.pkl` is 4,302 bytes,
SHA256 `4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975`.
It contains J[55,3], 55 parents and rot_mats[55,3,3]. This is sufficient static
kinematic geometry for the verified producer, without a mesh regressor or shape
blend weights. Its provenance/use permission still needs an asset-matrix entry.
The producer check used public motion frames [0,5,10,15] at 0/0.1/0.2/0.3 s,
the first 22 axis-angle rotations, identity rotations for the target's remaining
rest fingers/face as specified by the official helper, and output indices
[0..21,39,54]. It reproduced the existing helper's FK equations in NumPy/SciPy
on CPU; no neural model or physical state was evaluated.

## Calibration, Rights And Remaining Gates

1. Freeze the identified 22+thumb3 marker schema, target J0, parent chain and root
   base correction. Calibrate SOMA30 rest frames, segment proportions, neck
   collapse and hand anchors against authorized paired data. No zero-padded
   missing landmarks. Retain measured versus inferred quantities and residuals.
2. Independently calibrate six G1 wrist outputs, including elbow-twist handling,
   side signs and actual joint limits. Removing full-body future IK does not
   certify wrist feasibility or the eventual physical tracking behavior.
3. Check the actual target rig/SMPL-derived motion permissions separately from
   SONIC weights. Full SMPL/SMPL-X model downloads require owner-controlled
   registration/license decisions; none were requested or accepted here.
   The pinned SONIC license identifies source as Apache-2.0 and weights as
   NVIDIA Open Model License. The official SMPL-X model/software terms instead
   grant non-commercial research/education/artistic use, restrict third-party
   redistribution and direct commercial users to separate licensing.
   Official SMPL-Body terms license that subset under CC-BY-4.0, excluding shape
   blendshapes and their body-creation tools. Which terms cover the bundled
   target rig requires its actual provenance entry. The pinned card/license and
   inspected local third-party asset notice do not identify a separate license
   for these SMPL/SOMA sample PKLs or `human_joints_info.pkl`; motion/rig use and
   redistribution remain explicit unresolved asset-matrix items.
   [Pinned model card](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/README.md),
   [Pinned SONIC license](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/LICENSE),
   [SMPL download requirements](https://smpl.is.tue.mpg.de/),
   [SMPL-Body terms](https://smpl.is.tue.mpg.de/bodylicense.html),
   [SMPL-X model/software terms](https://smpl-x.is.tue.mpg.de/modellicense.html).
4. Use a separate declared mode-2 encoder/packet path, preserving the original
   weights and exact 1762 packing. Golden observations must match the official
   producer, including nonzero template origin and per-frame local coordinates.
   Then run separate SMPL-mode B0/E1 and one-call B1 after parent allocation.
5. Keep original G1-mode evidence and the current full-graph smoke unchanged.
   Later tests need held-out motions and timing of human transfer, wrist mapping,
   encoder, decoder, physics and scheduling. The mirrored walk is a held-out
   calibration check, not independent motion-family generalization.

No full SMPL parametric model is intrinsically required at runtime if the owner
has authorized target kinematic geometry. The bridge is not yet calibrated or
physically validated, so the 31 ms baseline cannot be replaced with a projected
single-call timing claim.

## Source Anchors

- C++ gatherers：`${UPSTREAM_ROOT}/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:589`:
  lines 589-601 heading alignment, 612-687 relative orientation, 906-962 marker
  copying, 1687-1691 mode scalar, 1739-1758 wrist/SMPL registry dimensions.
- Training observations：`${UPSTREAM_ROOT}/gear_sonic/envs/manager_env/mdp/observations.py:1567`:
  lines 1567-1588 wrist selection, 1716-1745 per-frame local marker features.
- Training reference root：`${UPSTREAM_ROOT}/gear_sonic/envs/manager_env/mdp/commands.py:1309`:
  lines 1309-1369 corrected SMPL root and 6D input.
- Static FK helper：`${UPSTREAM_ROOT}/gear_sonic/trl/utils/torch_transform.py:595`:
  lines 595-654 static FK and [0..21,39,54].
- Live producer：`${UPSTREAM_ROOT}/gear_sonic/scripts/pico_manager_thread_server.py:449`:
  lines 449-487 coordinate preprocessing, 1304-1312 reference root,
  1350-1403 wrist transfer.
- Native SOMA30 hierarchy：`${PROJECT_ROOT}/out/vendor/kimodo/kimodo/skeleton/definitions.py:182`:
  explicit spine/neck/hand chains and endpoint definitions.
- [Genuine native reference schema](../../motionbricks/mb_sonic/reference.py#L73):
  full 30-joint local/global rotations, native positions and actual timestamps.
- Local third-party asset notice：`${UPSTREAM_ROOT}/legal/THIRD-PARTY SOFTWARE NOTICES AND ASSET LICENSES - GEAR-SONIC.txt:1`:
  inspected asset attribution list; no SMPL/SOMA sample or static-rig entry.

Only this document was updated for the final review. Frozen retarget source
SHA256 remains `9cdc2f17c9b015ddd7588b109d9fc620a9e3c951d839f1b023ed0e207d76d112`.
No runtime/retarget edits, GPU, neural policy inference, server, physical rollout,
training or agreement acceptance occurred in this review.
