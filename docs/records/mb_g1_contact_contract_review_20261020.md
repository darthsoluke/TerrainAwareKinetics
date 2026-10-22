# SOMA -> G1 Contact And Tracker Contract Review

Scope: retarget/tracker sidecar; scene-query coordinate, height and path
consistency belongs to the main review. Runtime files were left untouched.
All new measurements used saved arrays and CPU FK/rotation arithmetic. No
training, services, SSH, GPU jobs or physical rollouts were run.

## Findings

1. **[P1] Metric Root travel survives, but contact-compatible stride does not.**
   `motionbricks/mb_sonic/retarget.py:306-355` maps segment orientations onto
   G1 link lengths while preserving metric Root translation plus one constant
   height offset. It never fits source foot positions or contact channels.
   `motionbricks/mb_sonic/trajectory_reference.py:187-196` then takes its foot
   targets from that already-mapped G1 FK, so its Cartesian objectives cannot
   recover the lost source excursion. With `foot_ground=None`, stance and
   clearance residuals at lines 205-222 and 243-256 are absent. The runtime
   construction at `motionbricks/mb_sonic/runtime.py:763-766` does not supply
   this optional helper.

   **Measured:** saved 0.6s packets give the following horizontal ankle
   excursion, in metres, left/right. This is the norm of the two coordinate
   ranges after subtracting each sample's Root, not a complete-cycle stride.
   Source positions were interpolated onto the prior's existing 50Hz grid.

   | Packet adoption | Native SOMA | Raw mapped G1 | Constrained G1 |
   | --- | --- | --- | --- |
   | v42, 2.00s, requested .1524m/s | .02090/.03588 | .01669/.02643 | .01180/.02784 |
   | v42, 6.08s, requested .7826m/s | .37809/.38200 | .28592/.28896 | .29444/.28714 |
   | v42, 16.12s, requested .7227m/s | .51176/.31490 | .39107/.24556 | .39829/.24954 |
   | v48, 44.02s, requested .2896m/s | .11365/.03727 | .09054/.03212 | .09185/.02322 |

   In the v42 6.08s packet, native Foot confidence exceeded .8 at both ends
   of 22 ankle/time intervals. Median horizontal world ankle speed was
   .06001m/s in SOMA, .23222m/s after mapping, and .23713m/s after constraints.
   At 16.12s, the corresponding medians were .13605/.23125/.23079m/s over
   30 intervals. These are predicted-contact marker measurements, not measured
   sole forces. They locate an added stationarity error before the actor.

   **Inference:** shorter target links with unchanged Root transport create
   stance slip unless foot placement and Root travel are fitted together.
   In the translating, fixed-heading case, a planted foot needs
   `d(relative_foot)/dt = -Root_velocity`; scaling only its excursion breaks
   this equality. Different limb lengths are inherent; losing that equality
   without a contact/stride objective is an integration defect.

2. **[P1] Native foot banking is discarded even in material-axis mode.**
   `motionbricks/mb_sonic/retarget.py:73-78` constructs each foot frame from
   ankle-to-toe and ankle-to-knee vectors. Material frames at lines 95-105
   cover Leg/Shin/Arm/ForeArm only. A rotation around the ankle-to-toe axis
   leaves the two foot markers fixed, so the actual foot rotation supplied
   to `_map_anatomical` cannot affect its output through this axis.

   **Measured controlled probe:** on frame 100 of the saved fixed proposal,
   right-multiplying LeftFoot's native global rotation by 10 degrees about its
   neutral Foot->ToeBase axis produced exactly zero change in all 36 mapped
   qpos values. The toe direction changed by less than 5e-19. This identifies
   information loss in the adapter, with no model inference involved. G1 has
   ankle roll; this loss cannot be explained solely by its missing toe joint.
   Its contribution to the recorded fall has not been measured.

3. **[P2, inactive in v48] The support helper uses a rotated material bottom
   point instead of the capsule's support point against the surface.**
   `motionbricks/mb_sonic/reference_ground.py:39-43` subtracts local
   `[0,0,radius]` from capsule endpoints; lines 71-81 rotate that fixed point
   with the foot and penalize its surface gap. Against normal `n`, the true
   capsule endpoint support is `endpoint_world - radius*n`. At foot tilt
   theta, the chosen point's normal gap is too high by `r*(1-cos(theta))`.
   Original foot radii are .008/.010m; the .010m capsule gives .603mm at
   20 degrees and 2.929mm at 45 degrees. Thus the soft support target and the
   exact collision inequality can disagree. This is an analytic geometry
   result, relevant when enabling the helper, rather than a v48 failure cause.

## Tracker And Representation Limits

`motionbricks/mb_sonic/protomotions_policy.py:114-125` supplies four future
joint positions/velocities and torso rotations, plus current proprioception
and the previous action. Target Root XYZ, Root linear velocity, sole positions
and contact channels are absent. Consequently a correction confined to target
Root translation has no effect on these actor inputs. Adding unused packet
fields cannot change that frozen input contract.

This agrees with the intentional uncoupled solve at
`motionbricks/mb_sonic/trajectory_reference.py:24-35,267-269` and the Root-relative
Jacobian at `motionbricks/mb_sonic/constrained_reference.py:77-83`. Three existing
CPU tests passed: Root-registration invariance and world/Root task Jacobians.
Source SOMA toe articulation, three-axis foot orientation and all native body
DOFs cannot be reproduced exactly by G1. Supported ankle roll, contact phase,
reachable foot clearance and metric horizontal excursion can still be retained
as explicit reference objectives.

The six-control spline and whole-motion temporal penalties at
`motionbricks/mb_sonic/trajectory_reference.py:163-186` also have no stride or
contact-event preservation condition. The sampled packets contain 14-15 fixed
prefix frames; whole-window range changes therefore include accepted history.
Editable-future horizontal ankle differences were only 6.05mm at v42 2.00s
and 5.89mm at v48 44.02s. This review does not isolate a spline-only speed loss.

## Smallest Robust Zero-Training Repair

Repair the existing foot-target construction and reference objective together:

- Transport the native Foot material orientation, using a static gauge
  consistent with the paired calibration. Preserve G1's supported pitch/roll;
  project the unsupported component within its joint constraints.
- Fit G1 sole sites to source-derived foot placement, swing clearance and
  support timing, with explicit embodiment offsets. During support, constrain
  site displacement/velocity against the planned reference Root transport.
  Use source contacts as bounded prediction confidence; Foot/ToeBase channels
  are not measured robot heel/toe forces.
- Ensure the fit changes the joints/torso the actor actually observes. For
  this foot fit, hold the planned reference Root transport fixed or derive
  Root-relative sole targets from it; a Root-only clearance correction is
  invisible to the actor. Preserve the committed prefix and original bounds.
  Regularize the correction while retaining the native gait as the target.
- If using `ReferenceFootGround`, use surface-normal capsule support geometry
  and wire source confidence into the solve. The saved support candidate's
  P99 is 289.979ms, so enabling it unchanged is not an interactive repair.

These are reference-layer changes. They require no model training, actual
Root/state overwrite, motor/render filtering, fallback controller or forces.
Their physical speed and stability benefit remains to be tested separately.

## Evidence And Attribution

All evidence below is under `out/mb_sonic_e0_e3_20261018/`:

- `simulated_xbox_root_plan_v42/recording/episode_0001_numeric/`: packets
  `b234d3118ab0a4a6f85b1c774f1d8a92ffb79cf2f0cdb3121bccdd97319730d5.npz`
  (2.00s), `4271efea3e2eb7a5fa20e2b1a2dca490a61fdcf6b6ae49a502590a54116fa981.npz`
  (6.08s), `91e46c5c9ea1b11fd979b41cbb84cf00520efa6fff70fc3436928cb8c8ad25f1.npz`
  (16.12s). Read `raw_source`, `adapted_reference` and its `native_pose_prior`.
- `interactive_clock_v48/recording/episode_0002_numeric/98991d281a7f563d33b6f7f906fe2fab4444876ed43a2886db0952b40b16987e.npz`
  supplies the 44.02s row. Current `retarget.py` and `trajectory_reference.py`
  hashes equal the v48 saved sources.
- `retarget_native_axes_v1/generation/proposal.npz` supplies the rotation
  probe. Source SHA256:
  `55e0b71c2bbbde3e0e1c1e102c2398e887193d443d249b722e50619b97f482fb`.
- `retarget_repair_comparison_20261020.json` records the same-source frozen
  actor comparison: material axes still give 22.915mm reference penetration;
  support reduces final Root error from 3.840m to 3.625m. This is prior physical
  evidence, not a new rollout or a controlled v48 comparison.
- `simulated_xbox_root_plan_v42/gamepad_review.json` records .67248/.66891m/s
  reference speed versus .19174/.18202m/s actual strafe/backward progress.
  The measured mapping excursion reduction is smaller than this execution
  shortfall, so mapping alone does not account for the full slowdown.
- At v42 2.00s, all 60 ankle intervals exceed .8 predicted Foot confidence,
  yet source ankle speed has median .04445m/s and P95 .20223m/s. Slow-command
  contact inconsistency can already exist upstream. These samples do not
  separate learned error from pre-Pose planning or conditioning mismatch.
- v43's full map000 pass is documented in `docs/mb_intent_errata_20261020.md:92-102`.
  The v48 episode-2 terminal record reports a pelvis/torso contact failure at
  49.983s. Different input timing prevents attributing that difference to one
  retarget defect. The adapter's exact rotation nullspace and added contact
  marker slip are direct mapping evidence; model underfit is not established.

## Bounded Foot-Banking Repair Implemented

The foot-banking finding above describes the pre-repair implementation.
`retarget.py` now transports each native Foot frame from its neutral
Foot->ToeBase and ankle->knee gauge. The gauge matches `native_rest_frames`;
its neutral frame arithmetic uses float64, including for float32 skeleton
inputs. `anatomical_frames` consumes this frame only in material-axis mode.
Marker-mode diagnostic mapping is unchanged.

Added tests in `tests/test_mb_sonic_retarget.py` check the neutral gauge under
identity and a rigid rotation/translation, plus +10-degree left and -10-degree
right foot banks on a saved native pose. Both banks reach the supported G1
ankle roll and foot FK within one degree, remain inside original joint bounds,
and leave Root and other joints unchanged. Inputs are read-only; the tests
forbid MjData allocation and physical stepping and preserve model qpos0.

Scoped CPU result: **55 passed**, with eight existing Torch deprecation
warnings. This includes all 54 retarget tests and the existing straight-limb
material-frame regression. The entire source suffix from `native_rest_frames`
through the end of `retarget.py`, including the clock/endpoint handling, is
unchanged from the pre-edit snapshot. No stance/contact objective was added.
The corrected-chart comparison with the planar governor disabled remains
main-thread integration work; this repair adds no physical validation claim.
