# G1 Retarget Repair: Fixed-Reference Comparison

## Current Result

Reference geometry is repaired on the fixed flat case. Physical gait improves
modestly and remains unsuitable for release. No replacement Xbox service was
started. Existing 28948 v32 and SONIC 28947 remain unchanged.

All three candidates use exactly the same saved 16.8s MB proposal, initial G1
qpos/qvel, 100 warmup steps, frozen ProtoMotions ONNX, original PD gains, joint
and motor limits, 50Hz controller and 1ms MuJoCo integration. Each completes
16s / 795 policy ticks / 15900 controlled physical substeps without a fall or
actual joint/force violation. No training, filtering, actuator residual,
physical Root overwrite, auxiliary force or automatic reset was added.

## Changes

1. Native material-axis limb frames replace bending-plane twist inference.
   Native +X knee and -Y/+Y elbow hinge conventions define the static material
   frame; the actual MB global rotation transports it. This preserves the
   calibrated anatomical rest-frame gauge and remains defined at extension.
2. The same whole-reference Root/joint solve receives foot-to-scene signed
   distance inequalities from the original G1 capsule geometry and canonical
   MuJoCo collision scene. Minimum reference clearance is 1mm. Collision search
   margins change only a private reference model, not the physical model.
3. The support candidate also uses MB's four heel/toe prediction channels for
   soft surface-height and zero-relative-velocity objectives. The coefficients
   are 200/m and 3/(m/s), respectively, inside the reference optimization.
   Actual contact and motion are still produced by the frozen controller and
   physics. Existing committed-prefix, joint and velocity constraints remain.

At 3.3333 -> 3.3667s, the old inferred thigh frame turns 34.558deg while the new
material frame turns 1.280deg, matching the native bone rotation. No temporal
filter is used to remove this amplification.

## Measured Comparison

| Metric | Original | Material Axes | + Ground | + Support |
| --- | ---: | ---: | ---: | ---: |
| Reference peak penetration, mm | 23.280 | 22.915 | 0 | 0 |
| Actual peak penetration, mm | 4.672 | 4.460 | 4.806 | 4.065 |
| Actual relative-body 8-14Hz energy, m2 | 9.051e-6 | 8.493e-6 | 1.030e-5 | 8.272e-6 |
| Actual joint acceleration RMS, rad/s2 | 25.699 | 24.537 | 26.300 | 25.027 |
| Final planar Root error, m | 3.840 | 3.751 | 3.692 | 3.625 |
| Actual lateral range, m | 2.734 | 2.777 | 2.772 | 2.983 |
| Actual forward velocity, m/s | .956 | .966 | .969 | .990 |
| Reference-solver P99, ms | 20.596 | 18.677 | 233.846 | 289.979 |

The support candidate reduces measured high-frequency body energy by 8.60%,
actual peak penetration by 12.99%, and final planar error by 5.60%. Its lateral
excursion increases by 9.11%. The source advances at 1.150m/s; actual progress
still falls behind. Ground-only constraints increase physical jitter, despite
eliminating reference penetration, so that variant is rejected.

## Release Blockers

- Remaining physical oscillation and metre-scale drift are too large.
- The contact solve takes too long for rapid interactive replanning.
- A new 6cm abrupt-reference-dip stress case exhausts the fixed contact QP;
  it stays as a strict expected-failure test, limited to the explicit constraint
  failure type. The failed solve is never actuated.
- Constraint row scaling, redundancy reduction and alternate solver attempts
  did not resolve the stress case. ProxQP's installed NumPy binding also rejects
  its minimal initialization; OSQP reaches its iteration limit. These trial
  implementations were removed. There is one explicit DAQP solver, no fallback.

The frozen actor observes current proprioception and four future joint/torso
targets, but no Root XY/linear-velocity target. Geometry repair can improve the
reference it tracks; it cannot directly close that missing world-trajectory
feedback channel. Further zero-training work must address contact-compatible
trajectory construction and that execution interface, with separate measurements
for body oscillation, lateral error, forward progress and response time.

No new 30fps browser or Xbox test was claimed. The replacement stays off 28948.

## Evidence

- `out/mb_sonic_e0_e3_20261018/retarget_repair_plan_20261020.json`
- `out/mb_sonic_e0_e3_20261018/retarget_repair_comparison_20261020.json`
- Cases: `retarget_native_axes_v1`, `retarget_native_ground_v1`,
  `retarget_native_support_v1`; all raw references, complete physical NPZ and
  audit reports are retained. No failed or unfavorable variant was removed.
- Source proposal SHA256:
  `55e0b71c2bbbde3e0e1c1e102c2398e887193d443d249b722e50619b97f482fb`.
- Frozen actor SHA256:
  `a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c`.
- Test suite: 70 passed, 1 strict expected failure; Torch deprecation warnings.
- CPU-only comparison; no new GPU allocation or formal training exposure.

Method reference: [GMR](https://github.com/YanjieZe/GMR), whose retargeting work
motivates assessing mapping quality separately from physical tracking.
