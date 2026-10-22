# Rejected Metric Source-Foot Candidate

## Status: Rejected And Removed

2026-10-20 23:30: The full XYZ anchoring variant described below was evaluated
in v50 and rejected. It completed136.53s without the earlier coverage exhaustion,
but actual Root height fell to0.15464m and several commanded directions worsened.
The source-foot vertical differences accumulated through successive committed
anchors. The saved physical audit passes its mechanical/fall-contact checks;
the walking posture and control outcome are unacceptable.

The parent then tested v51, which anchored only world XY and retained native
embodied height. It failed on actual torso contact at 9.346s. Both variants are
rejected. Their physical output and source snapshots remain preserved.

The optional `source_foot_metric` implementation and its dedicated tests have
been removed from production retargeting and trajectory solving. The scratch
`scripts/review_mb_metric_feet.py` was also removed; all saved CPU reports remain.
Native Foot banking, vectorized target quaternion conversion and the v49
equivalence regression are retained. Runtime/package flag cleanup belongs to
the parent. This cleanup did not access services, GPU or browser state.

Post-cleanup scoped CPU suite: **64 passed**, eight existing deprecation
warnings, 15.51s. `retarget.py` matches the v49 snapshot; the sole production
change in `trajectory_reference.py` versus v49 is batched target quaternion
conversion. The v49 qpos equivalence regression passes at 1e-12 tolerance.

Preserved physical evidence: `out/mb_sonic_e0_e3_20261018/metric_scene_foot_v50`
and `out/mb_sonic_e0_e3_20261018/metric_planar_foot_v51`, including their source
snapshots. CPU reports `metric_source_feet_cpu_v1_failure.json` and v2-v4 remain.

The sections below record the historical full-XYZ experiment. Contact-aware
retargeting redesign remains a separate task.

## Historical Contract (Removed)

```python
solver = TrajectoryReferenceSolver(
    adapter.kinematics, iterations=1, pose_task_frame='root',
    source_foot_metric=True,
)
reference = adapter.retarget(source, constraints=solver, boundary=boundary)
```

The adapter resamples the source LeftFoot/RightFoot world markers onto the
existing 50Hz grid and converts Y-up to Z-up. Direct solver callers pass the
same array as `source_foot_positions_m`, shape `[N,2,3]`. The option requires
Root-relative tasks and cannot be combined with `foot_ground`.

For each foot, at the last committed sample `a`, the world target is:

`target(t) = FK(committed_G1(a)).ankle + source_ankle(t) - source_ankle(a)`.

The full three-dimensional displacement remains metric. No limb-length scale
is applied. A constant per-packet source registration translation cancels in
the difference; it must not be added a second time. The accepted G1 pose
carries the anchor across packets. Axes remain the shared world axes; no
heading-relative rotation is applied to that displacement.

The existing independent continuous Root objective is solved first and held
fixed during foot fitting. It retains the original committed prefix and spline
continuation. Foot targets are expressed relative to that planned Root, so the
fit changes observed joints instead of moving unobserved Root XYZ. Across the
four saved packets, candidate/reference Root difference is at most 2.22e-16.

Native/material-axis orientation objectives, foot position/orientation costs
(50/m and 5/rad), spline, iteration budget, original joint/velocity limits,
posture and temporal costs are retained. No support locking or zero-velocity
target is imposed. Source predicted contact channels are used only for the
offline diagnostic, not as measured forces.

## CPU Same-Packet Evidence

`out/mb_sonic_e0_e3_20261018/metric_source_feet_cpu_v4.json` binds all four saved
packet hashes and their accepted-parent hashes. Each comparison uses identical
native mapped qpos and the identical committed boundary. Metrics cover editable
future frames; support intervals include the committed-to-editable boundary.
Support means both endpoint native Foot confidences exceed .8.

| Saved case | Position RMS, baseline -> candidate | Support ankle speed median, baseline -> candidate | Candidate P50 / P99 solve |
| --- | --- | --- | --- |
| v42 early/slow | 6.428 -> 1.047 mm | .0694 -> .0433 m/s | 11.745 / 15.082 ms |
| v42 around 6s | 40.747 -> 12.785 mm | .2386 -> .0767 m/s | 11.584 / 12.484 ms |
| v42 around 16s | 49.920 -> 11.775 mm | .2161 -> .0960 m/s | 11.495 / 14.885 ms |
| v48 around 44s | 15.537 -> 11.333 mm | no qualifying intervals | 11.980 / 12.976 ms |

The middle two source support-speed medians are .0630/.0831m/s. Corresponding
candidate velocity-error medians against the source are .0691/.0501m/s, versus
.1981/.1442m/s in the baseline. All comparisons preserve the committed prefix
exactly, retain positive original joint margins and stay below original joint
velocity bounds (maximum ratio .20025).

Each solver has three warmup calls and 30 measured calls per packet, one BLAS
thread, one Torch thread. These are solver-only timings on the parent's active
machine. They exclude generation, registration, IPC and the physics actor.

## Small Default-Path Performance Repair

Target rotation matrices now convert to quaternions in one batch before the
Gauss-Newton loop. This replaces 31*5 small conversions per iteration with five
batched conversions. No targets or optimization equations change.

The v49 frozen implementation parity test matches qpos within 1e-12. The v3
(scalar conversion) and v4 (batched conversion) reports have identical quality
metrics. Default solver P50 falls from 14.75-14.96ms to 10.02-10.53ms. Candidate
P50 falls from 16.26-16.40ms to 11.50-11.98ms. Running v49 processes and their
immutable package snapshots were left untouched.

## Remaining Clearance Obstacle

This contract preserves the source ankle's vertical displacement from the
committed anchor. It does not supply a source sole landmark, surface ID,
surface height/normal or a robot collision query. Absolute tilted-sole clearance
therefore cannot be imposed by this candidate.

Vertical target RMS improves from 13.851 to 11.813mm in the 6s packet, but rises
from 8.564 to 10.957mm in the 16s packet and from 10.552 to 10.753mm in v48.
The 16s packet's maximum 3D target error was 30.620mm. These short-window gains
did not survive the subsequent v50/v51 physical comparisons described above.

## Historical CPU Checks

Pre-cleanup focused suite: **71 passed**, eight existing Torch deprecation warnings,
13.48s. Command: `python -m pytest tests/test_mb_trajectory_reference.py
tests/test_mb_sonic_retarget.py -q --disable-warnings`.

The initial analysis attempt failed on an empty support interval percentile;
`metric_source_feet_cpu_v1_failure.json` retains that failure. v2 retains the
rejected exact-per-packet Root variant. v3 restores the original continuous Root;
v4 adds only batched quaternion conversion. Existing failures were not removed.

Historical full-XYZ source SHA256 (not current production):

- retarget.py: `e81530bc817e57256bd7964cb3d7d4038457ab48443edd76995fb1a5f927b044`
- trajectory_reference.py: `6eb474855bb9db76dec90d5bc5f725bc27f294ff1cf98506a87a173cff065908`
