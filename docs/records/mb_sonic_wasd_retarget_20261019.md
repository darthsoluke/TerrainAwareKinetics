# WASD Reference Retarget: Calibration Gauge Fix

Date: 2026-10-19. CPU embodiment conversion only; no model, GPU, browser,
server or physical rollout was launched by this owner.

## Root Cause And Change

The saved calibration estimated segment offsets from anatomical frames of the
official paired marker positions. The runtime instead supplied native joint
global rotations. Their unobserved axial-twist gauges differ. Thus a legal
joint vector could still represent an incorrectly twisted embodiment.

The actual native idle seed exposed this mismatch:

| Joint | Legacy Raw-Global Gauge | Matched Marker Gauge |
|---|---:|---:|
| left hip yaw | -1.43893567 rad | 0.13119602 rad |
| right hip yaw | -1.79181681 rad | -0.18034691 rad |

Runtime `retarget` now constructs segment frames from the source's actual
`world_positions_m` and named SOMA30 FK markers, using the same
`anatomical_frames` and native rest-frame basis as calibration. With C the
declared Y-up-to-Z-up rotation and F an anatomical marker frame:

```text
paired gauge:   H_pair = F_Zup * rest_frame_Yup.T
runtime gauge: H_run  = C * F_Yup * rest_frame_Yup.T
robot target:  H_run * original_calibration_offset
```

This changes the input gauge, not the XML hinge axes, limits, calibration
offsets, bone lengths or reference clock. Joint conversion still uses the
original XML rest transforms and analytic hinge/Euler projection. No generic
bounded optimizer was added: after the gauge correction, every available real
source tested below is legal. A later genuinely infeasible pose still fails.

The root is observed, so its original native Hips rotation is retained rather
than reconstructed from float32 marker roundoff. Root XYZ, the pre-existing
calibrated morphology Z offset, and the original SO(3)/position resampling are
unchanged. Tests compare every returned root pose against the legacy root
resampled at identical timestamps, with exact array equality. No actual G1
state is readjusted, no foot is locked, and no reference root is reanchored.

Unobservable/collinear marker frames still raise an error. There is no alternate
raw-rotation fallback or arbitrary replacement twist axis.

## Interface And Receipts

Existing calls remain:

```python
adapter = CalibratedSomaG1Adapter.load(calibration_path,
    robot_model_path=robot_xml, representation=native_representation)
initial = adapter.retarget(seed_source, start_time_s=-0.1)
packet = adapter.retarget_packet(source, start_time_s=source_time,
    consumer_horizon_s=1.12)
```

The same gauge fix applies to initialization and future references. The default
consumer duration remains 1.0 s; explicit 1.12 s still means 57 dense 50 Hz
samples. No automatic shortening, retry, clock stretch or angle clipping.

New adaptation receipts:

- `source_gauge`: `motionbricks.mb_sonic.anatomical_fk_markers.v2`.
- `source_constraint_audit`: every raw native sample, including unused tails.
- `dense_constraint_audit`: every returned 50 Hz sample and finite difference.
- `model_mapping_residual`: SO(3) error between the calibrated anatomical target
  and actual XML-chain rotation, over all raw frames and all 14 segments.
- `joint_velocity_limits_rad_s` and `velocity_limit_source`: named limits from
  the original same-model URDF, with its path/hash and declared-not-measured tag.

MuJoCo XML supplies position limits but not velocity limits. The accompanying
`g1_29dof_with_hand.urdf` explicitly supplies velocity values: hip/knee/waist
20 or 32 rad/s, ankle/waist roll-pitch/arms 37 rad/s, wrist pitch-yaw 22 rad/s.
Its SHA256 is
`8aab2ef245d9d56a690048a98f58be03858a5e7e8a97533e03327bd64562496a`.
These are URDF reference constraints, not measurements of achievable loaded
speed or force. Torque ctrlrange is not used as a velocity bound.

Both full-source and dense checks reject position or velocity violations using
the unchanged 1e-6 position tolerance and 1e-6 rad/s numerical velocity tolerance.
Linear interpolation between checked source joint angles preserves their box
bounds and cannot exceed the source segment's speed. Periodic angle unwrapping
is not used for these bounded hinge coordinates.

`audit_joint_limits` and `_map_rotations` retain the old raw-global conversion
as an explicitly labeled **legacy diagnostic**, not the runtime reference.
Its previous 11 violations on the original proposal remain visible in packets.
Existing saved failure reports, source NPZs and calibration bytes are untouched.

## Actual CPU Evidence

The actual failed v3 archive is
`out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v3/episode_0001_numeric/`
`f07d781c2545f8cfb0685d64067d941f175ff62235b9e55680cfa8289bb71a3d.npz`.
The test decodes its archived `metadata_json`/array references without generating
or substituting a frame.

At reference time 1.96 s, its right ankle roll changes from the incorrect
gauge's -0.261977605 rad to 0.035303034 rad. The corrected full 52-frame source
has minimum position margin 0.178642759 rad, peak speed 4.25618138 rad/s,
maximum named-velocity ratio 0.21280907, orientation RMS 0.02029753 rad and
maximum orientation residual 0.13926114 rad. Full-source checks and the actual
57-sample [1.16,2.28] packet both pass. Foot SO(3) RMS residuals are
0.0498261/0.0514594 rad; the unavailable ankle yaw is not claimed to be tracked.

All 12 saved v3 sources were read, plus the original, compiled-v2 and full-plan
MB proposals: 15 actual sources, 52 frames each, and the actual 4-frame seed.
Every raw frame and returned dense interval was checked. Across the 15 sources:

| Quantity | Actual Range |
|---|---:|
| position/velocity violations | 0 / 0 |
| minimum position margin | 0.16729435..0.18125309 rad |
| peak joint speed | 2.37637075..4.73289633 rad/s |
| maximum named-velocity ratio | 0.07442595..0.23664482 |
| per-source model orientation RMS | 0.01458975..0.02164360 rad |
| largest orientation residual | 0.14645456 rad |

The seed has minimum position margin 0.20125468 rad and orientation RMS
0.03959496 rad. Its root remains
(-0.00031480059,0.00007742255,0.79073706269) m. Exact XML FK places its eight
original foot collision spheres 0.00422417..0.01761373 m above flat Z=0.
No initial height adjustment was made; these are geometry gaps, not contacts.

The paired diagnostics now consume the original paired marker positions
directly, instead of manufacturing a native pose in the same rotation gauge
and using that as a forward oracle:

| Paired Walk | Joint RMSE | Root-Relative Body RMS | Body Rotation RMS |
|---|---:|---:|---:|
| A001, calibration | 0.05251242 rad | 0.00899822 m | 0.07505459 rad |
| A001_M, held out | 0.05190319 rad | 0.00963766 m | 0.07572023 rad |

Both 1202-frame pairs have zero position violations. The earlier tiny
G1/inverse/forward error is now labeled legacy rotation-gauge algebra; it does
not validate native FK or the physical-feedback inverse. No inverse
certification or physical tracking result is claimed from that identity.

Thirty warmed CPU repetitions of the failed source's complete 1.12 s packet,
including preserved legacy audit, full-source gauge conversion and residuals,
50 Hz FK/velocities, both constraint checks and returned snapshots:

| Threads | P50 | P95 | P99 | Max |
|---:|---:|---:|---:|---:|
| 2 | 12.09375 ms | 13.28634 ms | 13.80448 ms | 13.92260 ms |

These measure this conversion component. Parent owns GPU generation, policy,
physics, WASD UI, all-map initialization and final timing/quality measurements.
The available saved MB sources are scene 000 walks; all-map physical coverage
must come from that separate run, not these repeated CPU conversions.

## Preservation And Checks

- Calibration SHA256 remains
  `6be2cc02ae225c81a1ec28b6af59239c1656660d0f88b592ee4d725235d0bc6c`.
- Original 1.0 s packet SHA256 remains
  `450e4a0c8579017528882c00f96e5fbecf7822ab708208d1effd4ab3eb0f6041`.
- Final retarget source SHA256:
  `8fbcd9606eba1034e680dffbfee303c05ae5c4f8154c1b1085b3e2f6ae100664`.
- No trained weights, scenes, old reports, package initialization, runtime,
  configuration or UI files were edited by this owner.

The CPU tests cover the actual seed, all saved v3 raw sources, repeated genuine
MB proposals, paired and held-out reconstruction, root identity, resistance to
non-root raw-global gauge changes, source immutability, retained legacy failure,
and deliberate invalid-position/too-fast unit inputs that must still fail.

```powershell
$env:PYTHONPATH = '.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_mb_sonic_retarget.py -q
```

Final result: **47 passed in 9.65 s**, with eight existing TorchScript
deprecation warnings. The paired-frame residual checks and genuine native-FK
gauge tests are distinct; the legacy algebra test is labeled accordingly.

## Authorized Independent-Hinge Batching

The subsequent scoped candidate only batches existing SciPy
`Rotation.from_rotvec(...).as_matrix()` calls: full FK 43 calls become one;
each segment-map pass uses one call per group, 29 calls becoming 13. Every
individual float64 rotvec expression and parent-to-child matrix multiplication
order is unchanged. Gauge, root, offsets, Euler extraction, limits, checks,
full-source legacy records and consumer horizon are unchanged.

Baseline: the actual frozen source at
`out/mb_sonic_e0_e3_20261018/wasd_live_launch_v1/sources/retarget.py`, SHA256
`8fbcd9606eba1034e680dffbfee303c05ae5c4f8154c1b1085b3e2f6ae100664`.
Candidate SHA256:
`dfc6e4290335fc1dea8596864115195b27048348ca073d4a2562fea4aa0803a4`.

CPU proof covered all 16 actual saved raw sources in
`wasd_v3/episode_0001_numeric` and `e3_b1_fullplan_v3/episode_0001_numeric`.
All seven reference array fields (timestamps, qpos, qvel, body position,
quaternion, linear velocity and angular velocity) are **BITWISE IDENTICAL**.
Full 45-body FK positions/rotations are also bitwise identical. Source/dense
constraints, legacy audits and mapping residuals are identical. No model was
trained or loaded for inference, no GPU or physical rollout was run.

Twelve interleaved uncontented CPU calls per implementation, four Torch CPU
threads, on the same saved real packet:

| Implementation | P50 | P95 | Max |
|---|---:|---:|---:|
| frozen baseline | 12.10600 ms | 12.652325 ms | 12.6746 ms |
| batched hinges | 12.02140 ms | 12.537595 ms | 12.6350 ms |

This small uncontended difference does not establish a live GIL-contention
improvement. The original suite passes unchanged: **47 passed in 9.40 s**,
eight existing TorchScript deprecation warnings. No switch-interval or other
production setting was changed. The separately requested previous GIL result
is preserved, without rerunning, in
`out/mb_sonic_e0_e3_20261018/retarget_gil_cpu_20261019_v1.json`, SHA256
`3884b4e6a40d8ecd494f7873e2d8654bef7329dba2b742634669526f82d17d79`.
