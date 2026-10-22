# Reactive Gait Repair

2026-10-21. The user rejected v3 as unstable and unable to walk. The active
user page is restored to the original kinematic controller on28944. The
cadence-only v4 experiment uses28950, sharing one frozen model with28944.
Integration-origin and longer8/24 execution candidates failed the continuous
input comparison and are not in the runtime.

## Confirmed Defect

In v3, any changed input sequence bypassed the selected execution interval.
The user's session0004 received43.34 inputs/sec while4frames was selected.
661/819 dispatch pairs occurred early;517/763 adopted plans played only one
frame. Median velocity change was0.0117m/s with unchanged heading. The previous
keyboard-only test did not represent continuous gamepad input.

v4 keeps the newest input without using its arrival to shorten the selected
interval. Successive accepted forecast origins are also separated by at least
that interval, even if computation lead decreases. Genuine model samples and
left-move/right-face controls are retained. No training, pose filtering, IK,
Root replacement or frame duplication was added.

## Model Mechanism

The trained goal specifies terminal velocity at frame31, about0.933s after
the four-frame history. Every fresh call restarts token inference. Taking only
the earliest1/2frames repeatedly restarts the acceleration prefix and changes
the four-frame latent grouping. The four-pose codec mask matches training.
See [source audit](reactive_pose_context_audit_20261021.md).

Matched flat20s trials, requested forward speed1.2m/s, achieved settled forward
speed0.674/0.888/1.039m/s for1/4/8-frame execution. Each used the same phased
forward/stop/strafe/reverse commands and initial history. Sideways motion remained
weak. Urban trials also retained terrain and direction failures.

## Evidence

Directories under `out/pfnn_mixed_20261015/`:

- `reactive_gait_diagnosis_20261021_v1`: CPU reconstruction of6545 displayed
  frames and4592 forecasts.4589 conditioning histories match eventual playback
  exactly. Real session0004 forward intent1.6m/s achieved0.308m/s over13.5s.
  Body-relative joint stepP95 was11.60cm in its4frame mode.
- `reactive_integration_probe_20261021_v1`: eight600frame Urban trials,
  original/anchored integration crossed with1/2/4/8execution.
- `reactive_integration_flat_20261021_v2`: six600frame flat trials,
  original/anchored crossed with1/4/8. Failedv1 setup is retained.
- `reactive_analog_replay_20261021_v2`: same recorded2144frame input sequence,
  all modes fixed to4, synchronous calls and fixed3frame lead. Cadence-only
  used536plans and had RootY0.949..1.211m, but commanded-direction progress
  averaged only0.332m/s against requested1.559m/s. The one-frame final plan is
  the truncated end of the trace. Original scheduling and cadence-plus-origin
  both diverged in height. Original here reconstructs raw output by undoing
  the experimental graph's planar translation, with float32 rounding; it is
  not a bitwise reproduction of the original session or browser deadlines.
- `reactive_analog_longer_20261021_v1`: raw native output with fixed8/24
  execution. Progress improved to0.942/1.131m/s, but RootY reached37.39/24.22m;
  both candidates fail. Longer execution also changes control latency.

Planar integration correction reduced short flat jump metrics, yet failed the
long analog trajectory. It was removed from production. All failed trials remain.

## Deployment

Oldv3 PID25772 stopped2026-10-21T11:54:28+08:00. v4 PID11492 starts from
`reactive_control_20261021_v4/run`; source snapshot and logs are adjacent.
Original28944 and all15maps remain the user-facing entry. Stable all-terrain
gait is not yet achieved.46Python and32JavaScript checks pass. GPU admission
passes7cases with unchanged tokens/checkpoint and exact skin transport.
Formal training remains38.4M exposures,0new updates.

Browser v4 ran2310actual frames at30.0106fps,4frame selected,7.5updates/sec,
computeP50/P9534.11/44.85ms,0server failures. W6s,S2s,D2s,A2s then idle still
drifted from approximately[1.42,1.21,2.72] to[12.33,0.86,2.81] over the subsequent
idle period. This is a quality failure. The page was returned to28944Urban/speed3,
paused at its original seed. Evidence: v4`review.json`, browser telemetry and
screenshots. Browser script version was advanced tov4 to avoid cachedv3 config
rejection. All inference probe processes have exited; only11492 serves the viewer.

The next model change needs short-prefix continuation training with generated
histories and immediate-intent semantics. Full3D observation is also unfinished.
No additional training has started.
