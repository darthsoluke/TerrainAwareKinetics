# Continuous Reference Versus Interactive Control

## Code And Literature Review: 2026-10-20

See [Online Control Review](mb_online_control_review_20261020.md). New CPU-only
analysis of47saved v36plans identifies native0.582m/s -> join10.033m/s at1.82s,
followed by history11.736m/s at2.46s. The cause is a re-registered incoming
fragment joined to the old source and then reused as conditioning history.
Generation also precedes the protected handoff state; Root bounds are absent.
Two actual frozen ONNX calls with reference XYZ/linear velocity changed produce
bitwise-identical motor targets, confirming the tracker's missing direct Root
regulation input. Eight primary papers reviewed; ordered repair and architecture
alternatives are documented.0new MB/physics/GPU/training,0runtime source changes.
Fresh scan finds no Python service or28947/28948/28949listener; prior22856 is gone.

## Simulated Xbox: 2026-10-20 16:09

User is off site; an explicit `?input=simulated-xbox` mode now supplies100Hz raw
dual-stick axes through the real mapper/deadzone/latch/WebSocket path. The83s
trace includes ramps, independent movement/yaw, analog circles, reversals and
neutral noise. It is labeled Simulated Xbox and stops on focus loss or failure.

Continuous-input validation FAILS: PFNN000 falls at35.326s; the existing64m flat
arena falls at4.885s, far from its boundary. Both retain30fps/0dropped samples.
The prior142s discrete-key/control result is therefore not stable-gamepad approval.

Same-timestamp preview review finds that new commands overwrite targets already
seen by the tracker: PFNN maximum joint change.809475rad, torso1.795073rad. The
new code preserves the original100ms age bound plus160ms tracker lookahead and
one20ms control step; quaternion hemisphere is anchored to accepted reference.
On v36,46real adoptions have exactly0Root/joint preview change, torso<=7.55e-16rad.
323Python/30JS checks pass. Startup also waits for accepted-input ACK before
starting physics; initial foreign-session rejections remain in v1 evidence.

The repaired-preview trial still falls at6.935s/29.977fps/0dropped samples.
Its recorded reference Root peaks at52.072m/s and143.928rad/s, while actual joint
tracking RMS is median.27237rad/max.81122rad. Per-native-block Root speed peaks
at16.055m/s, joined source319.561m/s and nominal G1 interpolation137.185m/s.
Thus source re-registration/joining substantially amplifies invalid reference
dynamics; the current QP constrains joints but not Root motion. These are
reference velocities, not measured robot velocities.

All three failures and complete physical audits are retained. No training,
output filter, actual-state overwrite or auxiliary forces were added. Failed GPU
servers/children were stopped; GPU returned to P5/300MHz/810MHz.28949/PID22856
fixed-reference comparison remains.28947/28948 are not serving a certified
controller. The next repair target is a consistent Root coordinate/temporal
contract and executable reference dynamics, followed by the same dual-stick trace.

Closeout: `simulated_xbox_preview_v36/closeout.json`; raw inputs/screenshots and
preview/physical audits are in `simulated_xbox_v1`, `simulated_xbox_flat64_v2`,
and `simulated_xbox_preview_v36`. Historical implementation results follow.

## Implementation Update: 2026-10-20

The saved three-panel reference now runs at `http://127.0.0.1:28949/`.
All15,900controlled1ms substeps match the original offline run bitwise, including
qpos/qvel, bodies, actor targets and motor forces. The visible16s episode ran
at30.016fps/1.00RTF with no dropped samples. Evidence:
`fixed_reference_realtime_v1/equivalence.json` and `browser_receipt.json`.

The active online candidate uses one continuous native forecast, plus measured
current Root XYZ/yaw frame registration. Native articulation phase remains a
forecast; G1 limbs are not relabeled as measured SOMA motion. Captured physical
history is immutable, with no command-dependent observer or retroactive pairing.
Existing G1 reference is retained through its committed prefix. Body-pose tasks
are pelvis-relative so Root trajectory corrections cannot be paid for by
distorting the limbs. All position/velocity/force and timing limits remain.

`continuous_stream_trial_v4_profile` completed28.1s with all four translation
directions, independent yaw and stop. Its28,000controlled substeps passed the
independent physical audit; minimum RootZ=.724786m. This run used the explicit
development stable-GPU scope, restored afterwards. Final3s speed was.01557m/s.
Commanded1.2m/s produced direction-projected phase averages.279-.587m/s.

The first interactive v33 reached49.38s at30fps/1.00RTF, then stopped on the
unchanged200ms pending-input bound during pure yaw. The retained log identifies
12.38ms history preparation and43.24ms retargeting in that request. Normal-power
trials v2/v3 also retain their reference-coverage failures. No automatic retry.

v34 batches quaternion interpolation and projects the frame Hessians directly
into the trajectory basis. Tests compare the original sparse objective and
constrained QP numerically. Its600ms G1 buffer is replenished in320ms increments;
the frozen MB still generates native AR8 chunks with800ms genuine coverage.
Normal-power v34 reached25.62s/30fps, then stopped on the original reference
deadline. Several82-85ms results missed the next50Hz adoption boundary.

### Interactive Result

v35 uses the same accelerated code and600ms buffer with the explicit development
stable-GPU scope. It completed142.811s of controlled physics before manual Reset:
142,811substeps/7,141numeric packets all SHA-verified,0joint/motor violations,
0pelvis/torso fall contacts,0auxiliary forces, maximum scene contact penetration
.01569138m. All21browser commands cover forward/reverse/both strafes/both yaw
signs/combined move-yaw and stop. No reference candidate was discarded.

Browser receipt at98.8s contains2,970unique submitted frames,0dropped samples,
0failures; the active8.502s window measures29.9933fps. Live RTF reads1.00.
ChainP99=83.951ms, first visualP95=222.5ms, body-onset displayP95=919.6ms.
The current responsiveness targets therefore still need improvement, particularly
physical onset; normal dynamic-power timing has not passed.

Evidence: `wasd_continuous_stream_v35/interactive_review.json`,
`browser_receipt.json`, `physical_audit.json`, `browser_running.png`.
ParentPID30992/GPU planner204 serves28948;28949 comparison remainsPID22856.
The application is reset to episode2 Ready to Start for the user's Xbox trial.
All15PFNN maps remain available; this interactive physical result is forPFNN000.
323Python and27JavaScript checks pass.0training/0pose or actuator filters.

Historical analysis follows.

## Finding

The current priority is the online observation and reference-adoption contract.
The fixed-reference video and interactive application do not exercise the same
control path. Offline reference geometry is a real issue, but its repair alone
gave only modest physical improvement and a large latency cost. New read-only
evidence locates discontinuities upstream of generation.

No training, new physics rollout, actor call, MB call, live process restart or
product configuration change was performed for this investigation.

## New Evidence

The closed v32 episode lasts24.46s and has155 input messages,303 saved generated
plans and87 adoptions. All303 plan NPZ hashes were verified. The original log
SHA256 is21732d9c503a8fc07d9e25143c8fd28bf8caf8cb7ca504c5bb38018fc324f6c5.

### Command-Dependent Observation

`FrozenMBProvider.propose` selects feedback segments using exact zero versus
nonzero translation. The moving branch corrects Hips/Chest only; the stop branch
corrects all14mapped segments. Right-stick rotation with zero translation also
uses the full-body branch.

A12case counterfactual holds the measured body, native prior, absolute times,
Root and torso correction fixed. Changing only this observer branch changes the
maximum local leg rotation in the four-frame conditioning window by median
0.388207rad (22.243deg), maximum0.598138rad (34.271deg).

This is a discontinuity of the generator input before any new physical motion.
It is not a measurement of the eventual physical response. The current trace
has3stop/move observer changes.277of303plans use the zero-translation branch;
its command mix differs substantially from the continuous forward video.

### Mixed Native Priors Inside One History

81of303conditioning histories (26.73%) contain more than one prior plan ID.
19of87adopted plans consume such a history. At a prior boundary, the median
maximum local leg step is0.090981rad; within one prior it is0.028517rad.

For13moving histories whose first prior still covers all four times, a more
controlled comparison keeps the same times and uses that one existing prior
for the uncorrected local leg channels. The median maximum leg step falls from
0.070273rad (4.026deg) to0.012995rad (0.745deg). No future motion is extrapolated.
These are overlapping windows, not13independent physical trials.
The first frame still associated with the same prior reconstructs to within
1.094e-6 per rotation-matrix element in all13cases, checking the comparison's
unchanged-history baseline against the original2e-5 rotation tolerance.

In `_append_executed`, samples initially bind to the reference that drove the
physical interval. `_pair_consumed_reference` subsequently replaces the latest
already-measured sample's native/robot prior with the newly selected plan before
that plan has advanced physics. Thus a past measurement's derived human pose
can change with plan adoption. Joint C0/C1 continuity in the accepted G1 reference
does not ensure continuity of this raw native prior.

The native-prior versus measured-contact classification differs in13.70%of4824
checked channels. This is an observer-consistency diagnostic, not an established
neural cause: current online inference extracts Root and joint positions/rotations
as conditions; those four contact channels are not directly selected into them.

### Timing And Replanning

216of303generated plans are not adopted (71.29%). This includes superseded or
stale candidates; it is not216failed physical actions. The new-command adoption
interval median is80ms; the same-command median is400ms.

| Recorded service stage | Median ms | P99 ms |
| --- | ---: | ---: |
| MB generation | 22.002 | 35.394 |
| Reference retargeting | 27.951 | 39.514 |
| Provider total | 68.124 | 88.990 |

The provider has little headroom below the100ms freshness bound, and the run
ends on the unchanged200ms pending-input deadline. Faster code would improve
this margin; it would not remove the command-dependent history discontinuity.

Only1of154input deltas is smaller than.01m/s and.01rad/s simultaneously. Tiny
analog noise is therefore not the main explanation in this recording.

## Interpretation

The generator, retargeter and tracker form a coupled feedback system. A stable
fixed-reference tracker does not establish stability when reference estimation
changes after every control input and the resulting generation changes its own
next input. The controller must preserve both physical-state continuity and the
meaning of the model's history.

The latest v31 fixed-sensor future-reference swap already reduced the median
actor target difference to0.04795rad, versus1.347rad in the older v19 test.
Consequently, the old claim that every large current oscillation is caused by
an instantaneous target switch is too broad. History semantics and physical
tracking mismatch require their own interventions.

World-Root drift is a related but distinct issue: the frozen actor does not
observe Root XY or linear-velocity targets. The three-panel cameras separately
follow each Root, so gait continuity is easier to see than world-trajectory error.
Subsequent comparison should retain a shared-world view and trajectory plot.

## Repair Order

1. Reproduce the exact saved G1 reference in the real-time server path, with the
   same initialization, warmup, timestamps, controller and physics. Display in a
   diagnostic view using a shared world frame. User input is recorded but does
   not drive this fixed-reference equivalence case.
2. Split that identical reference into timestamped packets and adopt them using
   the recorded schedule. Actor future windows must equal the complete-array
   windows at the same absolute times. This isolates transport/packet adoption.
3. Make factual history and its causal prior association immutable after capture.
   Remove command-dependent changes to what the observer treats as measured.
   Use one explicit observer throughout start, stop, strafe, reverse and turn.
   Compare constant upper-body and full-body observers as diagnostic variants;
   select a consistent mapping on physical evidence, not quiet standing alone.
4. Preserve one continuing native motion history. Retain already-committed
   motion and current support phase while updating the editable future. Use
   the checkpoint's supported four-frame conditioning; do not assume that RTC
   flow inpainting or arbitrary future-pose conditioning exists in this model.
5. Schedule generation for actual reference deadlines and meaningful intent
   changes. Maintain the latest input continuously; avoid spending most compute
   on complete candidates that the adoption rule will discard. Deadline and
   coverage checks stay explicit; missed timing remains a failed trial.
6. Add online generation, then physical feedback, then the exact recorded
   independent-translation/yaw command sequence, one factor at a time. Compare
   body oscillation, stance slip, world-Root drift, intended-direction progress,
   latency and real-time factor. Publish to Xbox only after continuous direction
   changes are stable and the visible application maintains30fps.

The immediate implementation target is an observer whose output for an already
measured state is independent of a newly arrived joystick command or plan ID.
Reference contact optimization follows once the online baseline is consistent.

## Literature

- [MotionBricks, Appendix C](https://arxiv.org/html/2604.24833v1#A3): more frequent
  replanning degrades discrete-latent joint jitter in its ablation; it recommends
  a practical3-9frame interval plus command-triggered updates. These settings
  need testing against our different model and generator-tracker interface.
- [RTC](https://arxiv.org/html/2506.07339v2): preserve the committed prefix and
  condition the editable future for asynchronous consistency. Its published
  implementation targets diffusion/flow policies; our discrete MB needs an
  explicitly supported conditioning/execution mechanism.
- [MGMT](https://arxiv.org/html/2604.17335v1): the system conditions generation on
  actual robot history and fine-tunes its tracker with the frozen generator in
  the loop. It also exposes reference velocity and terrain to its tracker.
  This explains why copying the block diagram alone does not establish the
  same learned feedback behavior. Current work remains zero-training.

## Artifacts

- `out/mb_sonic_e0_e3_20261018/v32_history_contract_review_v1.json`
- `out/mb_sonic_e0_e3_20261018/v32_observer_mode_counterfactual_v1.json`
- `scripts/analyze_mb_online_history_contract.py`
- `scripts/probe_mb_history_mode_switch.py`

Live28948 remains the unchanged v32 research baseline. The two scripts read
closed artifacts and write new exclusive diagnostic reports only.
