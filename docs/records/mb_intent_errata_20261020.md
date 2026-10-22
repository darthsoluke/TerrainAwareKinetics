# Intent And Stopping Errata

## Corrections

2026-10-20 23:07: The Root-behavior probes below used incorrect terrain/reference
registration. They do not isolate the learned Root model from its input adapter.
The [registration repair](mb_scene_registration_repair_20261020.md) removes
metre-scale coordinate contamination, restores the learned Root path in v49,
and retains the earlier bounded-Root experiments as historical evidence.

The v39 post-trace mean planar speed0.1558m/s includes oscillatory body movement.
Its net displacement is[-0.2668,-0.2716]m over approximately104s, or0.00366m/s
net drift. It is incorrect to describe the entire period as continuous travel
at0.156m/s. The initial stop phase is a different problem: it travels
[-3.5281,2.7458]m during the evaluated7.58s. Both values remain in the reports.

`review_mb_gamepad_trial.py` now reports instantaneous speed and net drift
separately, and reports reference progress alongside physical progress. The
original reports are retained. The corrected v39 report is `gamepad_review_v2.json`.

The current runtime uses actual G1 Root as the scene-query anchor; its motion
context remains the accepted native forecast. It does not feed measured full G1
body history into the SOMA generator. ProtoMotions consumes future joints and
torso orientation; its pinned input has no desired Root XYZ/linear velocity.

## Stage Isolation

Five saved histories from v39 were evaluated on CPU with frozen weights, holding
the saved history heading and using the recorded translation command. These are
controlled probes, not byte-exact reproduction of the original integrated yaw.
No new physical steps or training updates occur.

|Saved time|Requested XZ velocity|Root mean speed|Decoded mean speed|
|---|---|---|---|
|6.14s|[-0.0013,0.7826]m/s|0.7655m/s|0.7467m/s|
|20.20s|[0.7227,0.0012]m/s|1.9770m/s|1.9482m/s|
|25.16s|[0.0012,-0.7227]m/s|1.9725m/s|1.9488m/s|
|72.34s|[0,0]m/s|1.1202m/s|1.1064m/s|
|100.20s|[0,0]m/s|0.9365m/s|0.8571m/s|

The inappropriate motion is already present in Root prediction. Codec output
closely follows that motion. Evidence: `intent_stage_probe_v1.json` under
`out/mb_sonic_e0_e3_20261018/`.

## Reference Planning Candidate

v42 adds explicit planar Root planning before Pose and codec conditioning.
Known four-frame history stays fixed. A constant bounded acceleration connects
the last known planar velocity to the latest requested velocity. Future height
and heading remain Root-network predictions. Pose then receives the planned
global/local Root conditions, and the original decoder generates the body.
The decoded motion still goes through the existing reference constraints.

This changes planar reference generation: the learned Root network no longer
chooses the future XZ path in this candidate. It continues to predict height and
heading. This tradeoff must be checked on terrain before deployment. Acceleration
is explicitly2.5m/s2, with no trained-weight changes or actual-state overrides.
No motor or rendered-pose filter is introduced.

On the same five CPU probes, the two stopping cases decode mean speeds0.1803 and
0.1325m/s, including the initial braking motion. Evidence:
`intent_stage_probe_bounded_v2.json`. Four planning tests and183targeted runtime
tests pass. Tensor planning is included in the existing graph preparation and
native/compiled checks.

## Physical Comparison

v42 completes all83s of the same browser stick trace and123.305s total. All
123,205controlled substeps pass the existing physical audit: no falls, actual
joint/motor violations, hidden forces or unknown ground. Maximum scene penetration
is1.345cm. Display29.99365fps,0drops, RTF1.00 under the same explicit DevGPU scope.
Chain P99=84.921ms; visual plan association P95=198.5ms.380Python regression tests
pass; one additional diagonal world-frame invariance test also passes (381total).

|Metric|v39|v42|
|---|---|---|
|Forward actual progress|0.248m/s|0.441m/s|
|Reverse strafe actual progress|-0.194m/s|0.0667m/s|
|Rapid reversal actual progress|-0.0868m/s|0.0580m/s|
|Stop instantaneous speed|0.624m/s|0.0543m/s|
|Stop net drift speed|0.590m/s|0.0101m/s|
|Post-trace net drift speed|0.00366m/s|0.00015m/s|
|Rapid-reversal joint8-14Hz energy|0.00509rad2|0.00002rad2|

The progress measurements average projection onto each current command direction,
after the first400ms of the phase. They do not establish response latency. Right
strafe and backward speed are now substantially below their requested0.723m/s:
actual progress0.192/0.182m/s, while reference speed averages0.672/0.669m/s. The
remaining mismatch reaches the tracker/body-execution layer. Full direction/speed
control remains open, despite improved stopping and reduced reversal oscillation.

Evidence: `simulated_xbox_root_plan_v42/browser_receipt.json`, `physical_audit.json`,
`gamepad_review.json`, and `completed.png`. Native/compiled validation is saved in
the runtime preparation record.

## PFNN Regression And Live State

v43 PFNN000 completes the entire83s trace and about137s total, with136,968
controlled substeps passing the full physical audit. No fall, unknown ground,
actual joint/motor violation or auxiliary force. Maximum scene penetration is
4.372mm. Display29.994fps,0drops, RTF1.00; chain P99=86.233ms and visual plan
association P95=178ms. The prior28.279s boundary failure is preserved in v41.

PFNN stop-phase instantaneous speed is0.0481m/s; post-trace neutral0.00409m/s.
Directional progress is0.495m/s forward,0.230right,0.073reverse strafe,
0.114backward, and0.036rapid reversals. The sideways/backward tracking shortfall
and first visual latency above100ms remain unresolved. Only map000 and the
separate64m flat arena were exercised in this turn;15PFNN maps remain selectable.

Active package `interactive_root_plan_v43`, server29536, planner19844, port28948.
The selected runtime source hashes equal the v42 audited package. The normal
Xbox/WASD page is reset to Ready, left translation/right yaw; simulated input is
off. The explicit DevGPU scope remains active while the service lives. Previous
v41/v42 GPU processes were stopped and residency recorded. Total training updates
and formal exposures added remain zero.
