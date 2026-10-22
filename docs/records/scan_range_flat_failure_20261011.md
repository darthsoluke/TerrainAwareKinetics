# Flat-Ground Failure From Expanded Scan Coordinates

The interactive demo previously sent raw 20 m / 6 m-half-width survey coordinates
into an encoder trained on at most 3.4 m forward extent and 0.6 m half-width.
Passing shape/gradient/compiled-parity checks did not validate this input change.
Even a constant plane triggered arbitrary Root and Pose changes. This was an
integration error in the expanded runtime controls, not evidence of unseen walls.

## Paired Reproduction

Checkpoint: A100-B, SHA256
0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d.
The same genuine four-frame BONES standing seed and identical commands are used
on an exact world-Y=0 plane. Prediction horizon is 12 tokens, four argmax passes.
Both straight sampling and one decoded-route refinement are assessed on CPU in
full precision. No live session is used or modified by the numerical probe.

`out/a100_live_20261010/scan_range_flat_probe_v1.json` records the failure:

- Zero command, one route refinement, 3.4 m scan: final Root rise 0.0112 m.
- Same conditions, 20 m scan: final Root rise 0.9393 m; decoded final XZ displacement
  approximately [0.168, 0.517] m despite zero input.
- Eight seconds of generated-history zero-input rollout at 20 m displaced the
  body approximately [6.106, -0.022, 9.356] m. Horizontal distance was 11.17 m.
- At 1.9 m, the same rollout displaced it approximately [-0.283, 0.019, 0.118] m.
  This already has 0.31 m of residual horizontal drift, so it is not a good idle.

The reproduced decoded artifact is primarily horizontal displacement. This probe
does not claim that every user-observed upward decoded trajectory was reproduced.

## Corrected Runtime Contract

Far geometry surveying remains available at 20 m and 6 m half-width. The current
checkpoint receives an independently resampled, metric-correct motion observation
bounded by the original training protocol: 3.4 m forward and 0.6 m half-width.
No far XYZ is compressed into a near patch; that would move physical obstacles.
The curves, actor outputs and terrain heights are never projected or overwritten.
The existing predicted-curve refinement operates on the bounded motion patches.

Diagnostic patches explicitly distinguish `intent` (actual model input) and
`survey` (`survey_only_not_model_input`). The viewer exposes both layers and the
actual motion-view extent. The distant survey is a preview until matching
long-range observation training and a bound model artifact are available.
Thus this correction prevents unsupported far inputs from disturbing the current
controller; it does not give the old checkpoint a learned long-range planner.

## Verified Result and Limits

`scan_range_flat_probe_v2.json` assesses the same interventions after the fix.
At 3.4, 6, 12 and 20 m requested scans, decoded outputs are exactly identical
under each paired history/command/refinement case. The 20 m and 3.4 m eight-second
feedback trajectories also match exactly (maximum XYZ difference 0.0 m).
The 20 m zero-input result becomes approximately [-0.203, 0.021, 0.093] m,
or 0.22 m horizontal displacement. Residual idle drift remains unresolved.

Focused tests check real far-survey coordinates/heights, exact equality of the
model inputs at 3.4/20 m, preservation of source history, and shared playback
contracts. The compiled demo checks native token/FK parity before serving. Browser
checks verify both diagnostic layers and real slider application.

Final verification: 27 focused tests passed. Ten compiled/native startup cases
preserved active tokens, with maximum FK difference 15.28 micrometers. Independent
live compiled sessions at 3.4 m / 0.6 m half-width and 20 m / 6 m half-width produced
byte-identical SOMA rig packets through 30 committed blocks (eight simulated seconds).
Evidence is `scan_range_live_review.json` and `scan_range_browser_review/checks.json`.

Obstacle braking, contact support and continuous collision-free traversal remain
separate missing controller capabilities. They require matched training through
Root, Pose, decoder and generated FK history, with explicit scene-aware objectives
and fixed closed-loop obstacle/start-stop tests. This range repair is not their
acceptance result and does not justify claiming PFNN-level terrain control.
