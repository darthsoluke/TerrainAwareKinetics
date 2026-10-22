# PFNN frame actor P1 endpoint review

2026-10-22. User requested continuation. This review uses the saved P1 endpoint
on PRO6000 GPU3 with zero optimizer updates. GPU0/1/2 remain reserved.

## Training closeout

The external eight-hour timeout stopped training at12:48:20 Singapore time.
Progress records1774 updates/453262 windows/3626096 next-frame targets. The
checkpoint contains1652 updates/422030 windows, seven complete coverage epochs.
Each completed epoch includes48232 primary rows and12058 balanced extras.
Checkpoint SHA256: `3685e51e17a7273be0c88e19e1490c57ef36a7e663e1105ad6194e9af32ca1d9`.

The remaining122 recorded updates/31232 windows were consumed but not saved.
The runner saved weights only at epoch boundaries; external SIGTERM arrived
before its independently started internal deadline and finalization. No P1
relaunch, recovery or budget extension was performed. In-flight work beyond the
last recorded update is not certified. Original38.4M accounting is separate.

Local runner repair now atomically saves each completed update before publishing
its progress. Interrupted writes leave the previous checkpoint intact.65 CPU
checks passed, including interruption after first progress and during a write.
The completed remote training snapshot remains unchanged. Future training uses
a new source-bound preflight; old proofs cannot certify this local runner edit.

## Fixed validation

All192 original validation identities were reused. Each mode executes26 actual
next-frame transitions per window. Commands are the same source-derived intent
labels; sensing always uses the current history and the predicted Root path.
Teacher modes replace history with the recorded four-frame window; closed mode
feeds back generated FK history. Recurrent memory persists through all26 steps.

| Metric, window mean | Initial teacher | Trained teacher | Trained closed |
| --- | ---: | ---: | ---: |
| Root position error, cm |2.300 |0.277 |8.342 |
| Root endpoint error, cm |2.321 |0.257 |18.813 |
| Root-relative joint error, cm |1.546 |0.756 |10.555 |
| Relative joint velocity RMSE, m/s |0.195 |0.160 |0.392 |
| Predicted/source relative velocity RMS ratio |0.980 |1.047 |0.693 |

Teacher Root and body errors improve in all six behavior categories and both
represented terrain types. Closed-loop errors accumulate in every category.
The velocity RMS ratio measures overall body motion magnitude over these short
windows; foot swing and longer gait cycles still require their own review.

The existing validation families contain flat and rocky terrain only. Jumpy and
beam are present in the training split. Preserve the family split and use a
separate, explicitly training-split terrain-fit diagnostic for their P1 gate.
Do not relabel it as held-out generalization or select on the test split.

Additional training-fit review completed92 fixed windows, four per populated
terrain/behavior stratum, selected with RNG3324 before model evaluation. All23
populated strata were retained; beam has no idle rows. These are augmented
windows, not92 independent captures. No new training or test-set use occurred.

| Training terrain | Root error initial -> trained, cm | Body error initial -> trained, cm | Trained body velocity RMS ratio |
| --- | ---: | ---: | ---: |
| beam |1.374 ->0.273 |1.407 ->0.704 |1.034 |
| flat |2.167 ->0.244 |1.300 ->0.524 |1.016 |
| jumpy |1.723 ->0.193 |1.228 ->0.499 |1.024 |
| rocky |1.596 ->0.249 |1.369 ->0.632 |1.054 |

P1 has evidence of improved one-step fit and preserved overall short-window
motion magnitude. Its closed-loop results identify the failure P2 must address;
foot-specific swing/stance measurements and the explicit P2 entry review remain
part of P2 preparation, rather than treating mean fit as delivery approval.

## Controlled rollout

The unchanged64m flat fixture was transferred through private Drive; scene SHA
`5c4b2049a2801f8be4268da42dec730f5094e9b992910acd661d0c21027f1464`.
Controls reuse the prior16s idle,16s start/stop and18s independent-facing task
definitions. The frame actor consumes the requested control every frame; its
history and recurrent state are never reset on command changes.

| Trial | Control screen | Peak penetration | Remote compute P99 |
| --- | --- | ---: | ---: |
| Idle16s |Fail; endpoint drift4.462m |29.47cm |2.81ms |
| Start/stop |Fail; no sustained stop during8s stop phase |22.51cm |2.83ms |
| Independent direction/facing |Fail; lateral/reverse speed inadequate |5.41cm |3.00ms |

All1500 emitted frames and the three genuine four-frame seeds are retained.
Geometry uses18056 official vertices and8 swept substeps, with zero unknown
body samples and zero uncertain interior samples. All three exceed2cm.
These are offline PRO6000 compute timings; local browser/Xbox timing is pending.
The control failure occurs with fast inference and freshly sampled observations.
The main next target is feedback stability, braking and gait retention.

Controls v1 stopped before rollout because CUDA replay requires inference mode;
v2 fixes that evaluator context and completes all three trials. Both records are
retained. Model/checkpoint hashes are unchanged after both completed reviews.

## Next stage

13:29: Foot-span/stance review and the separate P2 objective implementation are
now recorded in [P2 preparation](pfnn_frame_actor_p2_preparation_20261022.md).
No repeated model rollout or training was used for the new foot measurements.

1. Review the completed fixed validation and training terrain diagnostics above;
   retain missing strata and add foot-specific swing/stance measurements to the
   P2 entry review. The original validation families remain unchanged.
2. Implement P2 scheduled generated-history training with4/8/16-frame rollout,
   fixed source-history supervision and persistent recurrent state per rollout.
3. Review target semantics before launch. With generated history, current P1
   `nexttruth - history_root` would turn accumulated absolute position error into
   a velocity target. Use observed source displacement targets for the commanded
   motion; do not introduce an invisible absolute path target on flat terrain.
4. Review the boundary-velocity loss: subtracting the same generated boundary
   from prediction and target cancels that boundary in the loss. P2 needs actual
   generated frame-to-frame velocity versus source frame-to-frame velocity.
5. Bind a new P2 plan, finite budget, source snapshot, zero-update proof and
   separately accounted smoke before any P2 execute. Keep the prior model/page
   independent until long control, full skin/stance and local30Hz gates pass.

No P2 training has started. No model was promoted.

## Evidence

Local evidence root:
`out/pfnn_mixed_20261015/pfnn_frame_actor_20261022_v1/p1_zero_update_review_v1/`.
It contains the P1 checkpoint/logs/closeout, fixed window trajectories/reports,
both control-attempt records and all three controlled trajectories.

Private-Drive archive `p1_zero_update_review_v1.tar.gz`:35873562bytes, SHA256
`ee7650e1000c99353dd956ae62afcdd394d98a7ea04a5e97c076bad4c36d6dc3`.
Local archive hash matches remote. Existing data/cache/retargeting were not rebuilt.

Training-fit delta archive `p1_terrain_review_v1.tar.gz`:6850963bytes, SHA256
`e6448f3d99b5b2b0704affe627a87d59e0ddd9b30f05b3116a7617917fdd7d4c`.
The matching local directory includes92-window trajectories, selection/counts,
evaluation source versions and the original training runner/plan. It was also
transferred through private Drive and hash-verified locally.
