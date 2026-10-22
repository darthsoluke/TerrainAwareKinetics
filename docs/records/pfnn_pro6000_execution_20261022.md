# PFNN frame actor migration to PRO6000

2026-10-22. Latest real user: move from A100 to PRO6000; synchronize data through Google Drive or another route that does not carry data over the A100 public IP.

## Current work

**19:53 A/B/C closeout reviewed:** [physics findings](pfnn_physics_findings_20261022.md)
and [handoff](pfnn_physics_audit_status_20261022.md). The handed-off v3 arms and C
auto-evaluation are complete; do not rerun the old waiter. C12updates/24576windows,
v3 total36updates/73728windows, separate from P1. A new frozen zero-update review
used a shared source-support mask and14 valid-command frames, retaining all raw
trajectories. Foot-speed P50 A/B/C=27.938/31.240/31.778cm/s versus source2.533.
Original mask, initialization, effective sample-size and loss-sensitivity claims
were corrected; wrench rankings are withheld pending evaluator repair.
6local checks pass; both review packages privateDrive/hash-verified locally.
No new training/deployment. Review job exited,GPU3 released,soleSSH62753 closed.
Local evidence: out/pfnn_ablation_review_20261022. Original remote run/source
snapshots and checkpoint identities are preserved in original_snapshot locally.

**13:29 P2 preparation:** [new local objectives and foot review](pfnn_frame_actor_p2_preparation_20261022.md).
CPU analysis reused all saved trajectories,0model inference/0optimizer updates.
Validation teacher foot span ratio median1.003 is preserved, but source-stance
sole speed worsens3.66->8.75cm/s; closed-loop reaches42.36cm/s (source2.48cm/s).
The source-certified masks are unchanged. This confirms a temporal/contact issue
in addition to aggregate pose fit; no candidate promotion or repeated rollout.
New separate `pfnn_frame_closed_loop.py` implements displacement-relative targets,
actual-boundary velocity, relative path labels and whole-row mixed-history4/8/16
rollout with temporal gradients.8CPU tests pass. P2 runner integration and GPU
proofs remain pending; draft plan proposes10epochs/602900windows/12h onGPU3 only.
No P2 preflight/smoke/training launched. The CPU gait report was transferred via
privateDrive and hash-verified locally. Sole SSH45034 closed; no active task SSH.

**12:49: P1 stopped at its external8h cap (12:48:20), both training PIDs absent.**
Final progress records1774 optimizer updates/453262 windows/3626096 next-frame
targets. Seven complete coverage epochs were saved. The available checkpoint
contains1652 updates/422030 windows (epoch6), SHA256
`3685e51e17a7273be0c88e19e1490c57ef36a7e663e1105ad6194e9af32ca1d9`.
The last122 recorded updates/31232 windows belong to the interrupted epoch and
are absent from the checkpoint; retain them in the consumed-exposure ledger.
External timeout sent SIGTERM before the internal deadline checkpoint could run;
there is no complete.json. This is a shutdown/checkpoint-timing defect, not a
numerical failure. Do not restart/repeat P1 or silently recover its missing tail.
User said continue; the zero-update review is now complete. See
[P1 endpoint review](pfnn_frame_actor_p1_review_20261022.md).
Original192 validation windows: trained teacher Root/body errors0.277/0.756cm;
26-frame generated-history endpoint Root/body errors18.813/10.555cm. All three
long control trials fail control/skin, with idle drift4.462m and peak penetration
29.47/22.51/5.41cm. Remote graph compute P99=2.81--3.00ms. No deployment.
Separate92-window train-only terrain diagnostic improves one-step fit in all4
terrain kinds, with overall body velocity ratios1.016--1.054. Beam has no idle
stratum; original validation only contains flat/rocky. Do not alter splits.
All reviews have zero optimizer updates and unchanged checkpoint/model hashes.
Local future runner now saves atomic checkpoint before publishing each update;
65 checks passed. Remote P1 snapshot was not modified. P2 needs explicit
generated-history target/loss review, foot swing/stance entry checks, a bounded
plan and new source-bound proofs; no P2 training has started.
All review processes exited and GPU3 was released. Sole SSH7046 closed normally.
Both review archives were downloaded through private Drive and hash-verified;
paths/hashes are in the review document. GPU0/1/2 remain excluded. The30minute
heartbeat now follows P2 preparation instead of polling the ended P1 process.

Latest04:50, overriding all earlier allocation notes: **use only physical GPU3**
(`GPU-940a5466-9a53-26aa-c2eb-5496694d8c3b`). GPU0/1/2 belong to other programs,
even when idle. Do not run the old two-GPU hardware helper or any two-card launch.
New plan `configs/pfnn_frame_actor_pro6000_gpu3_v1.json` uses world_size1,
global256/micro256, unchanged10epoch/602900-window and28800-second caps.
The launcher accepts explicit `--gpus 3` and refuses substitution.91 focused
local runner/allocation checks passed. Runtime/statistics/initial-model identities
are now bound into the new preflight proof; model and losses are unchanged.

**All transfer hashes passed on PRO6000.** Final receipt:
`${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022/transfers/assets_v4/complete.json`.
Data root is its `data/` directory, containing `dataset_world_v2`,
`cache_pfnn_world_v2`, `parent/weights.pt`, `stats`, `code_base` and `proofs`.
No dataset regeneration occurred. The third receiver had exited while leaving a
stale downloading state; attempt4 reused verified archives and ran in its own
session (timeout1791673, child1791674, SID1791673). It has completed and exited.

PRO source is `${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022/code_pfnn_frame_gpu3_v1`;
run root is `runs/pfnn_frame_gpu3_v1` under that project. Code is copied from the
verified base and patched with `gpu3_overlay_v1/` from Drive. Overlay archive SHA
`611a831f07357dc5104a7536bda0d0a30fc2af66f8dd6e113602a0766c2f5ab8`.
Missing application dependencies were installed in the independent environment,
including peft0.21.2 and accelerate1.15.0. **198 CPU checks passed.**
The existing CUDA library versions were preserved; their seven pip exact-pin
differences did not prevent the actual GPU3 checks below.

**GPU3 preflight and Adam smoke both passed; P1 launched at04:48:20.**
Preflight `preflight_v1/complete.json` records zero optimizer updates and unchanged
initial/final model digest
`b1da12f6d1b860493b6d660a470754f31b6a06f01825de4f538f47691df6cb0b`.
Its first initialization was slow, then completed normally; no repeat is needed.
`adam_smoke_v1/complete.json` records exactly2 AdamW updates,512 window exposures,
4096 supervised next frames and changed final digest
`2b7a30095f21ad9eaff260de8aac9e9864caeeab4e79aa0b7dd8651bb2b222d1`.
Smoke batches took17.39/17.55s with peak6.81GB; the complete run took69.54s.
Both proofs bind source, plan, parent, audit, statistics and runtime. Smoke has
exited; its former timeout PID2457364 is absent.

P1 output: `runs/pfnn_frame_gpu3_v1/p1`; timeout PID2482846,28800-second cap,
single worker bound to physical GPU3 by UUID. Launch receipt `p1.launch.json`
contains the fresh complete GPU/PID scan. GPU0/1/2 remain excluded even though
the scan found no compute jobs. P1 initializes afresh, not from the smoke weights.
First-update verification passed: worker PID2482964 completed4 updates,
1024 windows and8192 supervised next frames by elapsed91.79s. Loss0.40642,
gradient norm0.59836, latest batch17.34s, peak allocated6.81GB; no failure files.
Preflight/smoke/P1 bindings and initial digests match exactly. `/proc` confirms
the worker's UUID mask selects GPU3 only; nvidia-smi confirms its real compute PID.
At the measured smoke throughput the8h
cap can be reached before10 epochs; record actual coverage and do not extend it.
P0 accounting is now4 total optimizer updates/1024 windows across A100 and PRO;
P1 has a separate ledger. The original38.4M ledger is unchanged.

Heartbeat12:22: same timeout2482846/worker2482964 remain alive with the declared
P1 command. Progress increased to1688 updates/431246 windows/3449968 next-frame targets,
elapsed27256.47s; loss0.04983, finite pre-clip gradient norm1.22357, batch17.76s. No failure
files; epoch7 remains in progress. All seven completed coverage receipts confirm48232/48232
primary rows plus12058 extras per epoch, no missing/duplicate primary rows.
The12755041-byte epoch6 checkpoint SHA256 is
`3685e51e17a7273be0c88e19e1490c57ef36a7e663e1105ad6194e9af32ca1d9`.
Fixed192-window validation total losses after epoch6: idle0.036212,
start0.042580, stop0.043558, strafe0.063582, reverse0.066564, forward0.043462.
The external8h deadline remains12:48:20. On the next check, distinguish final
progress exposure from the saved checkpoint exposure if timeout interrupts an
epoch. Do not resume or extend training to complete the remaining epochs.
Earlier fixed192-window validation total
losses (initial -> epoch0) are idle0.11284->0.09025, start0.19014->0.09755,
stop0.31094->0.11769, strafe0.44427->0.15340, reverse0.51386->0.16594,
forward0.52030->0.11023. Idle body-FK loss increased0.01261->0.01375; retain
per-category metrics for final review. No long-horizon control/skin gate run yet.
All four GPUs and compute PIDs checked; our sole compute PID remains on3.
Unrelated Blender PIDs1658192/1984435/2515895 occupy0/1/2,
and all0/1/2 remain reserved. No relaunch, transfer, source modification or budget change.
One SSH session87365
was opened for this check and closed normally; unrelated host SSH was untouched.

**M78:** existing PRO profile started with its stored account, localhost37890,
system proxy/TUN/system DNS disabled. Its original node was SingaporeAWS03 direct;
currently HongKongAWS03 direct. The isolated node-benchmark UI could not retain its
profile and was stopped; no unrelated proxy/service was changed. On Windows, the
existing M78 client was changed to HongKong01 IEPL x2, after comparing it with
HongKong02 residential. Inbound32MiB probes all returned identical SHA256
`53fcd49ce55405b224d97033835237f1b5fe27d7d3ebe77b4d5df5f992eb9161`:

| Measured path | MiB/s including setup |
| --- | ---: |
| PRO local M78 SingaporeAWS03 direct |4.378 |
| PRO local M78 HongKongAWS03 direct |4.218 |
| Windows local M78 HongKong01 IEPL x2 |6.859 |
| Windows local M78 HongKong02 residential |4.968 |
| PRO via task SSH forwarding to Windows M78 HongKong01 |4.104 |

No4--5x end-to-end improvement was established by these probes. Keep measurements
separate by host/path. Do not claim the faster local probe equals PRO throughput.
All bulk migration used private Drive. Source A100 public SSH carried commands
and small status only. Windows route evidence is saved locally as
`out/pfnn_mixed_20261015/pfnn_frame_actor_20261022_v1/m78_local_hk1_selected.png`.

Task SSH session69845 was closed normally at04:50 after checking the live P1
worker. The detached timeout/worker have their own sessions and continue running.
The comparison-only localhost37893 reverse forwarding closed with SSH. Normal
PRO operations use its own M78 localhost37890 and do not depend on that forwarding.
Previous sessions19124/2014/98457 are also closed. NoVNC temporary tab was closed.
At the next check, establish one persistent connection and reuse it for all work.

Local28944 and28950 currently have no listening process (04:49 health check).
No local model was replaced or service restarted during this transfer/training
task. Recheck the saved launch configuration before the next local control trial.

## Earlier migration notes

At02:20: all A100 archives were uploaded to private Drive and manifest-ready.
Source upload timeout PID2785743 published the final manifest before02:06.
No A100 file payload passed over SSH; the staging helper itself was retrieved from Drive.
Local manifest: `out/pfnn_mixed_20261015/pfnn_frame_actor_20261022_v1/pro6000_transfer_manifest.json`.

| Package | Bytes | SHA256 |
| --- | ---: | --- |
| dataset_world_v2.tar.gz |753707623 |05c87be2855e96cb178f743cf5c0f76443a1b9af69e5bb6bb6548389c2af829a |
| cache_pfnn_world_v2.tar.gz |8525195339 |a41e0ca947b43af5a6e21ef16f6bc458eb6c379bbc48b9b6532ae1a608d1d3ec |
| runtime_assets.tar.gz |1812908156 |fc626ed1cb3f83510b7ca20301022b20efea279f5d8c5de7596ee1ff796c1daa |

Drive root for this transfer: `private-store:motionbricks/pfnn_frame_pro6000_20261022/`.
Archives/ready manifest are under `assets_v1/`; validated PRO6000 launcher/test overlay is under `overlay_v1/`.
The new launcher has25 focused tests; together with runner checks56 tests pass. Model/training objective remains unchanged.

**Active connection:** PRO6000 persistent SSH exec session19124, local sshPID8572,
with `-R 127.0.0.1:27897:127.0.0.1:7897`. This is the only task SSH connection;
A100 is disconnected. Keep this connection alive while receiver/dependency transfers need the proxy.
PRO6000 direct Google access stalled. The task-specific reverse forwarding uses the user's existing7897 proxy,
with strict SSH host verification. Data travels GoogleDrive -> proxy -> PRO6000, never through A100 public SSH.

PRO6000 project and own OAuth config are installed. Independent base Python/13GB package copy completed;
Torch2.7.1+cu128 imports. Original cloth environment is unchanged.
Initial ordinary pip attempted to replace CUDA libraries; only our installer was stopped before installation.
Application root packages were installed using `--no-deps` to preserve the existing CUDA base.
Missing application dependencies are being filled from pinned A100 versions.
The copied base has CUDA12.9 runtime/cuBLAS/NVRTC alongside Torch2.7.1+cu128,
so pip reports seven CUDA exact-pin differences; actual GPU compatibility still requires preflight.

Receiver attempt1 (`transfers/assets_v1`) failed Google API EOF before any archive download; evidence retained.
Receiver attempt2 runs with `GODEBUG=http2client=0`, timeout PID1343346 / Python PID1343347,
output `${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022/transfers/assets_v2`, log `logs/drive_receive_v2.log`.
It downloaded the ready manifest and started the real753MB dataset package. Check `status.json`,
`failure.json`, `complete.json`, actual process and archive progress before any action; do not duplicate it.
Latest GPU snapshot had compute jobs on1/2, desktop graphics on0, and only3 genuinely idle.
No PRO6000 optimizer updates or P1 launch have occurred.

- A100 P1 has never started. Its pending launch is superseded. Original P0, full data audit, local frame budget and two-update Adam smoke remain preserved in [the execution record](pfnn_frame_actor_execution_20261022.md).
- Use two RTX PRO6000s with the same P1 limits: global256/micro128, at most10 epochs/602900 windows and8h wall/16 aggregate GPU-hours. No additional training exposures have occurred on PRO6000.
- Latest PRO6000 inventory: four RTX PRO6000 Blackwell Max-Q cards, each97887MiB. All four had active cloth compute processes; do not stop or modify those jobs. Rescan before any GPU launch.
- Storage before preparation: root filesystem1007G with608G free; `/data`6.0T with224G free; NAS17T with16T free. Project `${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022` is on root.
- The old private Drive `private-store:motionbricks/pfnn_mixed_20261015/` has source ZIP and final acceptance assets; use the new transfer folder above for full training files. Public A100 SSH carries control commands/status only, never file payloads.
- Local rclone: `out/tools/rclone/rclone-v1.75.2-windows-amd64/rclone.exe`; own OAuth config `${USER_HOME}/.codex/secrets/motionbricks-drive/private-client.conf`; local proxy7897. No shared OAuth client.
- PRO6000 rclone exists at `${PRIVATE_STORAGE}/.local/opt/mocap-drive-delivery/bin/rclone`. Its existing unrelated config must remain unchanged. Install a separate own-client config in the new project's `private/` directory without exposing values.
- Existing PRO6000 Python3.11.16/Torch2.7.1+cu128, numpy1.26.4, scipy1.13.1 and Warp1.17.0 are under `${PRIVATE_STORAGE}/hoi/pipeline_stages/cloth/runtime/venv`. Plan an independent copy/new environment, not modifications to that active environment. Base Python is `${PRIVATE_STORAGE}/hoi/pipeline_stages/cloth/runtime/pythons/cpython-3.11.16-linux-x86_64-gnu`.
- Local launcher now supports explicit `--gpu-family "RTX PRO 6000"`, inactive display, no compute PID and strict idle thresholds. It binds selected cards by UUID. No model, training objective or dataset changes requested for this migration.
- Automation `pfnn` was updated to this PRO6000 task, every30min and quiet without important changes.

## Connection rules

Use one persistent SSH connection at a time. PRO6000: `private-user@remote-host.example:12222`, local key `${USER_HOME}/Downloads/SSH_PRIVATE_KEY`, strict existing host verification.
The first PRO6000 session was closed by the remote host during preparation; no transfer or GPU launch was confirmed. Reconnect without weakening host checks. Before switching to A100 for upload commands, close PRO6000; background Drive operations do not require an SSH connection.

A100 control endpoint and private Drive configuration are recorded in the existing execution/history. Set `TMPDIR=${PRIVATE_STORAGE}/mbtmp` before any A100 Python import. No SCP, SSH binary/base64 file streaming or port-forwarded A100 file service for this migration.

## Remaining work

1. Follow existing P1 timeout PID2482846 and its actual worker, `p1/state.json`, `p1/progress.jsonl`, epoch checkpoints and completion/failure evidence. Do not launch another P1 or repeat the passed preflight/smoke.
2. Preserve the explicit GPU3 selection, source/runtime bindings and10epoch/602900-window/8h limits. No automatic resume, retry or budget extension.
3. On completion, record actual optimizer updates, exposures and completed coverage separately from both smoke runs and the old38.4M campaign.
4. Continue original control/geometry/local Xbox acceptance gates; no automatic candidate export. GPU0/1/2 remain unavailable to this task at every stage.
