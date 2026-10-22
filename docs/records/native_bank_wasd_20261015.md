# Native Bank WASD Trial

## Current Full World

The user requested the complete map. Port28942now starts with
`--world out/a100_live_20261010/terrain-world.npz --execution verified-cuda`.
It uses the existing256x256m authored heightfield, its verified render mesh,
37spawn entries and the real four-frame BONES idle seed. Default entry is
`zone-03`(Stairs16cm). All native model tensors and trained inference defaults
remain unchanged. Source-stair mode remains available without`--world`.
This authored map has no overhangs; loading it is not a general-quality claim.

Both display and model use the same full-world terrain. Existing map loading
validates the scene/mesh hashes and seed provenance. All five full-world CUDA
preflight commands produced identical active tokens; maximum FK difference is
7.87233e-6m, within the existing3e-5m numerical bound. The four native live
contract tests pass. Runtime config is saved as`full-world-config.json`in the
live artifact directory; scene SHA is
`e19876a349c54ba4a2c663b358f9e33bc0bab2e115f302e91ff3f63b7f819ffa`.

The browser visibly loaded the full arena and37entry options. The user began
controlling the page during verification, so no automated reset/spawn switch
was performed. Live telemetry observed29.993fps, zero buffer underruns and
16commit-wait ticks cumulatively; this is not a zero-wait claim. The recording
`full-world-browser.json`and screenshot`full-world-live.png`preserve a later
snapshot. SDF telemetry reported1536/1536known samples. This checks live map
integration, not a new exhaustive gait/penetration review.

Previous28942PID43628was checked by cmdline and stopped. Current process hint
isPID5396; verify cmdline before operations. Logs are
`native-bank-world-28942-v1.log`and`.err.log`in the run root. Existing28941and
28936were not restarted. Browser tab10is marked deliverable and the user's
position/input settings were left intact.

The user visually accepted the orange native result and requested a WASD
controller with correct inference. This explicitly authorizes an independent
interactive trial despite the remaining quality limitations in the preceding
research report. Preserve28941and28936;28940remains closed.

## Artifact

The full model is `unified_native_pose_fullbank_20261015_v2/weights.pt` from A100,
training checkpoint SHA
`03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1`.
The private live package is `out/parc_rugged32_20261013_v2/native_bank_live_20261015_v1`.
Optimizer state was removed; every Root/Pose/codec/adaptation tensor is unchanged.
Compact checkpoint SHA
`631d760c59d5f59bab3b795c11867f69e91c4704df0aa75270ba4c18443bed23`.
Full tensor digest
`d89017a4ff8e22979edb4971f6f4f66487e1277f19353deeeb688a2754f61b5c`.
Private archive SHA
`7bd9fedd200272069f78de7e5560c8f619f3d00985e38b5462cc8b49312d3abe`.
All7files were downloaded with the existing7897proxy and owned OAuth, verified
and extracted. No public SSH model/animation transfer or new proxy was used.

The older `soma_unified_terrain_server.py`loads only its original adaptation
weights. It cannot load this full native model. New
`scripts/soma_native_bank_server.py`strictly loads and hashes ALL model tensors.
The source scene contains only static terrain and four observed frames. Its
hash/source identity is bound to the new checkpoint's capacity source contract.
The legacy scene loader only recognizes older dataset manifests, so the new
runtime uses its own bound scene validation through `configure_terrain_live`.
No old dataset identity is assigned to the new scene.

## Inference

Exactly8tokens,4argmax Pose passes,4history frames,24generated playback frames,
published in8frameblocks at30Hz;0path refinements. Terminal world-XZ velocity
and desired heading are live inputs. Default up entry starts from the same four
real frames as the accepted source-start comparison. Its initial W speed is
3.725997m/s with the original body/travel heading offset. This is a suggested
initial command, not a prerecorded input schedule. Other speeds/directions and
key release are available. No future source pose/height/tokens enter the actor.

Both native eager and separately verified CUDA execution are supported. The
first CUDA probe chained unrelated directions beyond the bounded source terrain;
the independent-seed check retained all five commands. Its detailed diagnosis
found0changed tokens and max FK2.08167e-5m, only0.8167micrometers over the original
2e-5m numerical threshold. This checkpoint uses explicit3e-5m FK and direct
feature comparison tolerances, followed by complete96frameup/down comparisons.
The older executor's default remains2e-5m. No collision/gait quality gate changed.
All failed probes remain in their original directories; no rejected graph was served.

`CommittedUnifiedBackend`uses the already published immutable8frameblock to
compute the next plan during playback. The current plan is never altered; the
prepared plan is committed only after its exact predecessor is acknowledged.
It rejects a different played history. No interpolation, pose smoothing, IK,
Root replacement or collision projection is added.
The eager-only1024matrix device-warmup experiment did not resolve timing and was
removed. The final CUDA path uses the existing active-session graph warmup;
no system proxy, GPU driver or power-management setting was changed.

## Verification

`native_bank_live_eager_check_20261015_v2`and
`native_bank_live_prefetch_check_20261015_v2`passed for both up/down entries.
Each uses96real generated frames with continuous input, release and turns.
Maximum local/A100 FK difference: up0.01198mm/down0.01536mm. Local packet versus
serial feature difference0; five prefetched blocks per entry, including command
changes, also differ0. History and buffers remain unchanged before ACK. Official
rig transport reproduces full skin with max error0. Four focused tests cover
complete weight loading, trained-profile identity, stale-history rejection and
rejection of acceleration that failed active-token verification.

`native_bank_live_cuda_check_20261015_v4`passed both entries. Maximum local/A100
FK differences0.01463/0.02290mm, Root0.01144/0.01717mm; all five command-token
checks per scene match. Median packets12.22/12.39ms, P9523.23/22.96ms. Raw first
call feature roundoff is recorded separately; it is not an exact-bitwise claim.
All17selected contract/prefetch/execution tests and JavaScript syntax checks pass.

Browser real W input climbed the source course; standard keyup submits[0,0].
The diagnostic DOM now updates immediately on input/pause/failure, avoiding stale
last-frame command/paused flags. First compiled browser check:75played frames,
29.2167fps, max54.8ms interval, one buffer underrun. The eager comparisons had
26..29fpsand up to221.5ms intervals. Browser evidence is in the live artifact
directory. Final `native_bank_live_cuda_prefetch_20261015_v1`passed BOTH up/down
entries, including zero-difference prefetched release/turn commands and no
deferrals. The final browser W test played75real generated frames at30.030fps,
mean33.30ms/P9539.50ms/max43.40ms, zero underruns, zero deferred prefetches and
zero commit-wait ticks. All full-body SDF diagnostic layers were enabled during
that timing test. The final keyup records zero input velocity and pause is true.

Actual A/S/D tests also reached generated packets with the correct camera-relative
world directions; each keyup returned to zero velocity. These tests verify input
and execution, not instant velocity reversal or perfect motion quality. Evidence:
`browser-wasd-up-final.json`, `browser-asd-verification.json`, and
`wasd-controller-up-final.png` in the local live artifact directory. The page was
reset to its source-start state and paused. Diagnostic layers were hidden in the
final view for clarity; all sampling remains active and layers can be re-enabled.

Service127.0.0.1:28942uses`--execution verified-cuda`. Initial eager PIDs166900/
180984and intermediate compiled PID164696were verified by cmdline and stopped;
current PID43628is an identity hint only. Inspect cmdline before any operation.
Logs are`native-bank-server-28942-v4.log`and `.err.log` in the run root.
Browser tab9is marked deliverable. Existing28941(PID157044)and28936remain running.

This is a user-authorized trial. Constant-command terrain penetration, some
starts and multi-heading flat gait regressions remain from the research review.
Visual approval and inference correctness do not certify general terrain quality.
