# MB + SONIC E0 Asset Audit

## Audit Boundary

Captured **2026-10-18 15:07:40 UTC / 23:07:40 Asia/Singapore**. Workspace:
`${UPSTREAM_ROOT}`. All paths below are relative to that root.
Evidence addendum updated **2026-10-18 18:06:53 UTC / 2026-10-19 02:06:53 Asia/Singapore**.
Machine-readable evidence:
`motionbricks/out/mb_sonic_e0_e3_20261018/asset_audit.json`.
Its SHA256 is
`5cb7bcf02cc01a0c8a9111e9c2bc4edf72f971514099deb6cbf7950500ab17d1`.

Overall status: **BLOCKED**. Formal-layout exports are under
`motionbricks/out/mb_sonic_e0_e3_20261018/task_artifacts/`, with E0 audit,
version lock, model interface, purpose-specific license CSV, backend decision,
RUNBOOK and E0-E7 STATUS. No weights or datasets are copied.
Final STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md` and
E0-E3 index：`../out/mb_sonic_e0_e3_20261018/task_artifacts/artifacts/index.json`:
E1v4 foreground30Hz/full-chainP9915.597689ms PASS; E3v3 FAIL future G1 ankle
reference and full-chainP9988.61218ms. Final ledger0.11294198766661187GPU-hours,
0optimizer/0new exposures. Earlier run addenda below are historical snapshots.

Purpose is the already authorized, internal **E0-E3 frozen integration**. This
audit made **0 optimizer updates**, started no formal training, GPU job, server,
browser, or physics rollout, downloaded no dataset, and accepted no terms. CPU
ONNX metadata was inspected without inference; existing checkpoints were opened
with CPU mmap for metadata/stats, without constructing or forwarding the actor.
Existing MotionBricks sources/assets and v3 were not edited.

Formal START_HERE and the complete plan are now supplied and read:
`${USER_HOME}/Desktop/G1_Agent_START_HERE.md` SHA256
`d8c369018cd08e5931124e22146669da214858b1b9b2934b714b699481d356f3`,
and `${USER_HOME}/Desktop/G1_AllTerrain_Plan_v1_20261018.md` SHA256
`3ef1fc866f0cc19b6829194c07a46db21b95087a02259e60e081e819350564dd`.
Plan identity status is **PASS**. The schema/tool ZIP remains **BLOCKED** and its
proposed20-test suite is **NOT_RUN**. The final parent CPU suite is **257passed,
7CUDA opt-in skips,44TorchScript warnings,21.21s**, bound by
`final_tests_v1.xml`; earlier109/111counts are historical, not summed,
with all **16 scene verification reports** available under the qualified ray
contract below. The worker's pacing/yaw/diagnostic/Reset/G1-ghost/Follow source
passes **25 Python protocol/static tests**, including **13 Node protocol tests**,
plus JavaScript syntax.
These are implementation checks, not the missing package suite.

## Identity And Version Locks

The parent inventory `motionbricks/out/mb_sonic_e0_e3_20261018/assets.json`
has SHA256 `fbbb1295f771d23653bda600f88b5f30ce69c2dcc37f360b4c30cc1a609a1ca8`.
**All 20 recorded file sizes and SHA256 hashes were independently recomputed and
matched.** The JSON retains every expected/actual path, size, hash and result.

| Component | Locked identity |
| --- | --- |
| Repository HEAD | `ab1add20491c8093c696be27fb2e8eeb31001016` |
| SONIC Hugging Face revision | `6733128a3d8a523b1418b06bca3cdf61c8b0987f` |
| SONIC selection | Default release files; not low-latency or v1.1 |
| Vendored Kimodo HEAD | `58e781898b3d7e328a676a75d3e338c45dce3ad9` |
| Vendored PARC HEAD | `846ed98bd526c1dd93ddc1701b601036d553cd2f` |
| Interpreter | Python 3.10.20, Anaconda, Windows AMD64 |
| Physics / inference | MuJoCo 3.8.0; ONNX Runtime 1.23.2, CPU provider |
| Other inventory versions | torch 2.11.0+cu128; NumPy 2.2.6; SciPy 1.15.3; aiohttp 3.13.5 |

The checkout is dirty. HEAD records the baseline commit, while the content hashes
bind the inspected files. A Git commit alone does not identify the local edits.
Package versions above are the parent's existing environment lock; this audit
did not reinstall or upgrade dependencies.

Subsequent parent-authorized additions are joblib1.6.0 and cloudpickle3.1.2,
installed binary-only with `--no-deps`; actual imports were confirmed by the
parent and distribution versions independently read by this audit. Parent-supplied
PyPI wheel hashes are joblib
`3dbbf9f6e4b592a2357b854608e980fe6390d131d7a82f011a377ef2ebef7aba`
and cloudpickle
`9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a`.
MuJoCo3.8.0, torch2.11.0+cu128 and ORT1.23.2 remain unchanged. The original
20-entry assets inventory is preserved; additions are recorded separately.

### Principal Model And Scene Hashes

| Asset | SHA256 |
| --- | --- |
| Default SONIC `model_encoder.onnx` | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |
| Default SONIC `model_decoder.onnx` | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| Pinned SONIC `config.json` | `326dcc509ad93509cf262871256a74f508f43a8430f61ec6abe8034e985ce948` |
| Pinned `observation_config.yaml` | `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b` |
| Pinned SONIC `LICENSE` | `24ab66be50d1aca4fc5e029ef76ce4ceaac6557ea21665caf4b140695a76ffee` |
| Trained compact MB `final_acceptance_flat_v1/weights.pt` | `0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646` |
| Native SOMA bundle `a100_live_20261010/a100-terrain.pt` | `0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d` |
| PFNN `pfnn_map_catalog_v1/catalog.json` | `e117d11dd10547b4ec5c2b26be32f5758eb57a6575405aa57efda12fad222eca` |
| Original G1 `g1_29dof_with_hand.xml` | `c2d4835079c54ac8bfdfa6010ff802095e6d4bf5d9e88652e0540eff0ed122b5` |
| Original `scene_43dof.xml` wrapper | `855f19458e27a199859955cd878e518940fab9332912df22dd79a2683dc496c1` |

## Actual Model Contracts

ONNX Runtime CPU session metadata directly reports:

| Model | Input | Output | Type |
| --- | --- | --- | --- |
| Encoder | `obs_dict [1,1762]` | `encoded_tokens [1,64]` | Float32 |
| Decoder | `obs_dict [1,994]` | `action [1,29]` | Float32 |

The pinned model config specifies Unitree G1, **50 Hz**, token dimension **64**.
The G1 observation names and runtime ordering use ten real history observations.
The YAML's opening comment still says total dimension **436**; its named enabled
fields and actual ONNX input are **994**. No 436-dimensional replacement was
selected. The G1 reference path uses ten samples at native offsets
`0,5,...,45` / 50 Hz, covering **0.9 seconds**; the config's default SMPL horizon
is a distinct mode and is not substituted for the G1 contract.

Functional signal status: **PASS** for the parent CPU diagnostic
`sonic_reference_translation_sensitivity.json`. Translating the real first E1
reference Root/body positions by `[1,-0.5,1]`m, with actual warmup proprioception
unchanged, gives bitwise-identical encoder observation, tokens and motor action;
saved encoder-input max difference is0. The pinned G1 mode consumes joint
positions/velocities and relative Root orientation, not reference world Root/
body translation, including its height shift. World Root XYZ error is a
diagnostic, not proof that this mode directly tracks a supplied XYZ trajectory.
No physical rollout was modified;0 optimizer,0 GPU. New velocity/environment
conditions remain **NOT_RUN**.

The old native bundle has schema `motionbricks.mixed_online_soma_bundle.v1`,
**30 Hz**, SOMA30 bone names, four observed history frames `[1,4,366]`, and neutral
joints `[30,3]`. Both stored normalization tensors have **370** entries. The
native representation owns this feature indexing; no new stats were estimated.
The original named-tensor digest was recomputed with the existing name/dtype/
shape/bytes algorithm and matched
`42711c188447eee551d778d696241511d76297debab2d7f4af8ef71880573604`.
Both tensors are finite. Seed provenance binds shard
`official_53df774126c9f2ce-70115b3272_native30.npz`, SHA256
`2461dd064200bd9eab2072947b9da8fe0326111605fbadce70f280e4bcd0a396`,
start 0, and exactly four observed frames. Future clip data is not asserted.

Official skin path:
`motionbricks/out/vendor/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz`.
Hash `90ee2cdf50f168382a7dd7e9c88b118298aba0aca4b72022080c89c4bab0ceb2`;
**18056 vertices**, **36108 triangle faces**, **77 bind rig matrices**, and all
**8 native influence slots** per vertex. Stored weight sums span
0.9999998807907104 to 1.0000001192092896. The JSON additionally hashes the SOMA30/
SOMA77 joint, bind/rest and shape companions and the actual official skin source.
No rig truncation, weight renormalization or skeleton replacement was made.

## Trained MB Lineage And Existing Quality

Source of truth is the actual compact weights and
`motionbricks/out/pfnn_mixed_20261015/final_acceptance_flat_v1/acceptance.json`.
The compact checkpoint has **4000 historical Pose updates / 12800000 exposures**;
its contract records source quotas PFNN/PARC/BONES **0.50/0.25/0.25**, frozen
parent/code/protected hashes, cache receipts, and no postprocessing.

| Historical stage | Checkpoint SHA256 | Updates | Exposures |
| --- | --- | ---: | ---: |
| Root | `d3cdac4d0c260361f275a5ee8f940298486ecf201f00e2051494e93d7dce63e8` | 50000 | 12800000 |
| Codec | `7f8ce3b2498168f9bfa2490bbec52a5f30bce5d0f8b271d8e7f76120aabd61fa` | 20125 | 12800000 |
| Pose source | `b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19` | 4000 | 12800000 |

These historical counts are not E0-E3 optimizer work. The acceptance receipt's
model-state hash is
`a6366a7e8522371be01d632efa6a952839a690a1749e14d6fb65819ff584b50c`;
codec-state hash is
`ee4a08d79a5c5268799352507c9da8d323e5e81a40c36828fc68fe35f042f109`.
The file hashes were recomputed; historical stage tensor hashes and protected
passes are retained as recorded lineage rather than rerunning training proofs.

Existing acceptance permits a loopback experimental interaction, while
**`quality_approved=false` and `delivery_approved=false` remain unchanged**.
The old native bundle also records quality false. The historical review
`motionbricks/docs/pfnn_final_candidate_review_20261016.md` reports 52 complete
routes / 104 rollouts, **0 complete-route passes**, worst penetration **54.05 cm**
against 2 cm, and flat-arena penetration **3.05 cm**. Those failures remain part
of the baseline. Historical browser/protocol usability did not change these
quality flags.

## Complete Local Dependency Coverage

- **20 parent assets:** root LICENSE and legal notice; two original G1 XML files;
  trained compact weights and native bundle; PFNN catalog; five pinned SONIC
  files; six official walking CSVs plus `info.txt` and `metadata.txt`.
- **50 G1 mesh files, 30938500 bytes total:** every `<asset><mesh file=...>` from
  the original robot XML, resolved through its literal `meshdir="meshes"`.
  Names, resolved paths, sizes and SHA256 are individually retained in JSON.
  The robot XML has no external include; the scene wrapper includes this robot.
  The wrapper's hand-instability warning is preserved. No additional robot
  assets were downloaded or changed.
- **16 scene manifests:** the 15 existing PFNN IDs
  `000,001,002,003,004,005,006,007,009,010,011,012,013,014,urban`, plus
  `rigid_boxes`. Each manifest, copied source NPZ, hfield binary, MJCF include and
  original source NPZ is hashed. All declared dependency and original-source
  hashes match. Manifests preserve `(X,Y,Z)->(X,-Z,Y)`, full original float32
  heightfield resolution, no resampling/image quantization, declared rigid boxes
  and fixed scene friction. Renderer-only simplification would not authorize
  sensor/query/physics simplification.
- **Source ancestry:** the existing PFNN source receipt names archive
  `pfnn_source.zip`, 440618298 bytes, SHA256
  `e04c2ffe900e8904458cc3a6fbeebb54a51cd1af891940d5d37c037f31fd24d7`,
  and source inventory SHA256
  `52d62dc277de8f5dc563f68f6f871ce071ec81d321faab53d82ea79fbcc7db1f`.
  The receipt and inventory are hashed; this audit did not re-extract the archive.
- **Official walking reference:** `walking_quip_360_R_002__A428` has 455 samples,
  29 joints and 14 body records. At the released 50 Hz contract its real sample
  interval is 0 through 9.08 seconds. It is an existing CSV reference, not an MB
  prediction. Its adjacent metadata does not identify an originating data grant.
- **Four existing paired PKLs:** pinned SONIC `sample_data/{robot_filtered,
  soma_filtered}/210531/walk_forward_amateur_001__A001{,_M}.pkl` form two
  matching robot/SOMA stems, including an `_M` variant. They are for internal
  E0-E3 calibration only, not four independent recordings or newly authorized
  training data. This audit hashes bytes without deserializing the pickles.

| Paired sample under pinned SONIC `sample_data/` | Bytes | SHA256 |
| --- | ---: | --- |
| `robot_filtered/210531/walk_forward_amateur_001__A001.pkl` | 349988 | `005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b` |
| `robot_filtered/210531/walk_forward_amateur_001__A001_M.pkl` | 352135 | `7d9ec8a24acbb952cfce2048e2d3b5c156e8ae0c43e32443eb5ea42cbb22038e` |
| `soma_filtered/210531/walk_forward_amateur_001__A001.pkl` | 409470 | `c51764c1c0d1b7acfd6768d53f848f892d1340fdcab4b53cbaf281bdc104ff62` |
| `soma_filtered/210531/walk_forward_amateur_001__A001_M.pkl` | 409470 | `7c9f05ce2ead2f1fe6d7a2ecdb69081279a794ded659613104390c039721495d` |

## License And Use Matrix

The local root LICENSE explicitly separates **Apache-2.0 source** from
**NVIDIA Open Model License weights**, release date 2025-10-24. The pinned SONIC
LICENSE repeats this split. The local third-party asset notice names Unitree SDK
BSD-3-Clause, XRoboToolkit/tpm/BeyondMimic MIT and robosuite Apache-2.0. These named
software notices do not identify a specific G1 STL or sample CSV grant.

| Component | Known provenance | Internal E0-E3 purpose | Future training / redistribution |
| --- | --- | --- | --- |
| Repository source | Root LICENSE Part 1; both `legal/` notices; MotionBricks README | Existing local integration code | Separate task authorization; retain applicable license/NOTICE/copyright/modification notices when distributing |
| SONIC weights/config | Pinned LICENSE, revision and actual ONNX hashes | Frozen inference under existing applicable rights; no new acceptance | Separate derivative-model/use review; NVIDIA license copy, attribution, trustworthy-AI and separate-component conditions |
| Trained mixed MB weights | Actual acceptance plus PFNN/PARC/BONES ancestry; README states NVIDIA model license | Fixed experimental candidate, original failures preserved | Not cleared for new training or redistribution by this audit; examine all source-data uses and required attribution |
| PFNN code and data | `source_inventory_v3/official_source/LICENSE.md` | Author's academic/non-commercial scope; organization/purpose eligibility not evidenced here | Commercial grant must be established separately; publication citation requested; short local text does not establish an explicit redistribution grant |
| PARC source code | Both vendored LICENSE files | Existing local software | BSD-3-Clause base and MIT modifications; preserve both sets of notices and no-endorsement condition |
| PARC paired data | Official dataset card currently labels MIT; local release ancestry retained | Existing paired artifacts | Bind the particular release, its notices and any embedded motion's provenance before a new use |
| BONES-SEED | Official custom dataset agreement; historical acceptance/entity receipt absent from inspected inputs | No new raw access or terms acceptance; eligibility remains unresolved | Separate eligibility, purpose and Results/attribution review; raw redistribution and generative commercial-substitute uses are restricted |
| G1 XML/STLs | Exact local XML dependency graph; Unitree description README | Existing local simulation assets hashed | Original asset-specific license and upstream revision remain unknown; SDK license is not substituted for the mesh license |
| Official sample CSVs | Original repository example, six CSVs and two metadata files | Existing E1 frozen reference | Source dataset/sample-specific grant unknown; no training/redistribution clearance recorded |
| Four paired sample PKLs | Existing pinned SONIC `sample_data/` paths and revision | Internal SOMA/G1 calibration only | Model-repository availability and hashes do not identify the originating sample-data grant; new training/redistribution review remains required |
| Kimodo SOMA code/rig/skin | Vendored Apache LICENSE/SPDX; README separates code from model/data | Existing native SOMA30/77 assets | Code notices are known; binary skin/rig asset-specific provenance still requires separate review |

Public provenance checked without downloading data or clicking access/accept:

- [BONES official dataset README](https://huggingface.co/datasets/bones-studio/seed/blob/main/README.md),
  displayed commit prefix `2f59b20`, labels its license `bones-seed-license` and
  links the custom agreement. The inspected local inventory does not preserve
  the entity's prior accepted agreement.
- [BONES official license](https://bones.studio/info/seed-license), read
  2026-10-18, defines academic/startup eligibility, restricted raw-data transfer,
  Results obligations and a commercial-substitute generative-use restriction.
  The applicable historical grant and the intended MB motion-generation purpose
  require an explicit separate decision.
- [PARC official dataset card](https://huggingface.co/datasets/mxucg/PARC), read
  2026-10-18, labels the dataset MIT. The exact downloaded release's local license
  binding remains an open provenance item.

## Existing Physical Evidence And Open Items

The parent already executed real E1 official-CSV physics. This audit reads the
existing summaries and comparison, without rerunning the experiment:

| Existing E1 result | dt=0.005 s, v2 | dt=0.0025 s, halfdt v1 |
| --- | ---: | ---: |
| SONIC controller ticks | 250 | 250 |
| Physics steps including warmup | 1036 | 2072 |
| Policy-period recorded physical states | 1000 | 2000 |
| Chain P99 | 7.86671 ms | 11.988567 ms |
| Recorded real-time factor | 1.964592 | 1.312703 |
| Terminal failure | None | None |
| Maximum absolute free-root actuator force | 0 | 0 |

The comparison records root-position RMS difference **0.007374094 m** and peak
**0.013710471 m**. Browser FPS/display latency are null in both summaries.
The actual B1 MB/G1 physical smokes v1 and v2 are **FAIL**; see the current E3
addendum below. The first actual E1 browser transport
has status **FAIL**, retained in `e1_browser_analysis_v1.json`:112active unique
frames,21.4991284Hz,P95period68.7ms/max69ms,0display drops. Full-record250tick
controller chainP99=9.792443ms, applied external body/generalized forces=0.
The parent later measured E1v3 foreground display **PASS** as recorded below.
No asset/licensing or old-quality flag was promoted from E1 results.

Actual parent v2 display is also **FAIL**:24active frames,3.9099687Hz over
5882.4ms,0drops. Roughly33ms periods alternate with1010-1031ms stalls while
render submission CPU is0.4-0.7ms and physics remains about1x. The v2 receipt,
transport record and mobile screenshot are hash-bound. After the prior tab
closed, the parent opened owned tab2 and ran foreground v3 on new E1 process
140812. V3 display is **PASS**:157active samples/5200.4ms,29.99769248Hz,
sim.005->5.18,0drops. Actual document is visible/focused,50/50G1 meshes visible,
Follow keeps actualRoot centered. Receipt and desktop screenshot are hash-bound.
V3 chain is **FAIL**: sampledP9927.4647ms; complete250tick logP9927.249933ms,
max27.7333ms. E1 is read-only with0command associations. Fresh-handler Reset
click and post-patch mobile checks are NOT_RUN; old190760's Unknownlabmessage
failure caused the explicit parent restart. No UI was operated by this worker.

### Official CSV Kinematic Check

The existing CPU-only `official_reference_fk.json` checks **455 frames x 14
bodies** against the original G1 XML. Position error max is
**2.199535186577931e-6 m**, RMS **7.862066687217264e-7 m**; quaternion
`1-|dot|` max is **8.478865264338609e-7**. Its state assignments are confined to
separate kinematic comparison data, with no actual-rollout state overwrite.
The report hash and all per-body results are retained in the audit JSON.

### E2 Geometry And Grazing-Ray Issue

`scene_verification_v2.json` covers all **16 scenes**. Maximum sampled height
error is **2.3376831563837186e-6 m**, maximum normal-vector error
**7.073438788127345e-5**. Its boundary contract explicitly uses inward incidence,
compares at the actual hit XY and checks exact node data separately.
**158 exact-boundary vertical-ray mismatches remain recorded.** The parent
identifies a MuJoCo grazing-boundary ray-engine issue and reports physical edge
contact verified. A separate edge-contact receipt was not present in this
inspected output directory, so that verification is attributed to the parent.

The original `scene_verification_failure_v1.json` remains evidence: its initial
assumed scene `000` was corrected to actual scene `001`, retaining the original
assumption and **5.427687487294597 m** grazing-ray height error. The qualified
v2 scene flags are not an unqualified all-geometry pass. Both reports and each
retained mismatch are bound in the audit JSON.

### Original-Speed Video And Resources

The existing E1 `e1_official_walk_v2/video_receipt.json` binds a **150-frame,
30 Hz, 5-second, 1x** video from recorded actual MuJoCo states. Rendering uses a
separate `MjData`, not the executor, and performs **no re-simulation**. The JSON
hashes `recording.jsonl`, `physical_walk_original_speed.mp4`,
`physical_walk_frame.png` and the video receipt. The video hash is
`457239bcd52a20fc0649c78adb4067afdca79bef3ee25857538a6bd81588f5d4`.
This is offline physical-state video evidence, with no live browser claim.

| Reported resource entry | GPU charge |
| --- | ---: |
| E1 baseline CPU physics | 0 s |
| E1 half-dt CPU physics | 0 s |
| Official-reference FK CPU check | 0 s |
| This E0 hash/metadata/stat audit | 0 s |
| Parent's bounded MuJoCo evidence renderer | 2.547891499940306 s |
| Parent's native MB two-block diagnostic, conservative allocation charge | 120 s |
| Parent's failed compiled/native FK proof | 23.65733369998634 s |
| Parent's original-precision compiled v2 proof | 31.967484700027853 s |
| Parent's completed hot profile | 50.56892680004239 s |
| Parent's geometry admission | 12.776819699909538 s |
| Parent's full-plan proof | 34.83495340007357 s |
| Parent's E3 B1v1 physical smoke | 35.81933979992755 s |
| Parent's operator v1 failed/unused allocation | 2.2144644998479635 s |
| Parent's operator v2 blocked allocation | 18.981326899956912 s |
| Parent's E3 B1v2 physical smoke | 36.02788330009207 s |

The actual parent `resource_ledger.json` snapshot records **369.3964242998045 charged GPU
seconds / 0.10261011786105681 GPU-hours** against T0's2GPU-hour cap, with0optimizer
updates and0formal-training exposures. The native first run's precise occupancy
is unknown; its120s conservative charge is retained, not replaced by its
0.5652774s two-block generation time. The parent owns subsequent allocations.

New diagnostic statuses are retained separately: native MB performance **FAIL**;
compiled/native proof **FAIL** at FK0.000167431834m versus0.00003m despite equal
tokens; paired retargeting **REHEARSAL_ONLY**; full1.6s genuine MB reference
**FAIL**, rejected at1.24s by left-ankle-roll limits without clipping. The
kinematic adapter exists; actual online MB/G1 smoke acceptance is **FAIL**.
These reports, calibration and proposal identities are added to
the audit; no original weight or quality flag was replaced.

The separate original-precision compiled v2 internal-strategy parity status is **PASS**, preserving
the v1 **FAIL**: model tensors and source hashes unchanged, all compared active
tokens equal, max FK1.2065570444974583e-6m under3e-5m. Its two-block generation
wall time is61.2989ms. Actual MB physics subsequently ran and failed continuity;
whole necessary-chain acceptance remains FAIL.

The28parity stages also retain CPU/CUDA RNG equality. Runtime status remains
**FAIL**: the frozen-source hot profile scores20plans at10Hz after10warmups,
P99103.822504ms, with scene-query~15ms/FK~7ms bottlenecks. Loaded/source-after
hash is `8e4a0ca447febd5dec038ac9d7def1777438e41bca80f29d65232abda73df655`.
B's newer full-plan numeric admission is PASS atmaxFK3.937178917e-6m with
tensor/RNG equality; graph34.4838ms still misses20ms. Original graphs remain.

`retarget_packet_1s_v1_report.json` is **REHEARSAL_ONLY**, using the same uniform
1.0s consumer rule for all commands/maps: starts0 and.18 admit[0,1]/[.18,1.18],
while start.30 and the original full1.6s raw reference remain **FAIL**. The raw
11limit violations are preserved; no clipping/automatic shortening was used.
This report is reference/kinematic scope. Its later1.0s physical smoke and the
new1.12s reserve physical smoke both have status FAIL.

`native_context_consistency_v1.json` adds a CPU correctness **FAIL** for the prior
adapter: raw last4 RIC versus displayed FK mean8.4720487mm/max26.4104307mm; the
original observer had been omitted before the second AR call. The existing
observer's FK/Root preservation is **PASS**, max5.960464e-7m/1.862645e-9m, with
displayed motion unchanged,0optimizer and0GPU. Old compiled proofs retain their
same-internal-strategy scope. The newer full-plan admission uses the original
native observer and `native_played_fk_reencode`; numeric admission is PASS and
actual continuity is FAIL. Quality-causality analysis is NOT_RUN. Native query/
FK/observer profiling and all GPU/T0 allocations remain parent-managed.

### Actual E3 B1 Smokes

V1 is **FAIL**:6policy ticks/24real policy substeps after36warmup substeps,
endsim.30s,initial zero only. MaxFK3.937178917e-6m numerical admission PASS,
unchanged tensors and CPU/CUDA RNG. The1.0s reference leaves100ms after.9s
required future; firstpublish-to-nextready144.5844ms. Second provider81.2116ms
includes28.1122ms history snapshot; its plan was not published.

V2 is **FAIL**:1.12s reserve executed,12ticks/48policy substeps,endsim.42s,
initial zero only. History snapshots28.6254/31.4003ms and providers81.4406/
86.4201ms led to `maximum_plan_wall_age_exceeded` and
`maximum_generation_snapshot_age_exceeded` discards at100ms gates, then trusted
reference exhaustion. A report-serialization error occurred after runtime
closed; analyzed evidence comes from retained records, with no re-simulation.
B's exact CPU history optimization is in progress. Next physical retry,
WASD/turn/body-response program and E3 browser remain NOT_RUN.

- v1 analysis：`../out/mb_sonic_e0_e3_20261018/e3_b1_failure_analysis_v1.json`
- v2 analysis：`../out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v2/analyzed_report.json`
- v2 execution：`../out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v2/execution.json`
- v2 raw log：`../out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v2/episode_0001_000.jsonl`

Outstanding decisions are the schema/tool ZIP/package tests; MB/G1 continuous
physical acceptance and20ms chain timing; PFNN organization-purpose eligibility; BONES historical grant,
entity and generative use; G1 upstream asset license/revision; CSV dataset grant;
binary SOMA asset grant; and release-specific PARC data license binding.
Future training, release, redistribution or a new dataset use needs its own
authorization and source-specific review.
