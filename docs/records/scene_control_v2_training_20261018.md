# scene_control_v2 training handoff

Status: canonical native-global366 contract and real-source provider implemented;
38 bounded CPU tests passed; real final-model 8/12/16 CLI forwards completed with
zero Adam. Zero-Adam checkpoint/logit parity proof saved. Historical CPU diagnostic completed
with 4 Adam updates / 8 engineering sample exposures / 0 formal exposures.
Owned files: `motionbricks/scene_control_v2/training.py`,
`curriculum.py`, `scripts/train_scene_control_v2.py`, `tests/test_scene_control_v2_training.py`,
`motionbricks/scene_control_v2/provider.py`, `tests/test_scene_control_v2_provider.py`,
and this report. No package initializer/contracts/config files are owned here.

Plato contract correction: canonical `world` / `history` are native **global366**,
matching real bundle `initial_history=[1,4,366]`. Statistics remain **dual370**;
local motion remains **365**. `prepare_native_pose_inputs` consumes global366
directly. For an actual dual370 source, call
`dual_to_global_features(dual_motion, *, dual_rep)` explicitly before batching.
No automatic format detection or four-column padding is used. The original
four-Adam dual370 engineering artifacts remain historical and unchanged.

Plato integration note: the public implementation now exists. `preflight_training_config(config,
mode="preflight")` is pure configuration admission: no factory imports, model construction,
Adam or CUDA operations. CLI config root schema is
`motionbricks.scene_control_v2.training_config.v1`; keys: `policy_id`, `label_lineage`,
`curriculum`, `budget`. A unified configuration can place this complete dict in its
training section and call this function with that section.

`curriculum={"horizon_probabilities":[p8,p12,p16],"phases":[{"start_exposure":0,
"source_root_probability":p,"source_history_probability":p,
"token_probabilities":[all_mask,random_mask,self_iteration]}]}`.
All probabilities are explicit; no training distribution is chosen implicitly.
`budget={"kind":"diagnostic"|"formal","max_batches":int,"batch_size":int,
"max_exposures":max_batches*batch_size,"max_adam_updates":0}` for preflight/forward.
Optimizer mode additionally needs `approval_id`, positive `learning_rate`, positive
`clip_norm`, and `max_adam_updates=max_batches`. Preflight rejects optimizer budgets.

`run_bounded_trial(trainer, provider, *, curriculum, budget, output, mode, seed=2333)`
is the executable interface; `provider(request)` returns a full admitted batch.
CLI `--factory module:function` returns `(SceneControlTrainer, provider)` from the
resolved config; required only in forward/train mode. No stub provider is supplied.

## Integration Contract

- `Curriculum.draw(batch_size, *, exposure, seed)` returns a homogeneous-horizon
  `BatchRequest`: `horizon_tokens` (8/12/16), per-row `root_source`
  (`source`/`predicted`), `history_source` (`source`/`generated`), `token_mode`
  (`all_mask`/`random_mask`/`self_iteration`), and `self_iteration` (1/2/3).
  The data provider must fulfill the request exactly; unavailable recovery data is an error.
- `validate_training_batch(batch, *, lineage, policy_id)` is the importable admission
  function. `SceneControlTrainer.forward(batch, *, request, seed)` runs real model
  forwards and returns `loss`, `pose_ce`, logits, input tokens and mode accounting.
- Model protocol: `forward_pose(pose_tokens, batch) -> {pose_logits: [B,N,H,K]}`.
  The complete admitted batch is passed, including contact and scene fields.
  `NativePoseAdapter(backbone, volume_encoder, volume_width)` reuses the actual
  native Pose backbone, with explicit height+volume scene memory (no hooks).
  Volume encoder protocol: `forward(volume) -> (memory[B,Q,D], valid[B,Q])`.

Batch schema `motionbricks.scene_control_v2.training_batch.v1`:

| Field | Contract |
| --- | --- |
| `world` | float `[B,4*N+1,366]`, native raw global target, four-frame prefix is the actual history |
| `history` | float `[B,4,366]`, native raw global history in the same centered coordinate frame as world |
| `history_source`, `root_source` | per-row string lists, exact curriculum request |
| `root_condition` | dict: `values[B,4*N,R]` normalized native Pose Root features; `height[B,4*N,1]` physical height relative to last history frame; `target_id` per-row matching target IDs |
| `observation`, `volume` | explicit scene dictionaries, each carries scalar `policy_id` and `horizon_tokens`; native height observation contains `points`/`valid`; volume contents follow supplied encoder |
| `masks` | `frames: bool[B,4*N]` actual frame validity; `motion_loss_frames` same shape for motion objectives; independent `tokens: bool[B,N,H]` for code CE, including latent0 by default |
| `contact_targets`, `contact_valid` | matching `[B,4*N,C]`; valid bool; unknown contacts remain masked |
| IDs | per-row nonempty string lists: `source_id`, `family_id`, `scene_id`, `policy_id`, `target_id`, `history_id` |
| `recovery` | `None` for source-only history; generated rows require dict with accepted bool[B], matching history/target IDs and a real teacher ID |
| `pose_tokens` | long `[B,N,H]`, per-head labels, no packed int |
| `label_lineage` | exact equality to trainer's final frozen codec lineage dictionary |

Additional required bindings implemented after first contract publication:
`data_kind="engineering"|"formal"`; `root_condition.source` matches `root_source`,
`root_condition.history_id` matches the batch, `root_condition.condition_id` supplies
per-row root prediction/source identities; both `observation` and `volume` carry
matching `root_condition_id` and `history_id` lists. Thus stale source-path observations
cannot be reused for a predicted Root branch. Formal optimizer mode rejects engineering data.

Training inputs are native **global366**, not the complete dual370 representation.
Statistics contain both Root representations and remain **dual370**; normalized
local history is **365**. `prepare_native_pose_inputs` calls `global_to_local`
directly on global366 and never indexes it using dual370 indices. The separate
`dual_to_global_features(dual_motion, *, dual_rep)` extracts global Root + body
from an actual dual370 tensor; callers must explicitly request this conversion.

`label_lineage`: `schema=motionbricks.scene_control_v2.final_codec_labels.v1`,
`codec_sha256`, `vocabulary_id`, `codebook_sha256`, `encoder_sha256`, `scene_encoder_sha256`,
`normalization_sha256`, `label_manifest_sha256` (64 lowercase hex each),
`heads`, `entries_per_head`, `frames_per_token=4`, `frozen=true`, `final=true`.
`data_kind="engineering"|"formal"` is required for label lineage as well as the
batch. Formal optimizer execution rejects engineering label identities before
constructing Adam, even if the provider relabels the batch itself as formal.
This explicit training-side dictionary can be built from the codec worker's final-label receipt.
No implicit old-cache compatibility or temporary codec labels.

### Codec Worker Binding (Actual)

`make_training_label_lineage(identity, *, data_kind, codec_sha256, scene_encoder_sha256,
normalization_sha256, label_manifest_sha256)` consumes Wegener's
`quantizer.FrozenCodebookIdentity`; it copies the exact `identity_id`, codebook,
encoder hashes, representation ID, heads and entries-per-head. This separates
the frozen token vocabulary from full checkpoint/decoder and observation lineage.
`FrozenCodebookIdentity.normalization_sha256` is mandatory. Both
`make_training_label_lineage` and `attach_final_codec_labels` explicitly require
the supplied normalization digest to equal this frozen identity field. The real
provider computes it from `tensor_state_sha256({"mean": actual_mean,
"std": actual_std})`; representation names are not a substitute for statistics.

`attach_final_codec_labels(batch, cache, identity, *, lineage)` delegates cache
reading to `quantizer.read_token_cache`, checks the receipt against the frozen
identity and returns a new dictionary. No encoding, cache mutation, packed
indices or silent relabeling occurs. `representation_id` is required for this bridge.
These two functions implement the final-label identity mapping, not a parallel codec API.

`select_root_condition(source, predicted, *, root_source)` performs the actual
per-row tensor/identity selection before observation sampling. Both explicit
candidates supply `values`, `height`, `target_id`, `history_id`, `condition_id`.
The selected branch never substitutes source values for a missing prediction.

### Native Model Construction (Actual)

`build_native_pose_adapter(*, pose_args, dual_stats_mean, dual_stats_std,
volume_encoder, volume_width, backbone_state)` constructs SOMASkeleton30,
the native dual/global representation and the real `pose_backbone_network`.
`backbone_state=None` explicitly requests fresh weights; otherwise state loading
is strict. `pose_args` must support horizons 8..16 and terrain conditioning.
The adapter's height+volume memory is consumed by native cross-attention;
`prepare_native_pose_inputs(backbone,batch)` supplies only the four history poses,
selected Root values/height, and actual horizon. Future world values are not
passed as pose conditions. The supplied backbone is independently constructed;
do not pass a live service's shared module.

The trainer's present objective is focused valid-code CE. It passes the full
batch through the model protocol, retaining physical/contact targets, but does
not claim codec geometry loss or decoded-Pose physical loss. Wegener's
`codec_loss_terms` remains the concrete codec objective interface; its
metric prediction/target preparation belongs to the integrated codec stage.

## Execution Boundary

CLI defaults to preflight. Forward trials require explicit batch/exposure bounds;
optimizer mode additionally requires an explicit approval ID, diagnostic or formal
budget class, Adam-update cap, learning rate and clipping. Outputs use a fresh
directory, never resume/overwrite. Diagnostic Adam updates and exposures are not
formal exposures. The initially zero-update scope was extended only by the
explicit four-Adam CPU engineering authorization documented below.

## Integration Remaining

- The real source-history/source-Root provider below now supplies global366 and
  actual final-codec labels. Accepted generated-history recovery targets,
  predicted-Root matches and new body/time actor observations remain unavailable;
  requests for those branches are errors. No synthetic body-region fixture is
  converted to invented SOMA motion. Missing fields raise their exact keys.
- CLI accepts the training-config dictionary above. Plato's unified run config
  should bind/pass this complete section, not old 38.4M training budgets.
- Deployment backend must adopt `curriculum.ITERATION_POLICY` and
  `remask_argmax` for this exact four-pass token distribution. Training does not
  silently assert compatibility with a different runtime sampler.
- GPU and formal training budgets await the parent's explicit allocation. The
  four-Adam CPU engineering diagnostic below is complete and will not resume.
  No CUDA, old codec pilots, old formal training, services, browser pages, SSH,
  or remote transfers have been run by this worker.

## Actual Verification

Command, project cwd:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_scene_control_v2_training.py -q
```

Latest training + provider unit result: **38 passed, 9 upstream PyTorch warnings,
9.33 seconds**. Command adds `tests/test_scene_control_v2_provider.py` to the
training-only invocation above, with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
Warnings concern deprecated TorchScript and prototype nested tensors.

- Actual `pose_backbone_network` + SOMASkeleton30 + native feature normalization
  and explicit height/volume encoder: 8/12/16-token forwards, all three input
  modes, finite logits and CE. This is a fresh width-32, one-layer, 2-head x
  4-entry **engineering fixture**, not pretrained full-capacity quality evidence.
- The selected Root and volume each change actual native logits; changing
  future `world` values does not alter conditioning/logits. Known pose mask is
  exactly four frames. Self-iteration input stays identical when source token
  labels change, confirming it is the model's own iterative state.
- One CPU loss backward confirms native output/volume-projection gradients;
  every parameter remains bit-identical. No optimizer was constructed or run.
- Rejected mismatched generated history/source prefix without modifying world,
  unaccepted or mismatched recovery targets, stale history/path observation,
  wrong observation version/horizon, missing contact validity, wrong final
  codec lineage, and missing predicted Root data. Unknown contact NaNs remain
  masked; valid nonfinite targets fail admission.
- Codec worker's real frozen identity/cache utilities are exercised by the
  training receipt bridge. Per-head indices stay `[B,N,H]`.
- Default CLI preflight does not import even a nonexistent factory; bounded
  actual forward trials never construct Adam (enforced by the test), refuse an
  existing output directory, log every batch, and record failures with exact
  missing fields. Diagnostic/formal exposures and Adam counts remain zero.

The unit suite executes zero Adam steps. The additional formal-admission
regression rejects an engineering label identity before optimizer construction
and before creating an output directory. Full pretrained checkpoint forward,
GPU throughput, real recovery teacher data, full-skin/route acceptance and
formal training remain parent integration and budgeted execution work.

## Authorized CPU Adam Diagnostic

Authorization: current user's agreement to full-subagent execution, followed by
an explicit CPU engineering allocation: **4 Adam updates, batch 2, 8 sample
exposures, lr 5e-5, clip 0.5, seed 2333, formal exposures 0**.

`approval_id`:
`current-user-consent:full-subagent-execution;cpu-engineering-4adam-8examples;20261018`.

Output directory was confirmed absent before execution:
`out/scene_control_v2_20261018/training_diag_cpu_v1`.
This historical run used the then-declared **dual370 engineering fixture**, not
native-global366 training data. Its five files are retained without modification.
The exact tested native engineering adapter was used: width 32, one Transformer
layer, two token heads with four entries each, real SOMA/native forward and
backward. Labels are simulated, explicitly `data_kind="engineering"` with
`identity_semantics="simulated_engineering_fixture_not_codec_reconstruction"`.
The token embedding and engineering volume encoder were frozen for the
unchanged-component check. All execution stayed on CPU; CUDA was hidden from
the process. The quality gate remains false; no full-capacity model or real
motion dataset was evaluated.

Actual `run_bounded_trial(..., mode="train")` result:

- `status="complete"`; actual/diagnostic Adam **4/4**; diagnostic exposures **8**;
  formal Adam/exposures **0/0**. Optimizer checkpoint state records step **4**.
- Trainable elements: **155566**. **37** trainable tensors changed. All **5**
  frozen parameter/buffer tensors remained bit-identical, including the frozen
  token embedding and both volume-encoder parameters.
- Initial full state SHA256:
  `b8dfdace0a1c1073d0f3cce90af014237c72341ed4011e5f13de28ec55853466`.
  Trained full state SHA256:
  `3015d604493c80d6ec29cccdc1d866c44dd6131319509d461c34fc7f0961de1f`.
- Frozen-subset SHA256 before and after is exactly
  `512071a8c70073155907052b93d8e1c757ee1d716b2bb1028349e06371a14206`.
- Checkpoint loaded with `weights_only=True`, then strict model loading into a
  fresh adapter. Reloaded forward is bit-identical to the trained model:
  logits `[2,12,2,4]`, finite CE **1.3137059211730957**. This verification reused
  the last training batch and performed no additional Adam or sample exposures.
- Actual sampled training horizons: 8 tokens / 2 exposures; 12 tokens / 6
  exposures. Token modes: all-MASK 3, random-mask 1, self-iteration 4. Root and
  history branches each split source 3 / predicted-or-generated 5.

Artifacts in the output directory:

- `receipt.json`: exact budget, approval, counts, identities and parameter scope.
- `metrics.jsonl`: four completed optimizer batches.
- `checkpoint.pt`: trained model, Adam state and completed receipt.
- `initial_state.pt`: pre-update tensor state for independent comparison.
- `execution_verification.json`: changed/frozen tensor names, hashes, strict
  reload and bit-exact forward results; `budget_auto_resume=false`.

The manual, non-pytest helper is
`tests/test_scene_control_v2_training.py:execute_authorized_cpu_diagnostic`.
It enforces this exact 4/2/8 budget and refuses an existing output directory.
No additional optimizer execution or automatic resume is authorized here.

## Global366 Correction And Zero-Adam Proof

The former batch contract confused raw global motion with the complete dual
representation. Canonical `world[B,T+1,366]` and `history[B,4,366]` now match the
real native bundle. `prepare_native_pose_inputs` passes global366 directly to
`dual_rep.global_to_local`; it no longer calls `get_feature_subset` on that input.
`dual_to_global_features(dual_motion, *, dual_rep)` is the explicit raw dual370
conversion API. It selects the native global Root and body indices and removes
the four local-Root fields. No four-column padding is created. The canonical
training validator rejects unconverted dual370 and mixed-width history.

New zero-Adam regression coverage:

- Canonical366 admission; dual statistics370 and local365 stay unchanged.
- Sentinel-valued full dual370 input confirms actual Root/body selection, source
  immutability, and independence from the removed local-Root fields.
- Global366 history preparation succeeds while `get_feature_subset` is replaced
  with a test function that raises, proving global input is not reindexed as dual.
- Four native Pose sampling passes are bit-identical between explicit dual370
  extraction and direct global366 input at every horizon 8/12/16.

Fresh proof artifact:
`out/scene_control_v2_20261018/training_global366_zero_adam_v1/global366_parity.json`.
It was produced by the bounded CPU/no-grad helper
`verify_saved_global366_parity(checkpoint, output, *, bundle_path)` in the owned
training test module, reading the original diagnostic checkpoint, never calling
an optimizer or resuming training.

Actual proof results:

- Real bundle read with CPU mmap: `initial_history=[1,4,366]`, statistics `[370]`;
  global-to-local with its actual frozen statistics returns finite `[1,4,365]`.
  Only history/normalization tensors were used; the full bundle model was not run.
- Historical diagnostic tensor `[2,49,370]` explicitly converts to `[2,49,366]`,
  history `[2,4,366]`. The original checkpoint strict-loads into the same native
  small model. All four sampling-pass logits and replayed final logits are
  bit-identical; replayed CE is **1.3137059211730957**, matching the original run.
- Model-state SHA256 before and after remains
  `3015d604493c80d6ec29cccdc1d866c44dd6131319509d461c34fc7f0961de1f`.
- SHA256 checks cover all five historical diagnostic files before and after;
  every file is unchanged. The historical ledger remains 4 Adam / 8 engineering
  exposures / 0 formal exposures. This correction adds **0 Adam / 0 training
  exposures / 0 formal exposures**, with `quality_accepted=false`.

Plato was directly notified of the corrected API and the intermediate bad-shape
test-block fix. He independently reported the cross-module CPU suite at
**144 passed, 9 warnings**; the training worker's final owned suite is the
**30 passed** result above. The zero-Adam proof is ready for parent review.

## Noncausal Codec Mask Correction

Native codec convolutions are noncausal: latent0 can encode future frames and
affect their decoded motion. Codec-worker sensitivity/VJP evidence is in
`docs/scene_control_v2_codec_20261018.md`, Encoder Token-0 Dependency / Zero-Adam.
`masks.frames` now means actual frame validity; required `motion_loss_frames`
is an independent motion-objective mask. Real windows exclude history frames
there but set every valid code CE mask true, **including latent0**. Token CE
validity is no longer inferred from a four-frame motion block. The historical
diagnostic replay explicitly retains its old token0=false policy, preserving
its original CE, checkpoint and budget. Zero-Adam regression verifies full-code
CE, independent motion-frame masks and explicit legacy exclusion.

## Real Engineering Provider (Completed)

C directly confirmed the unique original train record
`generated_f733c63c38e13f58`. Explicit inputs:

- `out/parc_curated_20261009/reference_soma_v3/three_stair_flights/retarget.npz`.
- `out/parc_training_20261009/native_pilot_v1/terrain_geometry/generated_f733c63c38e13f58-30c4d74c84_paired_heightfield.npz`.
- `out/parc_training_20261009/native_pilot_v1/accepted.json`.
- `out/parc_training_20261009/native_pilot_v1/upstream_manifests/generated_f733c63c38e13f58-30c4d74c84_upstream.json`.

The raw NPZ has 646 frames at 30 Hz, world roots, local/global rotations and
SOMA target joints. Names/parents map exactly to native SOMA30. Native feature
encoding/restoration preserves world pose, heading, metres and source frame
IDs; no old canonical native shard is used. Slices `[1:34]`, `[1:50]`, `[1:66]`
retain each real T+1 endpoint. Time is explicitly derived frame_index/30, not
captured timestamps. Features/labels stay in memory; no old cache or retarget
is rebuilt and the original train split is unchanged.

Window translation applies to root XYZ, RIC Y, target joints, scene height
values and scene origin together. Heading is retained without an additional
yaw. Bound scene scale is already applied and is not multiplied again.
Strict native numeric gates: root absolute **1e-5 m**, rotation-FK **3e-5 m**.
Actual full-source errors: root **3.814697265625e-6 m**,
FK **5.80097457714146e-6 m**, comparing the same SOMA pose through the native
feature round-trip.

Actual policy:
`motionbricks.scene_control_v2.legacy_heightfield_source_root.engineering.v1`.
Volume uses CPU `unified_scene_field.field_from_heightfield(scene)` native
prism SDF/normals. 2D patch config is read from the actual bundle Root
`observation_range_policy` via `terrain_observation_config`, including 0.4m
forward spacing and intent-horizon semantics. Source path endpoint displacement
sets intent extent. Corridor centres/right axes use the native observer's pure
path-tangent `_curve` math, including sideways/backward motion, without hooks.
This is offline source-Root teaching, not the new body/time actor sampling.

Native four-channel foot features use the existing terrain-relative height/
velocity heuristic, marked derived. V2 full-body targets are NaN with all
contact/intended/actual/anchor validity false; no anchors or recovery teacher
are invented (`recovery=None`). Upstream `training_approved=false`, missing
instruction/license evidence and the original source family remain recorded.

### Model And Labels

Actual final checkpoint `out/pfnn_mixed_20261015/final_acceptance_flat_v1/weights.pt`
strict-loads the native 16-layer width1024 Pose, codec and shared scene encoder.
All modules are frozen/eval on CPU. Adapter volume projection explicitly copies
final `adaptation.pose.output.weight`, with zero bias. This is a declared
engineering assembly using final parameters, not the old controller operator.

Real final encoder/native frozen VQ produces int64 `[1,N,8]` labels in 0..9,
without packed-index conversion. Manifest binds source/scene/upstream files,
window/frame translations and actual global/local/token tensor hashes.
Actual frozen mean/std digest:
`407eff483f40d7beffa3218853e03f2a8e975ec1d97bfcbd395206d6cf736250`.
Encoder digest:
`1fa4c6855607567f5983a07150467431604d40d0322438ca319d6418d37f65f4`.
Both training lineage/cache bridges explicitly compare normalization against
`FrozenCodebookIdentity.normalization_sha256`; wrong-statistics negative cases
pass in the zero-Adam suite. The complete manifest is fixed before hashing,
and its saved hash is checked against the lineage after execution.

### Public API For Plato

`provider.make_real_engineering_factory(config)` returns the actual
`(SceneControlTrainer, RealEngineeringProvider)` for existing CLI `--factory`.
`provider.prepare_real_engineering_config(provider_spec, *, budget, curriculum)`
resolves real frozen-codec labels/lineage and returns a complete training config;
it does not execute Pose forward or construct an optimizer.

`provider_spec` fields:

| Field | Meaning |
| --- | --- |
| `schema` | `motionbricks.scene_control_v2.real_engineering_provider.v1` |
| `cpu_threads` | Exactly 2 |
| `artifacts` | `architecture_bundle`, `checkpoint`, `bundle_sha256`, `checkpoint_sha256` |
| `accepted_path`, `source_id` | Explicit record table and unique existing train record |
| `motion_path`, `scene_path`, `upstream_path` | Explicit files, checked against accepted SHA256 bindings |
| `window_starts` | Explicit lists, e.g. `{"8":[1],"12":[1],"16":[1]}` |

Relative paths resolve against `Path(provider.__file__).parents[2]`, the actual
project root, independent of cwd. No PRO directory is guessed. Top-level config
keeps the training schema, policy, actual lineage, curriculum and budget, adding
`provider: provider_spec`. Only batch1/sourceRoot/sourceHistory is implemented;
predicted Root or generated-history requests raise precise errors.
The resolved budget covers source-only warmup through exposure2. Its declared
future phase at exposure3 is unreachable within this trial and is not replaced
with fabricated matching data.

### Actual CLI Proof

Fresh output: `out/scene_control_v2_20261018/real_provider_cpu_v1`.
`resolved_config.json` contains actual lineage and relative paths, ready for
Plato to map to `configs/scene_control_v2_real_provider_cpu.json` without another
model construction or trial. Actual existing CLI factory execution, CPU2threads,
seed2338, all-MASK/sourceRoot/sourceHistory:

| Tokens | Global window | Labels | CE Including Latent0 |
| --- | --- | --- | ---: |
| 12 | `[1,49,366]` | `[1,12,8]` | 1.6907930374145508 |
| 16 | `[1,65,366]` | `[1,16,8]` | 1.6707731485366821 |
| 8 | `[1,33,366]` | `[1,8,8]` | 1.8909828662872314 |

**0 Adam / 0 diagnostic training exposures / 0 formal exposures**.
The manual helper invokes the existing CLI with a capture-only wrapper around
the real factory and a trap rejecting any Adam construction. Model parameters/
buffers and full codec/EMA state hashes are exactly unchanged before/after.
The original four-Adam files remain untouched. No forward checkpoint is saved.

Small artifacts: `provider_checks.json`, `resolved_config.json`,
`forward/receipt.json`, `forward/metrics.jsonl`, `forward/resolved_config.json`.
Executable CLI form (requires a fresh output directory):

```powershell
python scripts/train_scene_control_v2.py --config out/scene_control_v2_20261018/real_provider_cpu_v1/resolved_config.json --output <fresh-directory> --mode forward --factory motionbricks.scene_control_v2.provider:make_real_engineering_factory --seed 2338
```

Provider + training regressions: **38 passed, 9 upstream warnings, 9.33s**.
Coverage includes real source/scene/frame alignment, strict numeric gates,
non-cwd paths, missing-file failure, side/backward tangent orientation,
normalization identity and latent0 mask independence. Real forwards establish
provider/model execution and frozen state behavior. Long-window generation,
full-body contacts, complete routes and final quality remain unmeasured/false.
