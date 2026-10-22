# scene_control_v2 codec handoff (2026-10-18)

## Integration contract (implemented; CPU checks passed)

Owned files: `codec.py`, `quantizer.py`, `losses.py` in
`motionbricks/scene_control_v2`, their two codec/loss tests, and this report.
No package initializer, shared contracts, training entrypoint, or old module edits.

Object-ID correction: `contacts.object_id` is a persistent ID, NOT a table index.
`scene.object_ids int64 [B,O]` is REQUIRED alongside transform/validity tracks.
`contact_surface_transforms` explicitly matches IDs per batch and remains invariant
under a joint reorder of object_ids/object_to_world/object_valid. IDs 7/42 are valid;
-1 is absent. Integration must retain persistent IDs in contacts and query metadata,
without silently renumbering `object_id`. Plato was sent this update together with
the exact actual_state/actual_valid and interval-mask field list.

Normalization identity correction: `FrozenCodebookIdentity.normalization_sha256`
is REQUIRED, as exactly 64 lowercase hexadecimal characters. It is part of
`identity_id`, the nested cache vocabulary, and the top-level token-cache receipt.
`SceneControlCodec.freeze_vocabulary(representation_id, *, normalization_sha256,
parent=None)` requires it explicitly. Providers compute it from the actual frozen
tensors using `tensor_state_sha256({"mean": frozen_mean, "std": frozen_std})` and
use the same digest in identity, cache receipt and training label lineage.
Changing only normalization creates a different vocabulary identity, links its
parent when supplied, and causes `read_token_cache` to reject the old cache.

Public imports:

- `quantizer.QuantizerConfig(entries_per_head=10|128|256, num_heads=8,
  code_dim=256, decay=0.99)` and `ExplicitHeadQuantizer(config)`.
  `forward(latents_bdl)` returns `QuantizedLatents` with `latents [B,D,L]`,
  `indices int64 [B,L,H]`, `commitment scalar`, `perplexity [H]`.
  `dequantize(indices)` returns `[B,L,D]`. No combined-vocabulary/packed integer API.
- `codec.SceneControlCodec(native_codec, quantizer, contact_channels)` reuses
  the existing encoder/decoder/feature extraction without calling its old quantizer.
  `forward(motion, *, target_cond, has_target_cond, external_cond)` returns
  `recon_state`, `indices`, `l_commit`, `perplexity`, `contact_logits [B,T,C]`.
  `encode(motion)` and `decode(indices, *, target_cond, has_target_cond,
  external_cond)` keep `[B,L,H]` throughout.
- `codec.SomaGeometryAdapter(motion_rep, skinner, sample_vertex_indices)`:
  `forward(recon_state, *, world_from_window)` returns `joint_positions_m [B,T,30,3]`,
  `body_samples_m [B,T,S,3]`, `local_rot_mats [B,T,30,3,3]`.
  This uses rotation-based FK and selected official LBS vertices, not RIC positions.
- `codec.native_window_transform(init_heading_info, vertical_origin_m)` builds
  the exact fixed `[B,4,4]` transform from native `return_init_heading_info=True`
  metadata (`init_heading_quat [B,4]`, `root_pos_init_xz [B,2]`) and recorded
  vertical origin `[B]`. This helper is for raw `motion_rep(...)` feature construction,
  which canonicalizes its initial heading even with `removing_heading=False`.
  For integration's `NativeWindowFrame` heading-retained `global_to_local` path,
  use an identity rotation and `translation_world_m` in `world_from_window` instead.
  Do not apply the raw-feature initial-heading restore to that second path.
- `losses.codec_loss_terms(prediction, target, *, fps, first_new_frame)` returns
  `LossTerms`. `LossTerms.as_dict()` exposes scalar named terms;
  `weighted_sum(weights)` requires an explicit complete weight mapping.

No duplicate contact enum: data/integration own SUPPORT=0, BRACE=1, GRASP=2,
TOUCH=3, SLIDE=4. SUPPORT/BRACE/GRASP attach fixed material points. SLIDE/TOUCH
permit tangential motion and receive normal anchor error plus assigned-surface gap.
Slip compares local material points only across uninterrupted fixed-contact labels
on the same object and same kind.

All geometry and transforms are metres, seconds, right-handed Y-up world coordinates.
`world_from_window [B,4,4]` is REQUIRED even for identity. It is the fixed frame
restoration including initial XZ, heading, and vertical origin. Geometry is first
decoded/FK/skinned in the native window frame and then restored to paired world.
Targets, executed prefix and scene queries must use that same restored world frame.
This is a coordinate change, not predicted Root substitution or output correction.

Required prediction dictionary for `codec_loss_terms`:

| Key | Shape | Meaning |
| --- | --- | --- |
| recon_state | B,T,F | Normalized native local representation |
| l_commit | scalar | Quantizer commitment |
| joint_positions_m | B,T,J,3 | Differentiable rotation-based FK |
| body_samples_m | B,T,S,3 | Differentiable selected skin points |
| contact_logits | B,T,C | Binary contact logits, not supervision gates |
| contact_signed_distance_m | B,T,C | Differentiable SDF to each assigned contact surface |
| body_signed_distance_m | B,T,S | Differentiable SDF to scene union |

Required target dictionary:

| Key | Shape | Meaning |
| --- | --- | --- |
| recon_state | B,T,F | Source/teacher normalized reconstruction target |
| frame_valid | B,T bool | Actual unpadded frames |
| joint_positions_m | B,T,J,3 | Source/teacher FK in matching frame |
| joint_valid | B,T,J bool | Reliable joint target |
| contacts | nested dict | Batched paired.v1 contacts; see below |
| scene | nested dict | Batched paired.v1 scene; see below |
| contact_surface_valid | B,T,C bool | Assigned-surface SDF query validity |
| contact_interval_valid | B,T-1,C bool | Reliable same fixed-contact EVENT/material anchor interval |
| contact_sample_indices | C int64 | Contact channel to body_samples index |
| body_noncontact_valid | B,T,S bool | Reliable scene query AND verified noncontact body sample |
| executed_joints_m | B,P,J,3 | Actual executed prefix; P >= 2 |
| executed_valid | B,P,J bool | Reliable actual-prefix positions |

`contacts` consumes `actual_state`, `actual_valid`, `anchor_valid`, `object_id`,
`kind` [B,T,C], and `anchor_local_m`, `normal_local` [B,T,C,3]. `actual_valid`
must reflect reliable measured GT/accepted teacher labels, including negatives.
`scene` consumes `object_ids int64 [B,O]`, `object_to_world [B,T,O,4,4]`,
`object_valid [B,T,O]`. Contact object_id is a persistent ID looked up in object_ids,
with -1 indicating absence. The object table may be noncontiguous and reordered.
Intended `state/valid` and predicted logits never gate physical losses. Confidence
thresholding and provenance acceptance happen in the data/integration layer before
setting actual_valid; loss does not silently choose a confidence threshold.
`contact_interval_valid` is explicitly supplied by the event/teacher producer,
false at releases, recontacts and anchor/event switches. The loss additionally
breaks intervals when object/kind changes or a fixed local anchor changes by more
than 1e-6 m; endpoint contact alone never joins two separate support events.

`LossTerms` fields (explicit weight keys): `reconstruction`, `commitment`, `fk_m`,
`contact_bce`, `contact_anchor_m`, `contact_gap_m`, `relative_slip_mps`,
`penetration_m`, `boundary_mps`. Anchor and gap are two logged components of one
contact geometry objective. No extra joint velocity/acceleration/pose regularizers.
All loss weights must be finite and nonnegative, with an explicit entry per term.

Caller queries scene/object SDFs at predicted sample positions and merges outputs
into prediction. It must not substitute detached source distances. Missing labels
are represented by false masks, not fabricated ground. Unused numeric slots may
be NaN; masked slots are removed before loss evaluation.

`first_new_frame` identifies the first unpublished frame inside the decoded
sequence. Boundary velocity uses this frame against the last two actual executed
frames, without replacing any generated root/joint/prefix output.

Frozen vocabulary identity binds explicit quantizer config, codebook/projections,
encoder weights, normalization SHA256 and representation ID. Changing any creates a new identity whose
parent is the previous frozen identity, and requires token relabeling and new Pose
embedding/output training. Decoder-only changes preserve token identity but require
separate decoder version/quality acceptance. Cached labels bind the exact identity.
`representation_id` MUST version/fingerprint the frozen SOMA hierarchy, feature
layout/extraction modes, fps and coordinate convention. Normalization is bound
separately by its required actual-tensor SHA256, not by this string.
`freeze_vocabulary(representation_id, normalization_sha256=actual_stats_hash,
parent=old_identity)` returns the old identity
when content is unchanged, otherwise creates the new parent link. `encode` requires
a frozen vocabulary. Kmeans vocabularies must be initialized before freezing;
label generation never lazily initializes or updates EMA.
`read_token_cache` checks the receipt normalization, vocabulary content identity
and full nested identity. It has no legacy-cache or missing-normalization path.
A owns training-side equality checks between label lineage and this identity;
this worker does not modify A/E files.

For native 10 warmstart, explicitly copy the compatible `old_quantizer.vq.state_dict()`
into `new_quantizer.vq` with `strict=True` before freezing. This preserves VQ tensor
names while bypassing the old total-index methods. 128/256 require newly trained
vocabularies and relabeled Pose targets; no partial load or padding path exists.

## Checks and remaining work

### Encoder Token-0 Dependency / Zero-Adam

Read-only final-encoder check requested by parent after the first-latent/history
CE-mask issue. This used engineering feature tensors, not a motion-quality test
or evidence about accepted real training data. Existing four-Adam artifacts were
not read, replaced, or rerun. No training/runtime source was changed for this check.

- Weights: `out/pfnn_mixed_20261015/final_acceptance_flat_v1/weights.pt`,
  strictly loaded `codec.encoder.*` into a separate CPU `Encoder` instance.
  Final checkpoint SHA256 from the resolved configuration:
  `0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646`.
  Exact encoder tensor-state SHA256 measured in this check:
  `1fa4c6855607567f5983a07150467431604d40d0322438ca319d6418d37f65f4`.
- Actual architecture: input268, width512, latent256, down_t2, stride2, depth4,
  dilation growth3, ReLU, no normalization. Each residual stack executes
  dilations 27/9/3/1, with symmetric padding; it is noncausal.
- Nominal receptive-field widths through input-conv, downsample1, residual1,
  downsample2, residual2, output-conv: `3 -> 6 -> 166 -> 172 -> 492 -> 500`.
  Output temporal stride is4. Latent0 nominal input coordinates are `[-248,251]`.
  Propagating dependency sets through actual finite-length padded convolutions
  gives latent0 support at every frame `0..31` for a 32-frame input.
- Execution: CPU FP32, 2 Torch threads, seed2333, encoder eval, all encoder
  parameters frozen, input gradients enabled, optimizer never constructed,
  actual Adam updates0, formal exposures0. No GPU or service operation.
- Fixture: input `[1,268,32]` sampled as `0.1 * randn` after constructing/loading
  the encoder; output `[1,256,8]`. A second random direction was zeroed on frames
  `0..3`, then added at scale0.1. Prefix equality was checked bitwise.
- Future input perturbation L2: `8.649275779724121`.
  Latent0 change L2: `4.583591938018799`;
  latent0 maximum absolute change: `1.0690863132476807`.
- VJP of latent0 against a unit-normalized random 256-D probe:
  future-input gradient L2 `3.484330177307129`,
  history-input gradient L2 `2.922895669937134`.
  Every future frame `4..31` had nonzero input-gradient norm, with finite gradients.

Conclusion for training: latent0 is not a history-only token. Excluding decoded
motion loss on executed frames0..3 does not authorize excluding codec latent0
from Pose CE. Keep motion-frame validity/history masks and valid-code CE masks
separate; supervise all valid code positions, including position0. The result was
sent directly to A/Raman and Plato. A's source already exposes a distinct
`masks.motion_loss_frames` field; this worker did not edit it.

### Codec And Loss CPU Coverage

- 44 codec/loss CPU tests passed (2 Torch CPU threads, CUDA hidden), including
  all-invalid/NaN labels, moving translated/rotated objects, prediction-independent
  contact gates, fixed versus sliding/touch semantics, event/anchor/object switches,
  negative/nonfinite weight rejection, no-object validity, selected official skin,
  metric FK and encoder/decoder/contact/geometry gradients.
- Normalization regression:
  `test_same_encoder_codebook_and_repr_with_changed_stats_rejects_old_cache`
  holds encoder/codebook/representation fixed, changes the actual std tensor,
  recomputes the normalization digest, obtains a new identity/parent link and
  rejects the old cache. Six malformed hashes are rejected before freezing any
  module; omission of the required freeze argument and tampering of either cache
  receipt or nested vocabulary normalization are also covered. These add ten
  CPU-only cases to the earlier 34-test codec/loss suite, with no optimizer.
- Three normalization/lineage bridge tests also passed without Adam:
  A's `test_final_codec_identity_cache_bridge`, and E's
  `test_engineering_codec_cache_lineage_cannot_authorize_formal_training` /
  `test_normalization_identity_change_rejects_existing_cache`. A's builder and
  attachment path both compare lineage normalization to the required identity
  field. A/E own those implementation/callsite updates; this worker only ran
  their explicitly zero-optimizer checks, not their optimizer/4-Adam tests.
- Object-ID regression: `test_noncontiguous_object_ids_resolve_after_table_reordering`
  passed both orderings with different per-batch tables, IDs 7/42, missing IDs,
  and per-object validity. `test_noncontiguous_moving_contact_losses_and_gradients_survive_table_reordering`
  passed with positive anchor/gap/slip losses and nonzero finite body-point gradients.
  Contact IDs now remain persistent end to end, matched to `scene.object_ids`.
- Plato confirmed that the NumPy adapter now preserves contacts/query object IDs
  and exposes separate derived `object_table_index` fields. Four relevant integration
  CPU tests passed: `test_sparse_object_ids_and_swapped_table_preserve_contact_losses`,
  `test_object_ids_and_numpy_masks_are_preserved`,
  `test_fixture_queries_reach_real_contact_and_distance_losses`, and
  `test_contact_and_body_sdf_backpropagate_to_predicted_points`.
  Field synchronization confirmed: actual measured gates are nested
  `contacts.actual_valid`; query and event gates are `contact_surface_valid` and
  `contact_interval_valid`; codec loss does not read a top-level `contact_valid`.
- Native 10 VQ strict-state inheritance was checked with actual CPU forward,
  per-head index generation, dequantization and commitment outputs; quantized and
  decoded tensors match exactly. This is VQ compatibility, not a claim that a new
  contact head or the entire final formal checkpoint is execution-equivalent.
- Actual local LAFAN retarget seed
  `out/lafan_soma_20261006/preflight-v4/walk1_subject1.npz`, first 13 frames:
  added 60-degree world yaw and 0.7 m vertical origin, then native canonicalization
  and explicit restoration. Max absolute FK error was 4.76837158203125e-7 m;
  selected official-skin error was 5.364418029785156e-7 m. Backward gradients were finite.
  Raw source root positions were passed as explicit FK `posed_joints`, avoiding
  confusion with native `translation` (which is an offset from neutral joints).
- CPU test command: `D:/.anaconda/envs/motionbricks/python.exe -m pytest -q
  tests/test_scene_control_v2_codec.py tests/test_scene_control_v2_losses.py` with
  `PYTHONPATH=.;out/vendor/kimodo`. Only 8 existing Kimodo TorchScript deprecation
  warnings remain.

Integration must still supply differentiable assigned-object/union SDF results,
verified noncontact sample masks, reliable same-event intervals and actual executed
prefix. The shared contracts/preflight worker owns that bridge. Quality weights,
capacity selection, corpus labels, optimizer experiments and full-skin/8-substep
acceptance are not selected or run here. No GPU/optimizer, SSH, training restart,
service changes, or quality-gate changes have been performed.
