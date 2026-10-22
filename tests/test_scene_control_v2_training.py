"""CPU unit fixtures plus an explicitly invoked, authorized diagnostic helper."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from motionbricks.scene_control_v2.curriculum import (
    BatchRequest, Curriculum, CurriculumPhase, ITERATION_POLICY, remask_argmax,
)
from motionbricks.scene_control_v2.training import (
    BATCH_SCHEMA, CONFIG_SCHEMA, LINEAGE_SCHEMA, SHA_FIELDS, NativePoseAdapter,
    SceneControlTrainer, TrainingBudget, attach_final_codec_labels, build_native_pose_adapter,
    dual_to_global_features, make_training_label_lineage, preflight_training_config, prepare_native_pose_inputs,
    run_bounded_trial, select_root_condition, validate_training_batch,
)


POLICY = "engineering.body-time-root.v1"


def _lineage():
    return {"schema": LINEAGE_SCHEMA, "frozen": True, "final": True, "data_kind": "engineering",
            "identity_semantics": "simulated_engineering_fixture_not_codec_reconstruction",
            **{name: "a" * 64 for name in SHA_FIELDS},
            "representation_id": "engineering-soma", "heads": 2,
            "entries_per_head": 4, "frames_per_token": 4}


def _request(horizon=8, modes=("all_mask", "random_mask", "self_iteration"), iteration=3):
    size = len(modes)
    return BatchRequest(horizon, tuple("predicted" if i % 2 else "source" for i in range(size)),
                        tuple("generated" if i % 2 else "source" for i in range(size)),
                        tuple(modes), tuple(iteration if m == "self_iteration" else 0 for m in modes), POLICY)


def _batch(request, root_dim=2):
    size, n = request.batch_size, request.horizon_tokens
    target_ids = [f"engineering-target-{i}" for i in range(size)]
    history_ids = [f"engineering-history-{i}" for i in range(size)]
    condition_ids = [f"engineering-root-{i}" for i in range(size)]
    world = torch.zeros(size, n * 4 + 1, 366)
    world[..., 1] = 1
    world[..., 3] = 1
    metadata = dict(policy_id=POLICY, horizon_tokens=n, root_condition_id=condition_ids, history_id=history_ids)
    token_valid = torch.ones(size, n, 2, dtype=torch.bool)
    motion_loss_frames = torch.arange(n * 4)[None].expand(size, -1) >= 4
    return {"schema": BATCH_SCHEMA, "data_kind": "engineering", "world": world,
            "history": world[:, :4].clone(), "root_source": list(request.root_source),
            "history_source": list(request.history_source),
            "root_condition": {"values": torch.zeros(size, n * 4, root_dim),
                               "height": torch.zeros(size, n * 4, 1), "source": list(request.root_source),
                               "target_id": target_ids, "history_id": history_ids, "condition_id": condition_ids},
            "source_id": ["synthetic" for _ in range(size)], "family_id": ["engineering" for _ in range(size)],
            "scene_id": ["engineering-scene" for _ in range(size)], "policy_id": [POLICY for _ in range(size)],
            "target_id": target_ids, "history_id": history_ids,
            "observation": {**metadata, "points": torch.zeros(size, 2, 9, 9, 3),
                            "valid": torch.ones(size, 2, 9, 9, dtype=torch.bool)},
            "volume": {**metadata, "features": torch.zeros(size, 5, 6),
                       "valid": torch.ones(size, 5, dtype=torch.bool)},
            "masks": {"frames": torch.ones(size, n * 4, dtype=torch.bool),
                      "motion_loss_frames": motion_loss_frames, "tokens": token_valid},
            "contact_targets": torch.zeros(size, n * 4, 15),
            "contact_valid": torch.zeros(size, n * 4, 15, dtype=torch.bool),
            "recovery": {"accepted": torch.ones(size, dtype=torch.bool), "history_id": history_ids,
                         "target_id": target_ids, "teacher_id": ["engineering-teacher" for _ in range(size)]},
            "pose_tokens": torch.zeros(size, n, 2, dtype=torch.long), "label_lineage": _lineage()}


class _EngineeringVolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 8)

    def forward(self, volume):
        return self.projection(volume["features"]), volume["valid"]


def _build_engineering_adapter():
    args = dict(n_embd=32, n_head=4, n_layers=1, pose_feat_width=16, root_feat_width=8,
                token_length_feat_width=8, pose_token_mlp_num_layers=1, terrain_conditioning=True,
                pose_vqvae=dict(nb_code=16, code_dim=8, num_heads=2),
                root_vqvae=dict(nb_code=16, code_dim=8, num_heads=2), min_tokens=8, max_tokens=16,
                down_t=2, cond_root_feature_is_from_motion_rep="local",
                cond_root_feature="root_without_hip_height_without_heading",
                local_pose_feature="joint_positions_and_rotations_and_hip_height")
    result = build_native_pose_adapter(pose_args=args, dual_stats_mean=torch.zeros(370),
                                       dual_stats_std=torch.ones(370), volume_encoder=_EngineeringVolumeEncoder(),
                                       volume_width=8, backbone_state=None).eval()
    assert isinstance(result, NativePoseAdapter)
    with torch.no_grad():
        result.backbone._terrain_cross_attention.gate.fill_(.7)
    return result


@pytest.fixture(scope="module")
def adapter():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    result = _build_engineering_adapter()
    yield result
    torch.set_num_threads(previous)


def _curriculum():
    return Curriculum(policy_id=POLICY, phases=[CurriculumPhase(0, .5, .5, (.3, .3, .4))],
                      horizon_probabilities=(.4, .3, .3))


def _dual_fixture(size, frames):
    values = torch.arange(size * frames * 370, dtype=torch.float32).reshape(size, frames, 370) / 10000
    values[..., 1] = 1
    values[..., 3] = 1
    values[..., 4] = 0
    values[..., 5:9] = 12345
    return values


def _legacy_dual_pose_logits(adapter, tokens, dual_history, batch):
    """Historical dual370 preprocessing reference, used only for zero-Adam parity."""
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    backbone = adapter.backbone
    size, frames = len(dual_history), batch["world"].shape[1] - 1
    dual = backbone.motion_rep.dual_rep
    global_history = dual.get_feature_subset(dual_history, "global")
    local = dual.global_to_local(global_history, is_normalized=False, to_normalize=True,
                                 lengths=torch.full((size,), 4, dtype=torch.long))
    pose = extract_feature_from_motion_rep(local, backbone.local_motion_rep, backbone._args["local_pose_feature"])
    condition = global_history.new_zeros(size, frames, pose.shape[-1])
    condition[:, :4] = pose
    known = torch.arange(frames)[None].expand(size, -1) < 4
    return backbone(tokens, batch["root_condition"]["values"], condition, known,
                    torch.full((size, 1), frames // 4, dtype=torch.long),
                    terrain_observation=dict(batch["observation"], scene_control_v2_volume=batch["volume"]),
                    root_height_trajectory=batch["root_condition"]["height"])["pose_logits"]


def test_native_global366_is_canonical_and_dual370_requires_explicit_conversion(adapter):
    batch = _batch(_request())
    assert batch["world"].shape == (3, 33, 366)
    assert batch["history"].shape == (3, 4, 366)
    assert validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)["horizon_tokens"] == 8
    dual_rep = adapter.backbone.motion_rep.dual_rep
    assert dual_rep.motion_rep_dim == 370
    assert dual_rep.global_motion_rep.motion_rep_dim == 366
    assert dual_rep.local_motion_rep.motion_rep_dim == 365
    raw_dual = _dual_fixture(3, 33)
    unchanged = raw_dual.clone()
    converted = dual_to_global_features(raw_dual, dual_rep=dual_rep)
    assert converted.shape == (3, 33, 366)
    assert torch.equal(converted, torch.cat((raw_dual[..., :5], raw_dual[..., 9:]), -1))
    assert torch.equal(raw_dual, unchanged)
    altered_local_root = raw_dual.clone()
    altered_local_root[..., 5:9] *= -2
    assert torch.equal(converted, dual_to_global_features(altered_local_root, dual_rep=dual_rep))
    with pytest.raises(ValueError, match="raw floating-point.*370"):
        dual_to_global_features(converted, dual_rep=dual_rep)
    batch["world"], batch["history"] = raw_dual, raw_dual[:, :4]
    with pytest.raises(ValueError, match="global shape.*366"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)
    with pytest.raises(ValueError, match="history.*366"):
        prepare_native_pose_inputs(adapter.backbone, batch)
    batch["world"] = converted
    with pytest.raises(ValueError, match="history.*366"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)


def test_native_global_history_is_not_reindexed_as_dual(adapter, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A global366 history must not be indexed as dual370")

    monkeypatch.setattr(adapter.backbone.motion_rep.dual_rep, "get_feature_subset", forbidden)
    batch = _batch(_request(), adapter.backbone._args["local_root_dim"])
    condition = prepare_native_pose_inputs(adapter.backbone, batch)
    assert condition["pose_cond"].shape == (3, 32, 268)


@pytest.mark.parametrize("horizon", [8, 12, 16])
@torch.no_grad()
def test_explicit_dual_to_global_has_identical_logits_across_four_passes(adapter, horizon):
    request = _request(horizon, modes=("all_mask", "all_mask"))
    batch = _batch(request, adapter.backbone._args["local_root_dim"])
    dual_world = _dual_fixture(2, horizon * 4 + 1)
    batch["world"] = dual_to_global_features(dual_world, dual_rep=adapter.backbone.motion_rep.dual_rep)
    batch["history"] = batch["world"][:, :4].clone()
    validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)
    tokens = torch.full_like(batch["pose_tokens"], 4)
    adapter.eval()
    for iteration in range(4):
        expected = _legacy_dual_pose_logits(adapter, tokens, dual_world[:, :4], batch)
        actual = adapter.forward_pose(tokens, batch)["pose_logits"]
        assert torch.equal(expected, actual)
        if iteration < 3:
            tokens = remask_argmax(actual, next_iteration=iteration + 1, mask_id=4)


@pytest.mark.parametrize("horizon", [8, 12, 16])
@torch.no_grad()
def test_real_native_pose_forward_all_lengths_and_token_modes(adapter, horizon):
    request = _request(horizon)
    batch = _batch(request, adapter.backbone._args["local_root_dim"])
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY).eval()
    output = trainer(batch, request=request, seed=77)
    assert output["pose_logits"].shape == (3, horizon, 2, 4)
    assert torch.isfinite(output["loss"])
    assert output["input_tokens"][0].eq(4).all()
    assert output["input_tokens"][1].eq(4).any()
    assert output["input_tokens"][2].eq(4).any()
    assert output["input_tokens"][2].ne(4).any()
    assert output["focused_tokens"][0, 0].all()
    assert not batch["masks"]["motion_loss_frames"][:, :4].any()
    assert output["token_modes"] == {"all_mask": 1, "random_mask": 1, "self_iteration": 1}
    assert not batch["contact_valid"].any()
    condition = prepare_native_pose_inputs(adapter.backbone, batch)
    assert condition["has_pose_cond"][:, :4].all()
    assert not condition["has_pose_cond"][:, 4:].any()
    assert condition["pose_cond"][:, 4:].eq(0).all()


@torch.no_grad()
def test_selected_root_and_volume_really_affect_native_forward(adapter):
    request = _request(modes=("all_mask",))
    batch = _batch(request, adapter.backbone._args["local_root_dim"])
    tokens = torch.full_like(batch["pose_tokens"], 4)
    initial = adapter.forward_pose(tokens, batch)["pose_logits"]
    root_changed = deepcopy(batch)
    root_changed["root_condition"]["values"][:, 4:] += 2
    assert not torch.allclose(initial, adapter.forward_pose(tokens, root_changed)["pose_logits"])
    volume_changed = deepcopy(batch)
    volume_changed["volume"]["features"] += 3
    assert not torch.allclose(initial, adapter.forward_pose(tokens, volume_changed)["pose_logits"])
    future_changed = deepcopy(batch)
    future_changed["world"][:, 4:] += 8
    assert torch.equal(initial, adapter.forward_pose(tokens, future_changed)["pose_logits"])


def test_generated_history_cannot_be_hard_spliced_to_source_future():
    batch = _batch(_request())
    batch["history"][1, :, 0] += 1
    original = batch["world"].clone()
    with pytest.raises(ValueError, match="matched recovery target"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)
    assert torch.equal(original, batch["world"])


def test_explicit_root_branch_selection_never_substitutes_source():
    source = _batch(_request())["root_condition"]
    predicted = deepcopy(source)
    predicted["values"] += 1
    predicted["height"] += 2
    predicted["condition_id"] = ["prediction-0", "prediction-1", "prediction-2"]
    chosen = select_root_condition(source, predicted, root_source=("source", "predicted", "source"))
    assert chosen["values"][0].eq(0).all() and chosen["values"][1].eq(1).all()
    assert chosen["height"][1].eq(2).all() and source["height"].eq(0).all()
    assert chosen["condition_id"][1] == "prediction-1"
    with pytest.raises(KeyError, match="values"):
        select_root_condition(source, {}, root_source=("source", "predicted", "source"))


@torch.no_grad()
def test_self_iteration_input_is_model_generated_not_source_labels(adapter):
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY).eval()
    request = _request(modes=("self_iteration",))
    batch = _batch(request, adapter.backbone._args["local_root_dim"])
    changed = deepcopy(batch)
    changed["pose_tokens"].fill_(1)
    first = trainer(batch, request=request, seed=77)
    second = trainer(changed, request=request, seed=77)
    assert torch.equal(first["input_tokens"], second["input_tokens"])
    assert not torch.equal(first["pose_ce"], second["pose_ce"])


def test_unknown_contact_values_are_masked_not_made_into_negatives():
    batch = _batch(_request())
    batch["contact_targets"].fill_(float("nan"))
    assert validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)["batch_size"] == 3
    batch["contact_valid"][0, 4, 0] = True
    with pytest.raises(ValueError, match="Valid contact_targets"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)


@torch.no_grad()
def test_codec_ce_supervises_token_zero_independently_of_motion_history_mask(adapter):
    request = _request(modes=("all_mask",))
    batch = _batch(request, adapter.backbone._args["local_root_dim"])
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY).eval()
    result = trainer(batch, request=request, seed=2333)
    assert result["focused_tokens"][:, 0].all()
    assert not batch["masks"]["motion_loss_frames"][:, :4].any()
    expected = torch.nn.functional.cross_entropy(result["pose_logits"].reshape(-1, 4),
                                                 batch["pose_tokens"].reshape(-1), reduction="none").mean()
    assert torch.equal(result["pose_ce"], expected)
    batch["masks"]["motion_loss_frames"].zero_()
    assert torch.equal(trainer(batch, request=request, seed=2333)["pose_ce"], result["pose_ce"])
    batch["masks"]["tokens"][:, 0] = False
    legacy = trainer(batch, request=request, seed=2333)
    assert not legacy["focused_tokens"][:, 0].any()
    expected_legacy = torch.nn.functional.cross_entropy(legacy["pose_logits"][:, 1:].reshape(-1, 4),
                                                       batch["pose_tokens"][:, 1:].reshape(-1), reduction="none").mean()
    assert torch.equal(legacy["pose_ce"], expected_legacy)


def test_native_training_graph_has_gradients_without_an_optimizer(adapter):
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY).train()
    request = _request()
    batch = _batch(request, adapter.backbone._args["local_root_dim"])
    trainer.zero_grad(set_to_none=True)
    weights = {name: value.detach().clone() for name, value in adapter.named_parameters()}
    result = trainer(batch, request=request, seed=100)
    result["loss"].backward()
    assert adapter.backbone._proj_pose_output_logit.weight.grad.abs().sum() > 0
    assert adapter.backbone._terrain_encoder.volume_projection.weight.grad.abs().sum() > 0
    assert all(torch.equal(weights[name], value) for name, value in adapter.named_parameters())
    trainer.zero_grad(set_to_none=True)
    trainer.eval()


@pytest.mark.parametrize("field", ["accepted", "target_id", "history_id"])
def test_generated_history_requires_accepted_matched_teacher(field):
    batch = _batch(_request())
    if field == "accepted":
        batch["recovery"][field][1] = False
    else:
        batch["recovery"][field] = ["unrelated"] * 3
    with pytest.raises(ValueError, match="accepted continuous teacher"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)


@pytest.mark.parametrize("field", ["policy_id", "horizon_tokens", "root_condition_id", "history_id"])
def test_reject_stale_or_wrong_horizon_observation(field):
    batch = _batch(_request())
    batch["volume"][field] = {"policy_id": "old", "horizon_tokens": 12,
                              "root_condition_id": ["old"] * 3, "history_id": ["old"] * 3}[field]
    with pytest.raises(ValueError):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)


def test_missing_field_and_wrong_frozen_lineage_are_errors():
    batch = _batch(_request())
    del batch["volume"]
    with pytest.raises(KeyError, match="volume"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)
    batch = _batch(_request())
    batch["label_lineage"]["encoder_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="label lineage"):
        validate_training_batch(batch, lineage=_lineage(), policy_id=POLICY)


def test_curriculum_is_deterministic_explicit_and_covers_distributions():
    curriculum = _curriculum()
    assert curriculum.draw(10, exposure=0, seed=9) == curriculum.draw(10, exposure=0, seed=9)
    requests = [curriculum.draw(12, exposure=i * 12, seed=i) for i in range(30)]
    assert {r.horizon_tokens for r in requests} == {8, 12, 16}
    assert {x for r in requests for x in r.token_mode} == {"all_mask", "random_mask", "self_iteration"}
    assert {x for r in requests for x in r.root_source} == {"source", "predicted"}
    assert {x for r in requests for x in r.history_source} == {"source", "generated"}
    with pytest.raises(ValueError, match="8,12,16"):
        Curriculum(policy_id=POLICY, phases=curriculum.phases, horizon_probabilities=(1., 0., 0.))


def test_own_iteration_has_exact_shared_four_pass_remask_counts():
    logits = torch.arange(4.).expand(2, 8, 2, 4)
    for iteration, count in ((1, 14), (2, 11), (3, 6)):
        tokens = remask_argmax(logits, next_iteration=iteration, mask_id=4)
        assert (tokens.eq(4).flatten(1).sum(-1) == count).all()


def test_final_codec_identity_cache_bridge():
    from motionbricks.scene_control_v2.quantizer import FrozenCodebookIdentity, QuantizerConfig, make_token_cache

    identity = FrozenCodebookIdentity(QuantizerConfig(entries_per_head=10), "a" * 64, "b" * 64, "soma-v2",
                                      normalization_sha256="e" * 64)
    receipt = make_training_label_lineage(identity, data_kind="engineering", codec_sha256="c" * 64, scene_encoder_sha256="d" * 64,
                                         normalization_sha256="e" * 64, label_manifest_sha256="f" * 64)
    cache = make_token_cache(torch.zeros(1, 8, 8, dtype=torch.long), identity)
    attached = attach_final_codec_labels({"untouched": 1}, cache, identity, lineage=receipt)
    assert attached["pose_tokens"].shape == (1, 8, 8)
    assert attached["label_lineage"]["vocabulary_id"] == identity.identity_id
    changed = dict(receipt, vocabulary_id="0" * 64)
    with pytest.raises(ValueError, match="differs from codec"):
        attach_final_codec_labels({}, cache, identity, lineage=changed)
    with pytest.raises(ValueError, match="normalization differs"):
        make_training_label_lineage(identity, data_kind="engineering", codec_sha256="c" * 64,
                                    scene_encoder_sha256="d" * 64, normalization_sha256="0" * 64,
                                    label_manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="differs from codec"):
        attach_final_codec_labels({}, cache, identity, lineage=dict(receipt, normalization_sha256="0" * 64))


def _config():
    return {"schema": CONFIG_SCHEMA, "policy_id": POLICY, "label_lineage": _lineage(),
            "curriculum": {"horizon_probabilities": [.4, .3, .3], "phases": [
                {"start_exposure": 0, "source_root_probability": .5, "source_history_probability": .5,
                 "token_probabilities": [.3, .3, .4]}]},
            "budget": dict(kind="diagnostic", max_batches=1, batch_size=1,
                           max_exposures=1, max_adam_updates=0)}


def test_preflight_is_zero_update_and_cli_never_imports_factory(tmp_path):
    from scripts.train_scene_control_v2 import main

    config = _config()
    assert preflight_training_config(config)["actual_adam_updates"] == 0
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "preflight"
    result = main(["--config", str(path), "--output", str(output), "--factory", "nonexistent:forbidden"])
    assert result["mode"] == "preflight"
    assert result["formal_exposures"] == 0
    assert not (output / "checkpoint.pt").exists()
    with pytest.raises(ValueError, match="overwrite"):
        main(["--config", str(path), "--output", str(output)])


@torch.no_grad()
def test_bounded_native_forward_trial_is_not_formal_training(adapter, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer may be constructed in this CPU forward fixture")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY)
    budget = TrainingBudget("diagnostic", max_batches=2, batch_size=2, max_exposures=4, max_adam_updates=0)
    provider = lambda request: _batch(request, adapter.backbone._args["local_root_dim"])
    output = tmp_path / "trial"
    result = run_bounded_trial(trainer, provider, curriculum=_curriculum(), budget=budget,
                               output=output, mode="forward")
    assert result["forward_examples"] == 4
    assert result["actual_adam_updates"] == result["formal_exposures"] == result["diagnostic_exposures"] == 0
    assert result["status"] == "complete" and not result["quality_accepted"]
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        run_bounded_trial(trainer, provider, curriculum=_curriculum(), budget=budget,
                          output=output, mode="forward")


def test_optimizer_execution_requires_separate_explicit_budget():
    budget = TrainingBudget("diagnostic", 2, 2, 4, 0)
    with pytest.raises(ValueError, match="approval_id"):
        budget.validate("train")
    config = _config()
    config["budget"]["max_adam_updates"] = 1
    with pytest.raises(ValueError, match="zero Adam"):
        preflight_training_config(config)


def test_engineering_label_identity_cannot_enter_formal_before_adam(adapter, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Formal rejection must happen before constructing an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY)
    budget = TrainingBudget("formal", 1, 1, 1, 1, "unit-admission-only", 5e-5, .5)
    with pytest.raises(ValueError, match="Engineering label identities"):
        run_bounded_trial(trainer, lambda request: dict(_batch(request), data_kind="formal"),
                          curriculum=_curriculum(), budget=budget, output=tmp_path / "forbidden", mode="train")
    assert not (tmp_path / "forbidden").exists()
    config = _config()
    config["budget"] = asdict(budget)
    with pytest.raises(ValueError, match="Engineering label identities"):
        preflight_training_config(config, mode="train")


@torch.no_grad()
def verify_saved_global366_parity(checkpoint, output, *, bundle_path):
    """Read the exhausted four-Adam artifact; perform only bounded CPU forwards."""
    from motionbricks.scene_control_v2.quantizer import tensor_state_sha256
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.motionlib.core.utils.stats import Stats

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_files = sorted(path for path in checkpoint.parent.iterdir() if path.is_file())
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    historical = json.loads((checkpoint.parent / "execution_verification.json").read_text(encoding="utf-8"))
    assert saved["receipt"]["actual_adam_updates"] == 4
    assert saved["receipt"]["diagnostic_exposures"] == 8
    assert saved["receipt"]["formal_exposures"] == 0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False, mmap=True)
        real_history = bundle["initial_history"]
        assert real_history.shape == (1, 4, 366)
        assert bundle["stats_mean"].shape == bundle["stats_std"].shape == (370,)
        real_stats = Stats()
        real_stats.register_from_tensors(bundle["stats_mean"], bundle["stats_std"])
        real_rep = GlobalRootGlobalJoints(30, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=real_stats)
        real_local = real_rep.dual_rep.global_to_local(real_history, False, True, lengths=torch.tensor([4]))
        assert real_local.shape == (1, 4, 365) and torch.isfinite(real_local).all()
        model = _build_engineering_adapter()
        trainer = SceneControlTrainer(model, lineage=saved["receipt"]["label_lineage"], policy_id=POLICY).eval()
        trainer.load_state_dict(saved["model"], strict=True)
        before = tensor_state_sha256(trainer.state_dict())
        request = _curriculum().draw(2, exposure=6, seed=2336)
        frames = request.horizon_tokens * 4
        legacy_world = torch.zeros(2, frames + 1, 370)
        legacy_world[..., 1] = 1
        legacy_world[..., 3] = 1
        unchanged_legacy = legacy_world.clone()
        batch = _batch(request, model.backbone._args["local_root_dim"])
        # Preserve the historical diagnostic's explicit CE policy when replaying it.
        batch["masks"]["tokens"][:, 0] = False
        batch["world"] = dual_to_global_features(legacy_world, dual_rep=model.backbone.motion_rep.dual_rep)
        batch["history"] = batch["world"][:, :4].clone()
        result = trainer(batch, request=request, seed=2336)
        expected = _legacy_dual_pose_logits(model, result["input_tokens"], legacy_world[:, :4], batch)
        assert torch.equal(expected, result["pose_logits"])
        assert float(result["pose_ce"]) == historical["reloaded_pose_ce"]
        tokens = torch.full_like(batch["pose_tokens"], 4)
        pass_matches = []
        for iteration in range(4):
            old_logits = _legacy_dual_pose_logits(model, tokens, legacy_world[:, :4], batch)
            new_logits = model.forward_pose(tokens, batch)["pose_logits"]
            assert torch.equal(old_logits, new_logits)
            pass_matches.append(True)
            if iteration < 3:
                tokens = remask_argmax(new_logits, next_iteration=iteration + 1, mask_id=4)
        assert torch.equal(legacy_world, unchanged_legacy)
        after = tensor_state_sha256(trainer.state_dict())
        assert before == after == historical["trained_state_sha256"]
        assert source_hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
        proof = {"schema": "motionbricks.scene_control_v2.global366_zero_adam_parity.v1",
                 "data_kind": "engineering", "device": "cpu", "checkpoint": str(checkpoint.resolve()),
                 "source_artifact_file_sha256": source_hashes, "source_artifacts_unchanged": True,
                 "historical_adam_updates": 4, "historical_diagnostic_exposures": 8,
                 "additional_adam_updates": 0, "additional_training_exposures": 0, "formal_exposures": 0,
                 "legacy_dual_world_shape": list(legacy_world.shape),
                 "canonical_global_world_shape": list(batch["world"].shape),
                 "canonical_global_history_shape": list(batch["history"].shape),
                 "normalization_features": 370, "local_features": 365,
                 "real_bundle_history_shape": list(real_history.shape),
                 "real_bundle_stats_shape": list(bundle["stats_mean"].shape),
                 "real_bundle_normalized_local_shape": list(real_local.shape),
                 "real_bundle_read_scope": "mmap_cpu_history_and_normalization_only_no_model_or_optimizer",
                 "conversion": "explicit_native_global_index_selection_no_padding",
                 "four_pass_logits_bit_exact": pass_matches, "replayed_final_logits_bit_exact": True,
                 "replayed_pose_ce": float(result["pose_ce"]), "model_state_before_sha256": before,
                 "model_state_after_sha256": after, "quality_accepted": False, "budget_auto_resume": False}
        output.mkdir(parents=True, exist_ok=False)
        with (output / "global366_parity.json").open("x", encoding="utf-8") as report:
            json.dump(proof, report, indent=2, sort_keys=True)
        return proof
    finally:
        torch.set_num_threads(previous_threads)


def execute_authorized_cpu_diagnostic(output, *, budget, seed):
    """Manual invocation only; pytest never executes this four-Adam experiment."""
    from motionbricks.scene_control_v2.quantizer import tensor_state_sha256

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    budget.validate("train")
    assert (budget.kind, budget.max_batches, budget.batch_size, budget.max_exposures,
            budget.max_adam_updates, budget.learning_rate, budget.clip_norm, seed) == (
                "diagnostic", 4, 2, 8, 4, 5e-5, .5, 2333)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)

        def build():
            model = _build_engineering_adapter()
            model.backbone._pose_token_emb.requires_grad_(False)
            model.backbone._terrain_encoder.volume_encoder.requires_grad_(False)
            return SceneControlTrainer(model, lineage=_lineage(), policy_id=POLICY)

        trainer = build()
        assert all(value.device.type == "cpu" for value in trainer.state_dict().values())
        before = {name: value.detach().clone() for name, value in trainer.state_dict().items()}
        trainable = {name for name, value in trainer.named_parameters() if value.requires_grad}
        frozen_names = set(before) - trainable
        assert frozen_names
        batches = []

        def provider(request):
            batch = _batch(request, trainer.model.backbone._args["local_root_dim"])
            batches.append((request, batch))
            return batch

        receipt = run_bounded_trial(trainer, provider, curriculum=_curriculum(), budget=budget,
                                    output=output, mode="train", seed=seed)
        assert receipt["actual_adam_updates"] == receipt["diagnostic_adam_updates"] == 4
        assert receipt["diagnostic_exposures"] == receipt["forward_examples"] == 8
        assert receipt["formal_exposures"] == receipt["formal_adam_updates"] == 0
        assert receipt["status"] == "complete" and not receipt["quality_accepted"]
        assert receipt["label_data_kind"] == "engineering"
        assert len(batches) == 4
        after = trainer.state_dict()
        changed = sorted(name for name in trainable if not torch.equal(before[name], after[name]))
        assert changed
        assert all(torch.equal(before[name], after[name]) for name in frozen_names)
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert saved["receipt"] == receipt
        assert all(torch.equal(saved["model"][name], value) for name, value in after.items())
        adam_steps = sorted({int(value["step"]) for value in saved["optimizer"]["state"].values()})
        assert adam_steps == [4]
        assert saved["optimizer"]["param_groups"][0]["lr"] == budget.learning_rate
        restored = build()
        restored.load_state_dict(saved["model"], strict=True)
        trainer.eval()
        restored.eval()
        request, batch = batches[-1]
        with torch.no_grad():
            expected = trainer(batch, request=request, seed=seed + 3)
            actual = restored(batch, request=request, seed=seed + 3)
        assert torch.equal(expected["pose_logits"], actual["pose_logits"])
        assert torch.isfinite(actual["loss"])
        assert all(value.device.type == "cpu" for value in restored.state_dict().values())
        verification = {
            "schema": "motionbricks.scene_control_v2.engineering_adam_verification.v1",
            "data_kind": "engineering", "device": "cpu", "seed": seed,
            "scope": "width32_layer1_heads2_entries4_native_pose_with_simulated_labels",
            "approval_id": budget.approval_id, "actual_adam_updates": 4,
            "diagnostic_exposures": 8, "formal_exposures": 0,
            "quality_accepted": False, "changed_trainable_names": changed,
            "unchanged_frozen_and_buffer_names": sorted(frozen_names),
            "initial_state_sha256": tensor_state_sha256(before),
            "trained_state_sha256": tensor_state_sha256(after),
            "frozen_state_before_sha256": tensor_state_sha256({name: before[name] for name in frozen_names}),
            "frozen_state_after_sha256": tensor_state_sha256({name: after[name] for name in frozen_names}),
            "checkpoint_strict_load": True, "checkpoint_forward_bit_exact": True,
            "checkpoint_optimizer_steps": adam_steps,
            "reloaded_forward_shape": list(actual["pose_logits"].shape),
            "reloaded_pose_ce": float(actual["pose_ce"]),
            "verification_reuses_last_training_batch": True,
            "budget_auto_resume": False,
        }
        with (output / "initial_state.pt").open("xb") as initial_file:
            torch.save(before, initial_file)
        with (output / "execution_verification.json").open("x", encoding="utf-8") as report:
            json.dump(verification, report, indent=2, sort_keys=True)
        return verification
    finally:
        torch.set_num_threads(previous_threads)


def test_failed_provider_records_exact_missing_field_and_zero_exposure(adapter, tmp_path):
    trainer = SceneControlTrainer(adapter, lineage=_lineage(), policy_id=POLICY)

    def provider(request):
        batch = _batch(request)
        del batch["contact_valid"]
        return batch

    output = tmp_path / "failed"
    with pytest.raises(KeyError, match="contact_valid"):
        run_bounded_trial(trainer, provider, curriculum=_curriculum(),
                          budget=TrainingBudget("diagnostic", 1, 1, 1, 0), output=output, mode="forward")
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["forward_examples"] == 0
    assert "contact_valid" in receipt["error"]
