from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOTIONBRICKS_ROOT.parent
sys.path.insert(0, str(MOTIONBRICKS_ROOT))
sys.path.insert(0, str(MOTIONBRICKS_ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_models import _resolved_backbone_args, native_config
from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL
from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


def _motion_rep():
    return GlobalRootGlobalJoints(30.0, SOMASkeleton30(), "somaskel30_dual_root_global_joints")


@pytest.fixture(params=["root", "pose"])
def backbone_case(request):
    rep = _motion_rep()
    stage = request.param
    config = native_config(stage, max_steps=100)
    if stage == "root":
        args = _resolved_backbone_args(stage, config)
        args.update(n_embd=32, n_head=4, n_layers_shared=1, n_layers_root_token=1,
                    width=32, depth=1, terrain_conditioning=False)
        create = lambda a: root_backbone_network(deepcopy(a), rep)
        call = _root_call
    else:
        fake_tokenizer = SimpleNamespace(num_code=16, code_dim=8, _num_heads=2)
        args = _resolved_backbone_args(stage, config, fake_tokenizer)
        args.update(n_embd=32, n_head=4, n_layers=1, pose_feat_width=16,
                    root_feat_width=8, token_length_feat_width=8, pose_token_mlp_num_layers=1,
                    terrain_conditioning=False)
        args["pose_vqvae"].update(nb_code=16, code_dim=8, num_heads=2)
        args["root_vqvae"].update(nb_code=16, code_dim=8, num_heads=2)
        create = lambda a: pose_backbone_network(rep, deepcopy(a))
        call = _pose_call

    torch.manual_seed(231)
    disabled = create(args).eval()
    enabled_args = deepcopy(args)
    enabled_args["terrain_conditioning"] = True
    enabled = create(enabled_args).eval()
    missing, unexpected = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith(("_terrain_encoder.", "_terrain_cross_attention.",
                               "_terrain_root_height_")) for key in missing)
    yield stage, disabled, enabled, call


def _root_call(backbone, terrain=None):
    mask = torch.ones(1, 8, dtype=torch.bool)
    return backbone(
        global_root_values=torch.zeros(1, 8, 5), has_global_root_values=mask,
        local_root_values=torch.zeros(1, 8, 4), has_local_root_values=mask,
        poses=torch.zeros(1, 8, 268), has_poses=mask,
        num_tokens=torch.tensor([[6]]), groundtruth_num_tokens=torch.tensor([[6]]),
        terrain_observation=terrain,
    )


def _pose_call(backbone, terrain=None):
    return backbone(
        pose_tokens=torch.zeros(1, 6, 2, dtype=torch.long),
        local_root_values=torch.zeros(1, 24, 4),
        pose_cond=torch.zeros(1, 24, 268), has_pose_cond=torch.zeros(1, 24, dtype=torch.bool),
        num_tokens=torch.tensor([[6]]), terrain_observation=terrain,
    )


def _pose_height_call(backbone, height):
    return backbone(
        pose_tokens=torch.zeros(1, 6, 2, dtype=torch.long),
        local_root_values=torch.zeros(1, 24, 4),
        pose_cond=torch.zeros(1, 24, 268), has_pose_cond=torch.zeros(1, 24, dtype=torch.bool),
        num_tokens=torch.tensor([[6]]), root_height_trajectory=height,
    )


def _terrain(batch=1, requires_grad=False):
    points = torch.zeros(batch, 2, 9, 9, 3, requires_grad=requires_grad)
    valid = torch.ones(batch, 2, 9, 9, dtype=torch.bool)
    return {"points": points, "valid": valid}


def _primary(output):
    return output["pred_global_root_values"] if "pred_global_root_values" in output else output["pose_logits"]


def test_disabled_state_dict_and_forward_compatibility(backbone_case):
    _, disabled, enabled, call = backbone_case
    assert not any(key.startswith("_terrain_") for key in disabled.state_dict())
    restored = deepcopy(disabled)
    restored.load_state_dict(disabled.state_dict(), strict=True)
    assert torch.equal(_primary(call(disabled)), _primary(call(restored)))
    assert torch.equal(_primary(call(disabled)), _primary(call(enabled)))


def test_disabled_mode_rejects_explicit_terrain_observation(backbone_case):
    _, disabled, _, call = backbone_case
    with pytest.raises(ValueError, match="terrain_conditioning=True"):
        call(disabled, _terrain())


def test_terrain_gate_zero_is_exact_parity_and_nonzero_gate_conditions_output(backbone_case):
    _, disabled, enabled, call = backbone_case
    terrain = _terrain()
    reference = _primary(call(disabled))
    gated = _primary(call(enabled, terrain))
    assert torch.equal(reference, gated)
    enabled._terrain_cross_attention.gate.data.fill_(0.7)
    conditioned_output = call(enabled, terrain)
    conditioned = _primary(conditioned_output)
    assert torch.isfinite(conditioned).all()
    assert not torch.allclose(reference, conditioned)
    if "num_token_logits" in conditioned_output:
        assert not torch.allclose(call(disabled)["num_token_logits"], conditioned_output["num_token_logits"])


def test_all_unknown_and_nan_observations_are_finite_and_do_not_change_output(backbone_case):
    _, disabled, enabled, call = backbone_case
    enabled._terrain_cross_attention.gate.data.fill_(0.7)
    terrain = _terrain()
    terrain["points"].data.fill_(float("nan"))
    terrain["valid"].zero_()
    baseline = _primary(call(disabled))
    result = _primary(call(enabled, terrain))
    assert torch.isfinite(result).all()
    assert torch.equal(baseline, result)


def test_terrain_branch_gets_backbone_gradients_but_observation_does_not(backbone_case):
    _, _, enabled, call = backbone_case
    enabled._terrain_cross_attention.gate.data.fill_(0.7)
    terrain = _terrain(requires_grad=True)
    output = _primary(call(enabled, terrain))
    output.square().mean().backward()
    encoder_grads = [p.grad for p in enabled._terrain_encoder.parameters()]
    attention_grads = [p.grad for p in enabled._terrain_cross_attention.attention.parameters()]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
               for grad in encoder_grads)
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
               for grad in attention_grads)
    assert terrain["points"].grad is None


def test_local_terrain_change_changes_conditioned_output(backbone_case):
    _, _, enabled, call = backbone_case
    enabled._terrain_cross_attention.gate.data.fill_(0.7)
    flat = _terrain()
    bump = _terrain()
    bump["points"][0, 0, 4, 4, 1] = 0.8
    flat_result = _primary(call(enabled, flat))
    bump_result = _primary(call(enabled, bump))
    assert not torch.allclose(flat_result, bump_result)


def test_enabled_state_dict_round_trip(backbone_case, tmp_path):
    _, _, enabled, call = backbone_case
    enabled._terrain_cross_attention.gate.data.fill_(0.7)
    terrain = _terrain()
    expected = _primary(call(enabled, terrain))
    path = tmp_path / "terrain_backbone.pt"
    torch.save(enabled.state_dict(), path)
    restored = deepcopy(enabled)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(expected, _primary(call(restored, terrain)))


def test_terrain_observation_shape_validation(backbone_case):
    _, _, enabled, call = backbone_case
    with pytest.raises(ValueError, match="points must have shape"):
        call(enabled, {"points": torch.zeros(1, 9, 9, 3), "valid": torch.ones(1, 9, 9, dtype=torch.bool)})
    with pytest.raises(ValueError, match="valid must be bool"):
        call(enabled, {"points": torch.zeros(1, 2, 9, 9, 3), "valid": torch.ones(1, 2, 9, 9)})
    with pytest.raises(ValueError, match=r"shape \[B, 2, 9, 9, 3\]"):
        call(enabled, {"points": torch.zeros(1, 2, 10, 9, 3), "valid": torch.ones(1, 2, 10, 9, dtype=torch.bool)})


def test_remote_valid_terrain_point_is_rejected(backbone_case):
    _, _, enabled, call = backbone_case
    terrain = _terrain()
    radius = TERRAIN_PROTOCOL['max_horizontal_query_radius_m']
    terrain["points"][0, 0, 4, 4, 0] = radius + 0.01
    with pytest.raises(ValueError, match=f"within {radius} m horizontally"):
        call(enabled, terrain)


def test_pose_root_height_gate_zero_parity_and_height_sensitivity(backbone_case):
    stage, disabled, enabled, _ = backbone_case
    if stage != "pose":
        pytest.skip("Root backbone does not consume the pose-stage root-height bridge.")
    height = torch.zeros(1, 24, 1, requires_grad=True)
    original = _primary(_pose_call(disabled))
    assert torch.equal(original, _primary(_pose_height_call(enabled, height)))
    enabled._terrain_root_height_gate.data.fill_(0.7)
    changed_height = height.clone()
    changed_height[:, 8:12] = 0.15
    changed = _primary(_pose_height_call(enabled, changed_height))
    assert torch.isfinite(changed).all()
    assert not torch.allclose(original, changed)
    changed.square().mean().backward()
    assert height.grad is None
    assert enabled._terrain_root_height_projection.weight.grad is not None


def test_pose_root_height_bridge_validates_mode_shape_and_finiteness(backbone_case):
    stage, disabled, enabled, _ = backbone_case
    if stage != "pose":
        pytest.skip("Root backbone does not consume the pose-stage root-height bridge.")
    with pytest.raises(ValueError, match="requires terrain_conditioning"):
        _pose_height_call(disabled, torch.zeros(1, 24, 1))
    with pytest.raises(ValueError, match="num_positions \\* 4"):
        _pose_height_call(enabled, torch.zeros(1, 23, 1))
    nonfinite = torch.zeros(1, 24, 1)
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite values"):
        _pose_height_call(enabled, nonfinite)
