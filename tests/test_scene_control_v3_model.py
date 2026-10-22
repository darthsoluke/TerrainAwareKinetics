from collections import Counter
from dataclasses import replace

import pytest
import torch

from motionbricks.scene_control_v3 import (
    CONTACT_FEATURE_LAYOUT, SCENE_FEATURE_LAYOUT, CausalSceneActor, SceneActorConfig,
)


@pytest.fixture(autouse=True)
def cpu_two_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def inputs(model, batch_size=2):
    config = model.config
    return {
        "history": torch.randn(batch_size, config.history_frames, config.history_dim),
        "command": torch.randn(batch_size, config.command_dim),
        "contact_state": torch.randn(batch_size, config.contact_slots, config.contact_dim),
        "contact_valid": torch.ones(batch_size, config.contact_slots, dtype=torch.bool),
        "scene_features": torch.randn(batch_size, config.scene_slots, config.scene_dim),
        "scene_valid": torch.ones(batch_size, config.scene_slots, dtype=torch.bool),
        "hidden": model.init_hidden(batch_size),
    }


def actor():
    torch.manual_seed(1301)
    return CausalSceneActor().eval()


def test_default_tensor_contract_and_probe_identity():
    model = actor()
    assert model.probe_status == "untrained_capacity_probe"
    assert model.capacity_scope == "architecture_capacity_only"
    assert len(SCENE_FEATURE_LAYOUT) == 16
    assert len(CONTACT_FEATURE_LAYOUT) == 16
    data = inputs(model)
    with torch.inference_mode():
        result = model(**data)
    assert {key: tuple(value.shape) for key, value in result.items()} == {
        "root_delta": (2, 3), "heading_delta": (2, 1), "global_joint_rot6d": (2, 30, 6),
        "contact_logits": (2, 15), "release_logits": (2, 15), "root_preview": (2, 16, 3),
        "hidden": (2, 256), "input_valid": (2,),
    }
    assert all(value.device.type == "cpu" and torch.isfinite(value).all() for value in result.values())
    assert result["input_valid"].dtype == torch.bool and result["input_valid"].all()
    assert not data["hidden"].any()
    double_model = model.double()
    assert double_model.init_hidden(3).dtype == torch.float64
    assert double_model.init_hidden(1, dtype=torch.float32).dtype == torch.float32


def test_invalid_nan_slots_are_removed_before_computation_and_inputs_unchanged():
    model = actor()
    data = inputs(model, 3)
    data["scene_valid"][0] = False
    data["scene_valid"][1, ::2] = False
    data["contact_valid"][0] = False
    data["contact_valid"][1, ::2] = False
    data["scene_features"][~data["scene_valid"]] = torch.nan
    data["contact_state"][~data["contact_valid"]] = torch.nan
    before = {key: value.clone() for key, value in data.items()}
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    clean = {key: value.clone() for key, value in data.items()}
    clean["scene_features"][~clean["scene_valid"]] = 123456.0
    clean["contact_state"][~clean["contact_valid"]] = -123456.0
    with torch.inference_mode():
        masked_result = model(**data)
        replaced_result = model(**clean)
    for key in masked_result:
        assert torch.isfinite(masked_result[key]).all()
        torch.testing.assert_close(masked_result[key], replaced_result[key], rtol=0, atol=0)
    torch.testing.assert_close(masked_result["input_valid"], torch.tensor([False, True, True]))
    for key in data:
        torch.testing.assert_close(data[key], before[key], equal_nan=True, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)


def test_unknown_scene_has_explicit_tokens_not_flat_geometry():
    model = actor()
    data = inputs(model)
    data["scene_features"].zero_()
    data["scene_valid"][0] = False
    data["scene_features"][0] = torch.nan
    with torch.inference_mode():
        tokens, coverage = model.scene_encoder(data["scene_features"], data["scene_valid"])
    torch.testing.assert_close(tokens[0], model.scene_encoder.unknown_tokens, rtol=0, atol=0)
    torch.testing.assert_close(coverage, torch.tensor([[0.0], [1.0]]), rtol=0, atol=0)
    assert not torch.allclose(tokens[0], tokens[1])


def test_signed_distance_semantics_reach_backbone_without_legacy_15d_path():
    model = actor()
    assert model.config.scene_dim == 16 and SCENE_FEATURE_LAYOUT[15] == "distance_signed_flag"
    data = inputs(model, 1)
    data["scene_features"].zero_()
    signed = data["scene_features"].clone()
    signed[..., 15] = 1.0
    with torch.inference_mode():
        unsigned_result = model(**data)
        signed_result = model(**dict(data, scene_features=signed))
        with pytest.raises(RuntimeError, match="shapes"):
            model(**dict(data, scene_features=signed[..., :15]))
    assert not torch.allclose(unsigned_result["hidden"], signed_result["hidden"])


def test_commands_and_observed_history_reach_shared_causal_state():
    model = actor()
    data = inputs(model)
    data["command"].zero_()
    with torch.inference_mode():
        baseline = model(**data)
        command_result = model(**dict(data, command=torch.tensor([[2.0, -1.0, 0.5]]).expand(2, -1)))
        history_result = model(**dict(data, history=data["history"].flip(1)))
        next_result = model(**dict(data, hidden=baseline["hidden"]))
        replay = model(**data)
    for key in ("root_delta", "global_joint_rot6d", "contact_logits", "hidden"):
        assert not torch.allclose(baseline[key], command_result[key])
        assert not torch.allclose(baseline[key], history_result[key])
        assert not torch.allclose(baseline[key], next_result[key])
        torch.testing.assert_close(baseline[key], replay[key], rtol=0, atol=0)


@pytest.mark.parametrize("horizon", [1, 4, 8, 12, 16, 32, 64])
def test_all_preview_horizons_reuse_backbone_once_and_preserve_common_prefix(horizon):
    reference = actor()
    model = CausalSceneActor(replace(SceneActorConfig(), preview_frames=horizon)).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    assert sum(value.numel() for value in model.parameters()) == sum(
        value.numel() for value in reference.parameters())
    counts = Counter()

    def count(name):
        def hook(*_):
            counts[name] += 1
        return hook

    handles = [module.register_forward_hook(count(name)) for name, module in (
        ("scene", model.scene_encoder), ("backbone", model.backbone), ("preview", model.preview_head),
    )]
    data = inputs(reference)
    with torch.inference_mode():
        result = model(**data)
        expected = reference(**data)
    for handle in handles:
        handle.remove()
    assert counts == {"scene": 1, "backbone": 1, "preview": 1}
    assert result["root_preview"].shape == (2, horizon, 3)
    common = min(horizon, reference.config.preview_frames)
    torch.testing.assert_close(result["root_preview"][:, :common], expected["root_preview"][:, :common])
    for key in result.keys() - {"root_preview"}:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)


def test_recurrent_memory_stays_fixed_and_bounded_over_rollout():
    model = actor()
    data = inputs(model, 1)
    initial_buffers = {name: value.clone() for name, value in model.named_buffers()}
    with torch.inference_mode():
        for _ in range(120):
            output = model(**data)
            data["hidden"] = output["hidden"]
            assert data["hidden"].shape == (1, 256)
            assert torch.isfinite(data["hidden"]).all() and data["hidden"].abs().max() <= 1.0
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, initial_buffers[name], rtol=0, atol=0)


@pytest.mark.parametrize("hidden_dim", [256, 512, 1024])
def test_capacity_profiles_keep_same_tensor_api_and_dynamic_hidden_width(hidden_dim):
    model = CausalSceneActor(SceneActorConfig(hidden_dim=hidden_dim)).eval()
    data = inputs(model, 1)
    with torch.inference_mode():
        result = model(**data)
    assert data["hidden"].shape == result["hidden"].shape == (1, hidden_dim)
    assert result["global_joint_rot6d"].shape == (1, 30, 6)
    assert result["root_preview"].shape == (1, 16, 3)
    assert result["input_valid"].dtype == torch.bool
    assert torch.isfinite(result["hidden"]).all() and result["hidden"].abs().max() <= 1.0


def test_fixed_tensor_graph_exports_without_python_scalar_admission():
    model = actor()
    data = inputs(model, 1)
    exported = torch.export.export(model, (), kwargs=data, strict=True)
    with torch.inference_mode():
        expected = model(**data)
        actual = exported.module()(**data)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
