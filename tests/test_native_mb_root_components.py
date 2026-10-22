from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOTIONBRICKS_ROOT))
sys.path.insert(0, str(MOTIONBRICKS_ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_models import _resolved_backbone_args, native_config
from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.vqvae.neural_modules.encdec_double_cond import DoubleCondDecoder


def _motion_rep():
    return GlobalRootGlobalJoints(30.0, SOMASkeleton30(), "somaskel30_dual_root_global_joints")


@pytest.fixture
def root_backbones():
    rep = _motion_rep()
    config = native_config("root", max_steps=100)
    args = _resolved_backbone_args("root", config)
    args.update(n_embd=32, n_head=4, n_layers_shared=1, n_layers_root_token=1,
                width=32, depth=1, terrain_conditioning=False)

    torch.manual_seed(231)
    legacy = root_backbone_network(deepcopy(args), rep).eval()
    component_args = deepcopy(args)
    component_args["root_component_conditioning"] = True
    component = root_backbone_network(deepcopy(component_args), rep).eval()
    missing, unexpected = component.load_state_dict(legacy.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all("component_mask" in key for key in missing)
    return legacy, component


def _call_root(model, values=None, component_mask=None, has_values=None,
               num_tokens=None, groundtruth_num_tokens=None):
    batch_size = 1 if values is None else values.shape[0]
    if values is None:
        values = torch.zeros(batch_size, 8, 5)
    if has_values is None:
        has_values = torch.ones(batch_size, 8, dtype=torch.bool)
    if num_tokens is None:
        num_tokens = torch.full((batch_size, 1), 6, dtype=torch.long)
    if groundtruth_num_tokens is None:
        groundtruth_num_tokens = num_tokens

    kwargs = {}
    if component_mask is not None:
        kwargs["global_root_component_mask"] = component_mask
    return model(
        global_root_values=values,
        has_global_root_values=has_values,
        local_root_values=torch.zeros(batch_size, 8, 4),
        has_local_root_values=torch.ones(batch_size, 8, dtype=torch.bool),
        poses=torch.zeros(batch_size, 8, 268),
        has_poses=torch.ones(batch_size, 8, dtype=torch.bool),
        num_tokens=num_tokens,
        groundtruth_num_tokens=groundtruth_num_tokens,
        **kwargs,
    )


def test_legacy_and_full_component_mask_have_exact_parity(root_backbones):
    legacy, component = root_backbones
    values = torch.randn(1, 8, 5)
    all_known = torch.ones(1, 8, 5, dtype=torch.bool)

    expected = _call_root(legacy, values)
    absent = _call_root(component, values)
    explicit_full = _call_root(component, values, all_known)

    assert torch.equal(expected["pred_global_root_values"], absent["pred_global_root_values"])
    assert torch.equal(expected["num_token_logits"], absent["num_token_logits"])
    assert torch.equal(expected["pred_global_root_values"], explicit_full["pred_global_root_values"])
    assert torch.equal(expected["num_token_logits"], explicit_full["num_token_logits"])


def test_unknown_height_is_sanitized_and_known_height_conditions_output(root_backbones):
    _, component = root_backbones
    values = torch.randn(1, 8, 5)
    height_unknown = torch.ones(1, 8, 5, dtype=torch.bool)
    height_unknown[:, :, 1] = False
    has_values = height_unknown.any(dim=-1)

    finite_values = values.clone()
    finite_values[:, :, 1] = 123.0
    nan_values = finite_values.clone()
    nan_values[:, :, 1] = float("nan")
    finite_output = _call_root(component, finite_values, height_unknown, has_values)
    nan_output = _call_root(component, nan_values, height_unknown, has_values)
    assert torch.isfinite(nan_output["pred_global_root_values"]).all()
    assert torch.equal(finite_output["pred_global_root_values"], nan_output["pred_global_root_values"])

    fully_known = torch.ones_like(height_unknown)
    known_low = values.clone()
    known_high = values.clone()
    known_low[:, :, 1] = 0.0
    known_high[:, :, 1] = 1.0
    low_output = _call_root(component, known_low, fully_known)
    high_output = _call_root(component, known_high, fully_known)
    assert not torch.allclose(low_output["pred_global_root_values"], high_output["pred_global_root_values"])


def test_component_mask_scatter_tracks_variable_duration_end_slots(root_backbones):
    _, component = root_backbones
    component_mask = torch.ones(2, 8, 5, dtype=torch.bool)
    component_mask[:, :, 1] = False
    component_mask[0, 1, :] = False
    component_mask[1, 6, :] = False
    has_values = component_mask.any(dim=-1)
    values = torch.full((2, 8, 5), 19.0)
    captured = {}

    def capture_decoder_inputs(_module, _args, kwargs):
        captured.update(kwargs)

    handle = component._conv_output.register_forward_pre_hook(capture_decoder_inputs, with_kwargs=True)
    try:
        tokens = torch.tensor([[6], [8]])
        output = _call_root(component, values, component_mask, has_values,
                            num_tokens=tokens, groundtruth_num_tokens=tokens)
    finally:
        handle.remove()

    dense_mask = captured["target_component_mask"]
    expected = torch.zeros_like(dense_mask)
    expected[:, :4] = component_mask[:, :4]
    expected[0, 20:24] = component_mask[0, 4:8]
    expected[1, 28:32] = component_mask[1, 4:8]
    assert torch.equal(dense_mask, expected)
    assert torch.equal(captured["has_target_cond"], expected.any(dim=-1))
    assert torch.equal(captured["target_cond"][~dense_mask], torch.zeros_like(captured["target_cond"][~dense_mask]))
    assert output["pred_global_root_values"].shape == (2, 64, 5)
    assert not torch.equal(output["pred_global_root_values"][:, :4], values[:, :4])


def test_component_mask_embeddings_receive_gradients(root_backbones):
    _, component = root_backbones
    component.train()
    values = torch.randn(1, 8, 5)
    component_mask = torch.ones(1, 8, 5, dtype=torch.bool)
    component_mask[:, :, 1] = False
    output = _call_root(component, values, component_mask, component_mask.any(dim=-1))
    output["pred_global_root_values"].square().mean().backward()

    root_grad = component._proj_global_root_component_mask.weight.grad
    decoder_grads = [block.weight.grad for block in component._conv_output.target_component_mask_blocks]
    assert root_grad is not None and torch.isfinite(root_grad).all() and root_grad.abs().sum() > 0
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
               for grad in decoder_grads)


@pytest.mark.parametrize("invalid_mask, message", [
    (torch.ones(1, 8, 4, dtype=torch.bool), "shape"),
    (torch.ones(1, 8, 5), "bool dtype"),
])
def test_component_mask_rejects_shape_and_dtype_mismatch(root_backbones, invalid_mask, message):
    _, component = root_backbones
    with pytest.raises(ValueError, match=message):
        _call_root(component, component_mask=invalid_mask)


def test_component_mask_rejects_presence_mismatch_and_disabled_mode(root_backbones):
    legacy, component = root_backbones
    mask = torch.ones(1, 8, 5, dtype=torch.bool)
    inconsistent_presence = torch.zeros(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="must equal.*any"):
        _call_root(component, component_mask=mask, has_values=inconsistent_presence)
    with pytest.raises(ValueError, match="root_component_conditioning=True"):
        _call_root(legacy, component_mask=mask)


def test_double_cond_decoder_sanitizes_unknown_targets_and_trains_mask_embeddings():
    torch.manual_seed(19)
    decoder = DoubleCondDecoder(
        input_emb_width=5, output_emb_width=8, down_t=2, width=16, depth=1,
        target_cond_dim=5, external_cond_dim=4, target_component_conditioning=True,
    )
    x = torch.randn(1, 8, 4)
    external = torch.randn(1, 16, 4)
    target = torch.randn(1, 16, 5)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    target_mask[:, :, 1] = False
    has_target = target_mask.any(dim=-1)
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    sanitized_target = torch.where(target_mask, target, torch.zeros_like(target))
    expected = decoder(x, external, sanitized_target, has_target, token_mask)

    unknown_target = target.clone()
    unknown_target[:, :, 1] = float("nan")
    actual = decoder(x, external, unknown_target, has_target, token_mask,
                     target_component_mask=target_mask)
    assert torch.isfinite(actual).all()
    assert torch.equal(expected, actual)

    actual.square().mean().backward()
    grads = [block.weight.grad for block in decoder.target_component_mask_blocks]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_double_cond_decoder_rejects_masks_when_disabled():
    decoder = DoubleCondDecoder(
        input_emb_width=5, output_emb_width=8, down_t=2, width=16, depth=1,
        target_cond_dim=5, external_cond_dim=4,
    )
    x = torch.zeros(1, 8, 4)
    external = torch.zeros(1, 16, 4)
    target = torch.zeros(1, 16, 5)
    has_target = torch.ones(1, 16, dtype=torch.bool)
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="target_component_conditioning=True"):
        decoder(x, external, target, has_target, token_mask,
                target_component_mask=torch.ones_like(target, dtype=torch.bool))


def test_double_cond_decoder_validates_component_mask_contract():
    decoder = DoubleCondDecoder(
        input_emb_width=5, output_emb_width=8, down_t=2, width=16, depth=1,
        target_cond_dim=5, external_cond_dim=4, target_component_conditioning=True,
    )
    x = torch.zeros(1, 8, 4)
    external = torch.zeros(1, 16, 4)
    target = torch.zeros(1, 16, 5)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    has_target = torch.ones(1, 16, dtype=torch.bool)
    token_mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="same shape as target_cond"):
        decoder(x, external, target, has_target, token_mask,
                target_component_mask=torch.ones(1, 16, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool dtype"):
        decoder(x, external, target, has_target, token_mask,
                target_component_mask=target_mask.float())
    with pytest.raises(ValueError, match="must equal.*any"):
        decoder(x, external, target, torch.zeros_like(has_target), token_mask,
                target_component_mask=target_mask)
