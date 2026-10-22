"""CPU physical parity of selected vertices against the actual official skin."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from motionbricks.scene_control_v2.provider import load_bound_world
from motionbricks.data.soma_skin_reference import move_skinner
from motionbricks.scene_control_v3.native import NativeSOMAAdapter
from motionbricks.scene_control_v3.surface import OfficialSurfaceSampler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def cpu_two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source_skin():
    config = json.loads((ROOT / "configs/scene_control_v3_performance.json").read_text())
    adapter = NativeSOMAAdapter(ROOT / config["assets"]["bundle"]["path"], device="cpu")
    spec = json.loads((ROOT / "configs/scene_control_v2_real_provider_cpu.json").read_text())["provider"]
    with torch.inference_mode():
        world, _, _, _, _ = load_bound_world(spec, adapter.representation)
        assert world.shape == (646, 366)
        rig = adapter.source_frame(world[[0, 315, 645]][None])["rig"]
        transform = torch.eye(4, dtype=rig.dtype)
        angle = rig.new_tensor(0.47)
        transform[0, 0] = transform[2, 2] = angle.cos()
        transform[0, 2] = angle.sin()
        transform[2, 0] = -angle.sin()
        transform[:3, 3] = rig.new_tensor([12.75, 3.2, -8.125])
        rigs = torch.cat((rig, transform @ rig), dim=0)
        full_vertices = adapter.fullskin(rigs)
    return adapter.skinner, rigs, full_vertices


def assert_metric_parity(actual, expected):
    assert torch.isfinite(actual).all()
    error_m = torch.linalg.vector_norm(actual - expected, dim=-1).max()
    assert error_m <= 3e-5, f"Official surface error {error_m} m exceeds 3e-5 m"


@pytest.mark.parametrize("sample_count", [0, -1, 18057, 1.5, True])
def test_constructor_rejects_invalid_sample_count(source_skin, sample_count):
    skinner, _, _ = source_skin
    with pytest.raises(ValueError, match="sample_count"):
        OfficialSurfaceSampler(skinner, sample_count)


@pytest.mark.parametrize("case, message", [
    ("rank", "one-dimensional"), ("count", "sample_count entries"),
    ("negative", "range"), ("upper_bound", "range"),
    ("duplicate", "unique"), ("floating", "integer indices"),
])
def test_constructor_rejects_invalid_explicit_selection(source_skin, case, message):
    skinner, _, _ = source_skin
    ids = torch.arange(512)
    if case == "rank":
        ids = ids.reshape(16, 32)
    elif case == "count":
        ids = ids[:-1]
    elif case == "negative":
        ids[-1] = -1
    elif case == "upper_bound":
        ids[-1] = len(skinner.bind_vertices)
    elif case == "duplicate":
        ids[-1] = ids[0]
    elif case == "floating":
        ids = ids.float() + 0.5
    with pytest.raises(ValueError, match=message):
        OfficialSurfaceSampler(skinner, selected_vertex_ids=ids)


@torch.inference_mode()
def test_real_source_0_315_645_and_nonorigin_rigs_match_full_official_skin(source_skin):
    skinner, rigs, full_vertices = source_skin
    sampler = OfficialSurfaceSampler(skinner)
    assert sampler.selection_kind == "uniform_vertex_index_capacity_selection"
    assert sampler.complete_contact_layout is False
    ids = sampler.selected_vertex_ids
    assert ids.shape == (512,) and ids.dtype == torch.int64 and ids.unique().numel() == 512
    assert ids[0] == 0 and ids[-1] == 18055
    assert (rigs[1, ..., :3, 3] - rigs[0, ..., :3, 3]).abs().max() > 3.0
    points = sampler(rigs)
    assert points.shape == (2, 3, 512, 3)
    assert_metric_parity(points, full_vertices.index_select(-2, ids))


@torch.inference_mode()
def test_explicit_vertices_keep_all_official_influences_including_beyond_four(source_skin):
    skinner, rigs, full_vertices = source_skin
    assert skinner.lbs_weights.shape[-1] > 4
    tail_weight = skinner.lbs_weights[:, 4:].sum(-1)
    ids = tail_weight.argsort(descending=True)[:512]
    sampler = OfficialSurfaceSampler(skinner, 512, selected_vertex_ids=ids)
    assert sampler.selection_kind == "explicit_vertex_ids_capacity_selection"
    torch.testing.assert_close(sampler.selected_vertex_ids, ids, rtol=0, atol=0)
    torch.testing.assert_close(sampler.lbs_indices, skinner.lbs_indices.index_select(0, ids), rtol=0, atol=0)
    torch.testing.assert_close(sampler.lbs_weights, skinner.lbs_weights.index_select(0, ids), rtol=0, atol=0)
    points = sampler(rigs)
    assert_metric_parity(points, full_vertices.index_select(-2, ids))
    truncated = SimpleNamespace(bind_vertices=sampler.bind_vertices,
                               lbs_indices=sampler.lbs_indices[:, :4],
                               lbs_weights=sampler.lbs_weights[:, :4],
                               bind_rig_transform_inv=sampler.bind_rig_transform_inv)
    truncated_points = type(skinner).lbs(truncated, rigs)
    assert torch.linalg.vector_norm(points - truncated_points, dim=-1).max() > 3e-5


@torch.inference_mode()
def test_forward_uses_only_selected_buffers_and_does_not_mutate_inputs_or_skin(source_skin):
    skinner, rigs, full_vertices = source_skin
    sampler = OfficialSurfaceSampler(skinner)
    names = ("bind_vertices", "lbs_indices", "lbs_weights", "bind_rig_transform_inv")
    original_skin = {name: getattr(skinner, name).clone() for name in names}
    original_buffers = {name: value.clone() for name, value in sampler.named_buffers()}
    rig_before = rigs.clone()
    observed = []
    official_lbs = sampler._official_lbs

    def selected_lbs(receiver, rig):
        observed.append((receiver.bind_vertices.shape, receiver.lbs_indices.shape))
        return official_lbs(receiver, rig)

    sampler._official_lbs = selected_lbs
    first = sampler(rigs)
    first.zero_()
    second = sampler(rigs)
    assert observed == [(torch.Size([512, 3]), torch.Size([512, skinner.lbs_indices.shape[1]]))] * 2
    assert_metric_parity(second, full_vertices.index_select(-2, sampler.selected_vertex_ids))
    torch.testing.assert_close(rigs, rig_before, rtol=0, atol=0)
    for name, value in sampler.named_buffers():
        torch.testing.assert_close(value, original_buffers[name], rtol=0, atol=0)
    for name in names:
        torch.testing.assert_close(getattr(skinner, name), original_skin[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@torch.inference_mode()
def test_zero_parameter_frozen_buffers_follow_cpu_dtype_and_device(source_skin, dtype):
    skinner, rigs, _ = source_skin
    sampler = OfficialSurfaceSampler(skinner).to(device="cpu", dtype=dtype)
    assert list(sampler.parameters()) == []
    assert dict(sampler.named_buffers()).keys() == {
        "selected_vertex_ids", "bind_vertices", "lbs_indices", "lbs_weights", "bind_rig_transform_inv",
    }
    assert sampler.bind_vertices.shape == (512, 3)
    assert sampler.bind_rig_transform_inv.shape == (77, 4, 4)
    for name, value in sampler.named_buffers():
        assert value.device.type == "cpu" and not value.requires_grad
        assert value.dtype == (torch.int64 if name in ("selected_vertex_ids", "lbs_indices") else dtype)
    typed_rig = rigs.to(dtype)
    points = sampler(typed_rig)
    typed_skin = move_skinner(copy.copy(skinner), "cpu", dtype)
    expected = type(skinner).lbs(typed_skin, typed_rig).index_select(-2, sampler.selected_vertex_ids)
    assert points.device == typed_rig.device and points.dtype == dtype
    assert_metric_parity(points, expected)
    assert sum(value.numel() for value in sampler.buffers()) < 18056
