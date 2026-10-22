import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_terrain_batch import (
    gather_current_terrain_observation,
    root_height_condition,
)


def make_encoded_terrain(rows=7, frames=20, height=2, width=3, *, requires_grad=False):
    points = torch.zeros(rows, frames, 2, height, width, 3)
    row = torch.arange(rows).view(rows, 1, 1, 1, 1)
    frame = torch.arange(frames).view(1, frames, 1, 1, 1)
    points[..., 0] = row
    points[..., 1] = frame
    points[..., 2] = row * 100 + frame
    points.requires_grad_(requires_grad)
    valid = ((row + frame) % 3 != 0).expand(rows, frames, 2, height, width).clone()
    return points, valid


def test_randomized_source_rows_and_segment_starts_align_at_current_context_frame():
    generator = torch.Generator().manual_seed(914)
    points, valid = make_encoded_terrain()
    batch_size, sampled_frames, context_frames = 19, 8, 4
    chosen_ids = torch.randint(points.shape[0], (batch_size,), generator=generator)
    starts = torch.randint(points.shape[1] - sampled_frames + 1, (batch_size,), generator=generator)
    frame_indices = starts[:, None] + torch.arange(sampled_frames)
    output = gather_current_terrain_observation(
        {"terrain_points": points, "terrain_valid": valid},
        {"chosen_ids": chosen_ids, "frame_indices": frame_indices}, context_frames)
    current = frame_indices[:, context_frames - 1]
    torch.testing.assert_close(output["points"], points[chosen_ids, current])
    torch.testing.assert_close(output["valid"], valid[chosen_ids, current])
    assert output["points"].shape == (batch_size, 2, 2, 3, 3)


def test_future_terrain_changes_do_not_change_current_frame_observation():
    points, valid = make_encoded_terrain(rows=4, frames=16)
    chosen_ids = torch.arange(4)
    starts = torch.tensor([0, 2, 4, 1])
    frame_indices = starts[:, None] + torch.arange(8)
    info = {"chosen_ids": chosen_ids, "frame_indices": frame_indices}
    source_batch = {"terrain_points": points, "terrain_valid": valid}
    before = gather_current_terrain_observation(source_batch, info)

    changed_points, changed_valid = points.clone(), valid.clone()
    for row, start in enumerate(starts.tolist()):
        current_frame = start + 3
        changed_points[row, current_frame + 1:] = -777.0
        changed_valid[row, current_frame + 1:] = ~changed_valid[row, current_frame + 1:]
    after = gather_current_terrain_observation(
        {"terrain_points": changed_points, "terrain_valid": changed_valid}, info)
    torch.testing.assert_close(after["points"], before["points"])
    torch.testing.assert_close(after["valid"], before["valid"])


def test_terrain_labels_are_detached_from_autograd():
    points, valid = make_encoded_terrain(requires_grad=True)
    output = gather_current_terrain_observation(
        {"terrain_points": points, "terrain_valid": valid},
        {"chosen_ids": torch.tensor([2, 0]), "frame_indices": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])},
    )
    assert not output["points"].requires_grad
    assert not output["valid"].requires_grad


def test_context_frame_count_selects_last_context_frame_and_is_configurable():
    points, valid = make_encoded_terrain(rows=2, frames=10)
    info = {"chosen_ids": torch.tensor([1]), "frame_indices": torch.tensor([[4, 5, 6, 7, 8]])}
    batch = {"terrain_points": points, "terrain_valid": valid}
    default = gather_current_terrain_observation(batch, info)
    short_context = gather_current_terrain_observation(batch, info, context_frames=2)
    torch.testing.assert_close(default["points"], points[1:2, 7])
    torch.testing.assert_close(short_context["points"], points[1:2, 5])


@pytest.mark.parametrize("points,valid,info,context_frames", [
    (None, torch.zeros(1, 2, 2, 1, 1, dtype=torch.bool), {}, 4),
    (torch.zeros(1, 2, 2, 1, 1, 3), None, {}, 4),
    (torch.zeros(1, 2, 1, 1, 3), torch.zeros(1, 2, 1, 1, dtype=torch.bool), {}, 4),
    (torch.zeros(1, 2, 2, 1, 1, 3), torch.zeros(1, 2, 2, 1, 1, dtype=torch.bool), {}, 4),
    (torch.zeros(2, 3, 2, 1, 1, 3), torch.zeros(2, 3, 2, 1, 1, dtype=torch.bool),
     {"chosen_ids": torch.tensor([0, 1]), "frame_indices": torch.tensor([[0, 1], [1, 2]])}, 4),
    (torch.zeros(2, 3, 2, 1, 1, 3), torch.zeros(2, 3, 2, 1, 1, dtype=torch.bool),
     {"chosen_ids": torch.tensor([0]), "frame_indices": torch.tensor([[0, 3]])}, 2),
])
def test_missing_or_misaligned_terrain_inputs_raise(points, valid, info, context_frames):
    with pytest.raises(ValueError):
        gather_current_terrain_observation(
            {"terrain_points": points, "terrain_valid": valid}, info, context_frames)


def test_invalid_source_or_frame_indices_raise():
    points, valid = make_encoded_terrain(rows=2, frames=5)
    with pytest.raises(ValueError, match="chosen_ids index"):
        gather_current_terrain_observation(
            {"terrain_points": points, "terrain_valid": valid},
            {"chosen_ids": torch.tensor([2]), "frame_indices": torch.tensor([[0, 1, 2, 3]])})
    with pytest.raises(ValueError, match="frame_indices index"):
        gather_current_terrain_observation(
            {"terrain_points": points, "terrain_valid": valid},
            {"chosen_ids": torch.tensor([0]), "frame_indices": torch.tensor([[0, 1, 2, 5]])})


def test_sample_info_requires_both_alignment_fields():
    points, valid = make_encoded_terrain(rows=1, frames=5)
    with pytest.raises(ValueError, match="missing required fields"):
        gather_current_terrain_observation(
            {"terrain_points": points, "terrain_valid": valid}, {"chosen_ids": torch.tensor([0])})


def test_batch_requires_both_terrain_arrays():
    points, _ = make_encoded_terrain(rows=1, frames=5)
    with pytest.raises(ValueError, match="terrain_valid"):
        gather_current_terrain_observation(
            {"terrain_points": points},
            {"chosen_ids": torch.tensor([0]), "frame_indices": torch.tensor([[0, 1, 2, 3]])})


class FakeGlobalRep:
    indices = {"global_root_pos": np.array([0, 1, 2])}

    def __init__(self):
        self.unnormalize_calls = 0

    def unnormalize(self, values):
        self.unnormalize_calls += 1
        return values * 2.0 + 10.0


@pytest.mark.parametrize(("frames", "features"), [(366, 8), (64, 5)])
def test_root_height_condition_handles_training_and_five_feature_inference(frames, features):
    rep = FakeGlobalRep()
    raw = torch.zeros(2, frames, features)
    raw[..., 1] = torch.arange(frames)[None] + torch.tensor([[3.0], [-2.0]])
    normalized = ((raw - 10.0) / 2.0).requires_grad_()
    result = root_height_condition(normalized, rep)
    assert result.shape == (2, frames, 1)
    torch.testing.assert_close(result[..., 0], raw[..., 1] - raw[:, 3:4, 1])
    assert torch.equal(result[:, 3], torch.zeros(2, 1))
    assert rep.unnormalize_calls == 1
    assert not result.requires_grad


def test_root_height_condition_can_accept_raw_features_without_unnormalizing():
    rep = FakeGlobalRep()
    raw = torch.randn(3, 64, 5)
    result = root_height_condition(raw, rep, normalized=False)
    torch.testing.assert_close(result[..., 0], raw[..., 1] - raw[:, 3:4, 1])
    assert rep.unnormalize_calls == 0


def test_root_height_condition_rejects_short_or_malformed_inputs():
    rep = FakeGlobalRep()
    with pytest.raises(ValueError, match="at least context_frames"):
        root_height_condition(torch.zeros(2, 3, 8), rep)
    with pytest.raises(ValueError, match="global_root must have shape"):
        root_height_condition(torch.zeros(2, 8), rep)
