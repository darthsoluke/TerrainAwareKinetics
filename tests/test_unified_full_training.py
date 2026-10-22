import torch
from torch import nn
import pytest

from motionbricks.data.unified_full_training import NativeTrainingStage, rotate_volume_observation
from motionbricks.data.unified_scene_field import AnalyticPlaneField, VolumetricEgoSampler


def test_cached_quarter_turn_matches_requerying_rotated_world_and_geometry():
    history = torch.zeros(1, 4, 5)
    history[..., 1] = 1
    history[..., 3] = 1
    history[0, :, 2] = torch.arange(4)*.025
    velocity = torch.tensor([[.2, 1.5]])
    sampler = VolumetricEgoSampler()
    before = sampler.observe(history, velocity, AnalyticPlaneField(.1), .933333333)
    rotated = history.clone()
    rotated[..., 0], rotated[..., 2] = history[..., 2], -history[..., 0]
    rotated[..., 3], rotated[..., 4] = 0., 1.
    actual = sampler.observe(rotated, torch.stack((velocity[:, 1], -velocity[:, 0]), -1),
        AnalyticPlaneField(.1), .933333333)
    with pytest.raises(ValueError, match='Non-plane'):
        rotate_volume_observation(before, torch.tensor([1]))
    cached = rotate_volume_observation(before, torch.tensor([1]), field_is_plane=True)
    torch.testing.assert_close(cached['features'], actual['features'], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(cached['distance'], actual['distance'], atol=3e-5, rtol=1e-5)
    assert torch.equal(cached['valid'], actual['valid'])


class StatefulCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer('updates', torch.zeros(1))

    def forward(self, value):
        if self.training:
            self.updates.add_(1)
        return value*self.weight


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.root, self.pose, self.codec = nn.Linear(2, 2), nn.Linear(2, 2), StatefulCodec()
        self.adaptation = nn.ModuleDict({name: nn.Linear(2, 2) for name in
            ('encoder', 'root_stage1', 'root_stage2', 'decoder', 'height_encoder', 'pose')})


def test_pose_training_does_not_update_frozen_codec_buffers():
    controller = Controller()
    stage = NativeTrainingStage(controller, 'pose').train(True)
    controller.codec(torch.ones(2))
    assert controller.codec.updates.item() == 0
    assert not controller.codec.weight.requires_grad
    assert controller.pose.training
    stage.eval()
    assert not controller.pose.training
