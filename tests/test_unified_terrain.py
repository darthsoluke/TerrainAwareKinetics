import pytest
import torch

from motionbricks.data.unified_scene_field import (
    AnalyticBoxField, AnalyticPlaneField, VolumeSampling, VolumetricEgoSampler, field_from_heightfield,
)
from motionbricks.data.unified_terrain import SharedTerrainEncoder, UnifiedSceneResidual, UnifiedTerrainController
from motionbricks.data.unified_terrain import sdf_observation_debug
from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.soma_terrain_execution import TerrainExecution
from motionbricks.data.native_preserving_terrain import PROFILE


def history():
    value = torch.zeros(1, 4, 5)
    value[..., 1] = 1
    value[..., 3] = 1
    return value


def test_plane_normal_is_consistent_at_and_near_surface():
    points = torch.tensor([[0., 0., 0.], [0., 5e-9, 0.], [0., -5e-9, 0.]])
    distance, normal, valid = AnalyticPlaneField(0).query(points)
    torch.testing.assert_close(distance, points[:, 1], atol=0, rtol=0)
    assert normal[:, 1].eq(1).all() and valid.all()


def test_known_bounds_do_not_become_infinite_flat_space():
    field = AnalyticPlaneField(0, (-1., 1., -1., 1.))
    _, normal, valid = field.query(torch.tensor([[0., .1, 0.], [2., .1, 0.], [.99, 1., 0.]]))
    assert valid.tolist() == [True, False, False]
    assert not normal[1:].any()


def test_only_complete_uniform_heightfield_selects_plane_backend():
    scene = dict(heightfield_y_m=torch.zeros(9, 9), min_point_world_xyz_m=torch.tensor([-4., 0., -4.]), dx_m=1.)
    assert isinstance(field_from_heightfield(scene), AnalyticPlaneField)
    scene['heightfield_valid'] = torch.ones(9, 9, dtype=torch.bool)
    scene['heightfield_valid'][4, 4] = False
    assert not isinstance(field_from_heightfield(scene), AnalyticPlaneField)
    _, _, valid = field_from_heightfield(scene).query(torch.tensor([[0., 0., 0.]]))
    assert not valid.item()


def test_reversal_keeps_both_requested_and_actual_directions():
    h = history()
    h[0, :, 2] = torch.arange(4)*.05
    points = VolumetricEgoSampler().points(h, torch.tensor([[0., -1.5]]), .8)
    current = h[:, 3, 2]
    assert points[:, 1, :, 2].min() < current-1
    assert points[:, 2, :, 2].max() > current+1
    assert points.shape == (1, 3, 512, 3)


def test_zero_input_and_stationary_history_are_finite_and_deterministic():
    sampler = VolumetricEgoSampler()
    a = sampler.points(history(), torch.zeros(1, 2), .8)
    b = sampler.points(history(), torch.zeros(1, 2), .8)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.isfinite(a).all()
    assert a[..., 1].min() < 0 and a[..., 1].max() > 2


def test_sampling_covaries_with_world_yaw_and_translation():
    sampler = VolumetricEgoSampler()
    h = history()
    h[0, :, 2] = torch.arange(4)*.025
    v = torch.tensor([[.2, 1.5]])
    p = sampler.points(h, v, .8)
    for yaw in (.7, 1.57079632679, 3.14159265359):
        c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
        matrix = torch.stack((torch.stack((c, s)), torch.stack((-s, c))))
        changed = h.clone()
        changed[..., [0, 2]] = h[..., [0, 2]]@matrix.T+torch.tensor([15., -8.])
        changed[..., 1] += 2
        changed[..., 3], changed[..., 4] = c, s
        actual = sampler.points(changed, v@matrix.T, .8)
        expected = p.clone()
        expected[..., [0, 2]] = p[..., [0, 2]]@matrix.T+torch.tensor([15., -8.])
        expected[..., 1] += 2
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize('center,half', [([0, 1, 1], [.8, 2, .25]), ([0, 2, .5], [1, .2, 1])])
def test_volume_sees_wall_and_ceiling_not_only_ground(center, half):
    observation = VolumetricEgoSampler().observe(history(), torch.tensor([[0., 1.6]]),
                                                AnalyticBoxField(center, half), .8)
    assert observation['distance'].min() < 0
    assert observation['points'][..., 1].max() > 2
    assert observation['features'].shape[-1] == 15


def test_all_unknown_is_an_error_not_flat_fallback():
    field = AnalyticPlaneField(0, (100., 101., 100., 101.))
    with pytest.raises(ValueError, match='unknown'):
        VolumetricEgoSampler().observe(history(), torch.zeros(1, 2), field, .8)


def test_encoder_keeps_fixed_spatial_blocks_and_masks_unknown():
    sampler = VolumetricEgoSampler()
    observation = sampler.observe(history(), torch.zeros(1, 2), AnalyticPlaneField(0), .8)
    observation['valid'][:, 2] = False
    observation['features'][:, 2] = 0
    encoder = SharedTerrainEncoder(8)
    tokens, valid = encoder(observation)
    assert tokens.shape == (1, 24, 128)
    assert valid[:, :16].all() and not valid[:, 16:].any()
    assert not tokens[:, 16:].any() and torch.isfinite(tokens).all()
    tokens.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_zero_residual_does_not_add_time_to_original_latent():
    module = UnifiedSceneResidual(32, time_query=True)
    value = torch.randn(1, 16, 32)
    memory = torch.randn(1, 24, 128)
    time = torch.arange(16)[None].float()/30
    output = module(value, memory, torch.ones(1, 24, dtype=torch.bool), time)
    torch.testing.assert_close(output, value, atol=0, rtol=0)


def test_flat_environment_receives_gradients_after_zero_output_wakes():
    torch.manual_seed(71)
    sampler = VolumetricEgoSampler()
    encoder = SharedTerrainEncoder(8)
    module = UnifiedSceneResidual(32)
    value = torch.randn(1, 8, 32)
    observation = sampler.observe(history(), torch.tensor([[0., 1.6]]), AnalyticPlaneField(0), .8)
    optimizer = torch.optim.Adam([*encoder.parameters(), *module.parameters()], lr=5e-5)
    gradients = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        memory, valid = encoder(observation)
        loss = module(value, memory, valid).square().mean()
        loss.backward()
        gradients.append(sum(float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None))
        assert module.output.weight.grad.norm() > 0
        optimizer.step()
    assert gradients[0] == 0 and gradients[1] > 0


def test_decoder_time_query_can_distinguish_identical_codes():
    torch.manual_seed(9)
    module = UnifiedSceneResidual(32, time_query=True)
    torch.nn.init.normal_(module.output.weight, std=.05)
    value = torch.ones(1, 16, 32)
    output = module(value, torch.randn(1, 24, 128), torch.ones(1, 24, dtype=torch.bool), torch.arange(16)[None].float()/8)
    assert (output[:, 0]-output[:, -1]).abs().max() > 1e-5


def test_off_source_recovery_targets_are_not_silently_invented():
    source = torch.zeros(1, 33, 5)
    played = source[:, :4].clone()
    played[:, 3, 0] = .01
    with pytest.raises(ValueError, match='off-source'):
        UnifiedTerrainController.native_losses(None, {}, source, played)


def test_sampler_rejects_nonfinite_or_invalid_shapes():
    with pytest.raises(ValueError):
        VolumeSampling(grid_size=7)
    with pytest.raises(ValueError):
        VolumetricEgoSampler().points(history()[:, :3], torch.zeros(1, 2), .8)
    with pytest.raises(ValueError):
        VolumetricEgoSampler().points(history(), torch.tensor([[float('nan'), 0.]]), .8)


def test_debug_reports_exact_model_samples_not_a_new_observation():
    observation = VolumetricEgoSampler().observe(history(), torch.tensor([[0., 1.6]]), AnalyticPlaneField(0), .8)
    debug = sdf_observation_debug(observation['points'], observation['distance'], observation['valid'], history())
    assert debug['role'] == 'model_input' and debug['points_per_region'] == 512
    torch.testing.assert_close(torch.tensor(debug['world_xyz_m']), observation['points'][0].reshape(-1, 3))
    torch.testing.assert_close(torch.tensor(debug['signed_distance_m']), observation['distance'][0].reshape(-1))


def test_buffered_diagnostics_keep_scene_snapshot_and_advance_its_age():
    points = [[1., 2., 3.]]
    debug = dict(unified_sdf=dict(world_xyz_m=points), decoded_world_xyz_m=points*24)
    second = NativeTerrainOnlineBackend._advance_debug(debug, 8)
    third = NativeTerrainOnlineBackend._advance_debug(second, 8)
    assert second['unified_sdf']['world_xyz_m'] is points
    assert third['observation_lag_frames'] == 16
    assert 'observation_lag_frames' not in debug


def test_capture_rejects_unverified_controls_but_allows_reuse_interval():
    from types import SimpleNamespace
    execution = SimpleNamespace(controller=object())
    assert TerrainExecution.supports(execution, dict(PROFILE))
    assert TerrainExecution.supports(execution, dict(PROFILE, refresh_frames=8))
    assert not TerrainExecution.supports(execution, dict(PROFILE, inference_steps=2))
    assert not TerrainExecution.supports(execution, dict(PROFILE, num_tokens=16))
