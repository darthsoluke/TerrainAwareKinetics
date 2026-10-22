from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY
from motionbricks.data.reactive_prefix_training import (
    LOSS_WEIGHTS, POSITION_SCALE_M, VELOCITY_SCALE_MPS,
    TrainingRootPathObservation, configure_scope, prefix_command, prefix_loss,
)
from motionbricks.data.root_path_terrain import PredictedRootObservation
from motionbricks.data.unified_scene_field import AnalyticPlaneField, VolumeSampling, VolumetricEgoSampler
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Prefix helper tests must remain CPU-only')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)


@pytest.fixture(scope='module')
def source_pose():
    rep = GlobalRootGlobalJoints(30., SOMASkeleton30(), 'somaskel30_dual_root_global_joints')
    time = torch.arange(33, dtype=torch.float32)
    angle = time * .01
    c, s, zero, one = angle.cos(), angle.sin(), torch.zeros_like(time), torch.ones_like(time)
    joint_rotation = torch.stack((one, zero, zero, zero, c, -s, zero, s, c), -1).reshape(33, 3, 3)
    rotations = torch.eye(3).expand(33, rep.skeleton.nbjoints, 3, 3).clone()
    rotations[:, 1] = joint_rotation
    root = torch.stack((3. + time * .02, 1.7 + time * .003, -7. - time * .015), -1)
    motion = rep(dict(local_joint_rots=rotations, translation=root,
                      foot_contacts=torch.ones(33, 4)), to_normalize=False,
                 lengths=torch.tensor(33))
    assert motion.shape == (33, 366)
    return rep, motion[None]


def aligned(source, offset):
    predicted = source[:, :32].clone()
    predicted[:, 4:8] = source[:, offset + 4:offset + 8]
    return predicted, source[:, offset:offset + 4].clone()


@pytest.mark.parametrize('offset', [0, 4, 24, 25])
def test_command_uses_next_four_frame_interval_and_final_heading(source_pose, offset):
    _, original = source_pose
    source = original.clone().requires_grad_()
    before = source.detach().clone()
    result = prefix_command(source, offset)
    expected_velocity = (source[:, offset + 7, [0, 2]] - source[:, offset + 3, [0, 2]]) / (4 / 30)
    torch.testing.assert_close(result[:, :2], expected_velocity)
    torch.testing.assert_close(result[:, 2:], source[:, offset + 7, 3:5], rtol=0, atol=0)
    assert result.shape == (1, 4) and not result.requires_grad
    torch.testing.assert_close(source, before, rtol=0, atol=0)


def test_command_accepts_exact_eight_frames_and_ignores_terminal_proxy(source_pose):
    _, source = source_pose
    expected = prefix_command(source[:, :8], 0)
    changed = source.clone()
    changed[:, 8:] += 1000
    changed[:, 7, 3:5] = source.new_tensor([0., -1.])
    actual = prefix_command(changed, 0)
    torch.testing.assert_close(actual[:, :2], expected[:, :2])
    torch.testing.assert_close(actual[:, 2:], source.new_tensor([[0., -1.]]))


@pytest.mark.parametrize('offset,stride', [(-1, 4), (1.5, 4), (True, 4), (0, 0), (0, 1), (0, 8), (0, 4.)])
def test_rejects_invalid_prefix_clock_and_offsets(source_pose, offset, stride):
    _, source = source_pose
    with pytest.raises(ValueError):
        prefix_command(source, offset, stride)
    with pytest.raises(ValueError):
        prefix_loss(None, source[:, :32], source[:, :4], source, offset, stride)


@pytest.mark.parametrize('shape,offset', [((1, 7, 366), 0), ((1, 33, 366), 26),
                                        ((2, 33, 366), 0), ((1, 33, 5), 0), ((33, 366), 0)])
def test_command_rejects_short_or_malformed_source(shape, offset):
    with pytest.raises(ValueError):
        prefix_command(torch.zeros(shape), offset)


@pytest.mark.parametrize('offset', [0, 4, 24, 25])
def test_exact_source_prefix_has_zero_loss_at_each_offset(source_pose, offset):
    rep, source = source_pose
    predicted, history = aligned(source, offset)
    losses = prefix_loss(rep, predicted, history, source, offset)
    assert set(losses) == set(LOSS_WEIGHTS) | {'total'}
    for value in losses.values():
        assert value.shape == ()
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=1e-9)


@pytest.mark.parametrize('which,shape', [('predicted', (1, 31, 366)), ('history', (1, 5, 366)),
                                       ('source', (1, 32, 366)), ('source', (1, 34, 366))])
def test_loss_rejects_wrong_contract_shapes(source_pose, which, shape):
    rep, source = source_pose
    inputs = dict(predicted=source[:, :32], history=source[:, :4], source=source)
    inputs[which] = torch.zeros(shape)
    with pytest.raises(ValueError):
        prefix_loss(rep, **inputs, offset=0)


def test_world_error_is_not_recentered_and_scales_are_dimensionless(source_pose):
    rep, source = source_pose
    predicted, history = aligned(source, 0)
    predicted[:, 4:8, 0] += POSITION_SCALE_M
    losses = prefix_loss(rep, predicted, history, source, 0)
    torch.testing.assert_close(losses['root_xyz'], source.new_tensor(.5 / 3), rtol=0, atol=1e-5)
    torch.testing.assert_close(losses['root_relative'], source.new_zeros(()), rtol=0, atol=1e-8)
    # Only the first interval has a 5cm jump: 1.5m/s / .5m/s = 3.
    expected_velocity = (POSITION_SCALE_M * 30 / VELOCITY_SCALE_MPS - .5) / (4 * 3)
    torch.testing.assert_close(losses['joint_velocity'], source.new_tensor(expected_velocity), rtol=0, atol=2e-5)
    expected = sum(LOSS_WEIGHTS[key] * losses[key] for key in LOSS_WEIGHTS)
    torch.testing.assert_close(losses['total'], expected, rtol=0, atol=0)
    history[..., 0] += POSITION_SCALE_M
    together = prefix_loss(rep, predicted, history, source, 0)
    torch.testing.assert_close(together['joint_velocity'], source.new_zeros(()), rtol=0, atol=1e-8)
    assert together['root_xyz'] > 0


def test_actual_history_boundary_does_not_use_source_or_decoded_past(source_pose):
    rep, source = source_pose
    predicted, history = aligned(source, 4)
    history[:, -1, 0] += .02
    before = history.clone()
    losses = prefix_loss(rep, predicted, history, source, 4)
    assert losses['joint_velocity'] > 0
    assert losses['root_xyz'] == 0
    changed = predicted.clone()
    changed[:, :4] = 1e6
    changed[:, 8:] = -1e6
    actual = prefix_loss(rep, changed, history, source, 4)
    for key in losses:
        torch.testing.assert_close(actual[key], losses[key], rtol=0, atol=0)
    torch.testing.assert_close(history, before, rtol=0, atol=0)


def test_target_velocity_rewards_source_movement_not_stillness(source_pose):
    rep, source = source_pose
    perfect, history = aligned(source, 0)
    stopped = perfect.clone()
    stopped[:, 4:8] = history[:, -1:]
    moving = prefix_loss(rep, perfect, history, source, 0)
    still = prefix_loss(rep, stopped, history, source, 0)
    assert moving['joint_velocity'] == 0 and still['joint_velocity'] > .01


def test_future_labels_and_redundant_features_do_not_replace_played_history(source_pose):
    rep, source = source_pose
    predicted, history = aligned(source, 4)
    before = history.clone()
    target = source.clone()
    target[:, 8:12, 0] += .1
    assert prefix_loss(rep, predicted, history, target, 4)['root_xyz'] > 0
    torch.testing.assert_close(history, before, rtol=0, atol=0)
    changed = source.clone()
    changed[:, 12:] = 1e6
    for name in ('ric_data', 'local_vel', 'foot_contacts'):
        changed[..., rep.indices[name]] = 1e6
    losses = prefix_loss(rep, predicted, history, changed, 4)
    assert losses['total'] == 0


def test_fk_pose_rotations_and_only_emitted_prediction_receive_gradients(source_pose):
    rep, original = source_pose
    source = original.clone().requires_grad_()
    predicted, history = aligned(source.detach(), 4)
    # A rotation error changes FK body positions, rather than redundant RIC.
    predicted[:, 4:8, rep.indices['global_rot_data'][7]] += .2
    predicted[:, 4:8, 0] += .01
    predicted.requires_grad_()
    history.requires_grad_()
    losses = prefix_loss(rep, predicted, history, source, 4)
    assert losses['root_relative'] > 0 and losses['global_rotation'] > 0
    losses['total'].backward()
    assert source.grad is None and history.grad is None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:, 4:8].abs().sum() > 0
    assert predicted.grad[:, :4].count_nonzero() == 0
    assert predicted.grad[:, 8:].count_nonzero() == 0


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))

    def forward(self, value):
        return dict(pred_global_root_values=value * self.weight)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(3.))

    def forward(self, volume):
        return volume['features'] * self.weight, volume['valid']


class Pose(nn.Module):
    def __init__(self, controller):
        super().__init__()
        object.__setattr__(self, 'controller', controller)
        self.weight = nn.Parameter(torch.tensor(4.))

    def forward(self, root, *, terrain_observation):
        return (root['pred_global_root_values'].mean() + self.controller.memory.mean()
                + terrain_observation['points'].mean()) * self.weight


def mock_observer(training):
    controller = nn.Module()
    controller.root = Root()
    controller.pose = Pose(controller)
    controller.adaptation = nn.ModuleDict(dict(encoder=Encoder()))
    controller._handles = []
    controller._adapting = True
    controller._context = dict(training=training, scene={'known': True},
                               history=torch.ones(1, 4, 366, requires_grad=True))
    controller.volume_observation = dict(features=torch.tensor(-100.))
    controller.observation = dict(points=torch.tensor(-200.))
    observer = TrainingRootPathObservation(controller)
    calls = []

    def sample(self, history, scene, root, coarse, observation):
        calls.append(dict(grad=torch.is_grad_enabled(), history=history, root=root,
                          scene=scene, coarse=coarse, observation=observation))
        path = root[..., :2] + history[:, -1:, :2]
        points = path.mean()
        height = SimpleNamespace(as_model_inputs=lambda: dict(points=points))
        return dict(features=path, valid=torch.ones(path.shape[:-1], dtype=torch.bool)), height, path

    observer.sample = MethodType(sample, observer)
    return controller, observer, calls


def test_training_hook_uses_same_post_root_observation_as_eval_and_inherited_pose_hook():
    eval_controller, eval_observer, eval_calls = mock_observer(False)
    train_controller, train_observer, train_calls = mock_observer(True)
    value = torch.ones(1, 32, 5)
    for controller in (eval_controller, train_controller):
        root = controller.root(value)
        result = controller.pose(root, terrain_observation=controller.observation)
        assert torch.isfinite(result)
    torch.testing.assert_close(train_observer.path, eval_observer.path, rtol=0, atol=0)
    torch.testing.assert_close(train_controller.memory, eval_controller.memory, rtol=0, atol=0)
    assert not eval_controller.memory.requires_grad and train_controller.memory.requires_grad
    assert TrainingRootPathObservation._before_pose is PredictedRootObservation._before_pose
    for controller, observer, calls in ((eval_controller, eval_observer, eval_calls),
                                        (train_controller, train_observer, train_calls)):
        assert len(calls) == 1 and not calls[0]['grad']
        assert not calls[0]['root'].requires_grad and not calls[0]['history'].requires_grad
        assert calls[0]['scene'] is controller._context['scene']
        assert observer.context is controller._context
        assert not observer.path.requires_grad


def test_training_query_detach_preserves_root_pose_and_encoder_neural_gradients():
    controller, _, _ = mock_observer(True)
    with torch.no_grad():
        # The hook restores neural gradients even inside native no-grad packing.
        controller.root(torch.ones(1, 32, 5))
    assert controller.memory.requires_grad
    root = controller.root(torch.ones(1, 32, 5))
    controller.pose(root, terrain_observation=controller.observation).backward()
    for module in (controller.root, controller.pose, controller.adaptation['encoder']):
        assert module.weight.grad is not None and module.weight.grad.abs() > 0
    assert controller._context['history'].grad is None


def test_real_sampling_matches_deployed_root_path_in_both_modes_with_detached_coordinates():
    controller, observer, _ = mock_observer(True)
    observer.sample = MethodType(PredictedRootObservation.sample, observer)
    history = controller._context['history'].detach().clone()
    history[..., :3] = history.new_tensor([.2, 1., .3])
    history[..., 3:5] = history.new_tensor([1., 0.])
    history.requires_grad_()
    controller._context['history'] = history
    scene = dict(heightfield_y_m=torch.full((33, 33), .15),
                 min_point_world_xyz_m=torch.tensor([-8., 0., -8.]), dx_m=.5)
    controller._context['scene'] = scene
    controller.frames = 32
    controller.inference = SimpleNamespace(
        global_motion_rep=SimpleNamespace(unnormalize=lambda root: root),
        _root_model=SimpleNamespace(args=dict(observation_range_policy=INTENT_RANGE_POLICY)))
    controller.sampler = VolumetricEgoSampler(VolumeSampling(grid_size=4))
    field = AnalyticPlaneField(.15)
    controller.scene_field = lambda _: field
    coarse = controller.sampler.observe(history[..., :5], history.new_tensor([[.4, 1.]]), field, 28 / 30)
    longitudinal, lateral = torch.meshgrid(torch.linspace(0., 2., 9), torch.linspace(-.6, .6, 9), indexing='ij')
    points = torch.stack((lateral, torch.full_like(lateral, .15), longitudinal), -1)
    controller.observation = dict(points=points[None, None].expand(1, 3, -1, -1, -1))
    root_input = torch.zeros(1, 32, 5)
    root_input[0, :, 0] = torch.arange(32) * .01
    root_input[0, :, 2] = torch.arange(32) * .02
    root_input.requires_grad_()
    reference = object.__new__(PredictedRootObservation)
    reference.controller = controller
    with torch.no_grad():
        expected_volume, expected_height, expected_path = reference.sample(
            history.detach(), scene, root_input.detach() * controller.root.weight.detach(),
            coarse, controller.observation)
    for training in (False, True):
        controller._context['training'] = training
        controller.volume_observation = coarse
        controller.root(root_input)
        for key, value in expected_volume.items():
            torch.testing.assert_close(controller.volume_observation[key], value, rtol=0, atol=0)
            assert not controller.volume_observation[key].requires_grad
        for key, value in expected_height.as_model_inputs().items():
            torch.testing.assert_close(observer.height.as_model_inputs()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(observer.path, expected_path, rtol=0, atol=0)
        assert not observer.path.requires_grad
        assert controller.memory.requires_grad == training
    controller.memory.sum().backward()
    assert controller.adaptation['encoder'].weight.grad.abs() > 0
    assert controller.root.weight.grad is None and root_input.grad is None and history.grad is None


@pytest.mark.parametrize('skip', ['deferred', 'no_context', 'not_adapting'])
def test_training_hook_preserves_inactive_and_deferred_execution(skip):
    controller, observer, calls = mock_observer(True)
    if skip == 'deferred':
        observer.deferred = True
    elif skip == 'no_context':
        controller._context = None
    else:
        controller._adapting = False
    controller.root(torch.ones(1, 32, 5))
    assert not calls and observer.context is None


class StatefulQuantizer(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.register_buffer('ema_updates', torch.zeros(()))

    def forward(self, value):
        if self.training:
            self.ema_updates.add_(1)
        return super().forward(value)


def test_scope_freezes_encoders_quantizer_ema_and_all_dropout_but_keeps_selected_gradients():
    controller = nn.Module()
    controller.root = nn.Sequential(nn.Linear(2, 2), nn.Dropout(.9))
    controller.pose = nn.Linear(2, 2)
    controller.codec = nn.Module()
    controller.codec.encoder = nn.Linear(2, 2)
    controller.codec.decoder = nn.Linear(2, 2)
    controller.codec.quantizer = StatefulQuantizer()
    controller.adaptation = nn.ModuleDict({name: nn.Linear(2, 2) for name in
        ('encoder', 'root_stage1', 'root_stage2', 'pose', 'decoder', 'height_encoder')})
    controller.other = nn.Linear(2, 2)
    for parameter in controller.parameters():
        parameter.grad = torch.ones_like(parameter)
    names = configure_scope(controller)
    active = ('root', 'pose', 'codec.decoder', 'adaptation.root_stage1', 'adaptation.root_stage2',
              'adaptation.pose', 'adaptation.decoder', 'adaptation.height_encoder')
    expected = {f'{module}.{parameter}' for module in active for parameter in ('weight', 'bias')}
    expected -= {'root.weight', 'root.bias'}
    expected |= {'root.0.weight', 'root.0.bias'}
    assert set(names) == expected
    assert not any(module.training for module in controller.modules())
    assert all(parameter.grad is None for parameter in controller.parameters())
    output = sum(module(torch.ones(1, 2)).sum() for module in (
        controller.root, controller.pose, controller.codec.decoder,
        controller.codec.encoder, controller.codec.quantizer, *controller.adaptation.values(), controller.other))
    output.backward()
    for name, parameter in controller.named_parameters():
        assert parameter.requires_grad == (name in expected)
        assert (parameter.grad is not None) == (name in expected)
    assert controller.codec.quantizer.ema_updates == 0
    torch.testing.assert_close(controller.root(torch.ones(1, 2)), controller.root(torch.ones(1, 2)), rtol=0, atol=0)
