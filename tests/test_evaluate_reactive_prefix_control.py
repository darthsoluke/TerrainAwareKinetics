"""CPU-only control clock, genuine history, metric and fail-closed regression."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_reactive_prefix_control as evaluation


def test_authored_arena_keeps_seed_ground_and_explicit_known_extent():
    seed = torch.zeros(1, 4, 366)
    seed[..., :3] = torch.tensor([2., 1.2, -3.])
    scene = evaluation.control_arena(seed, .15)
    assert scene['heightfield_y_m'].shape == (321, 321)
    torch.testing.assert_close(scene['heightfield_y_m'], torch.full((321, 321), .15))
    torch.testing.assert_close(scene['min_point_world_xyz_m'], torch.tensor([-30., .15, -35.]))
    assert float(scene['dx_m']) == pytest.approx(.2)


def test_unknown_sensor_points_are_masked_while_entirely_unknown_inputs_fail():
    valid = torch.tensor([[True, False]])
    controller = SimpleNamespace(observation={'valid': valid}, volume_observation={'valid': valid})
    observer = SimpleNamespace(height=SimpleNamespace(valid=valid))
    assert evaluation.check_observations(controller, observer)['predicted_path_volume'] == 1
    controller.volume_observation['valid'] = torch.zeros_like(valid)
    with pytest.raises(ValueError, match='No known'):
        evaluation.check_observations(controller, observer)


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Evaluation tests must remain CPU-only')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)


@pytest.fixture
def factual_seed():
    from motionbricks.data.native_mb_data import build_native_representation

    rep = build_native_representation()
    rotations = torch.eye(3).expand(32, 30, 3, 3).clone()
    root = torch.tensor([3., 1.7, -7.]).expand(32, 3).clone()
    world = rep(dict(local_joint_rots=rotations, translation=root, foot_contacts=torch.ones(32, 4)),
                to_normalize=False, lengths=torch.tensor(32))[None]
    return rep, world


def body_at_root(world, skeleton):
    joints = world[..., :3].unsqueeze(-2).expand(*world.shape[:2], 30, 3).clone()
    right_hip, left_hip = skeleton.hip_joint_idx
    yaw = torch.atan2(world[..., 4], world[..., 3])
    across = torch.stack((-yaw.cos(), torch.zeros_like(yaw), yaw.sin()), -1) * .1
    joints[..., right_hip, :] += across
    joints[..., left_hip, :] -= across
    return joints


def test_trials_fixed_clock_independent_heading_and_actual_sample_hold(factual_seed):
    rep, world = factual_seed
    seed = rep.change_first_heading(world[:, :4], 1.1, False, False)
    trials = evaluation.controlled_trials(seed)
    assert [trial['frames'] for trial in trials] == [480, 480, 540]
    trial = trials[-1]
    controls = trial['requested_controls']
    forward = torch.tensor([np.sin(1.1), np.cos(1.1)], dtype=torch.float32) * 1.2
    right = torch.tensor([np.cos(1.1), -np.sin(1.1)], dtype=torch.float32) * 1.2
    for frame, velocity in ((0, forward), (90, -right), (180, -forward), (270, right)):
        torch.testing.assert_close(controls[frame, :2], velocity)
    torch.testing.assert_close(controls[:360, 2:], seed[0, -1, 3:5].expand(360, 2))
    assert not torch.equal(controls[360, 2:], controls[400, 2:])
    assert torch.equal(controls[450:, :2], torch.zeros(90, 2))
    torch.testing.assert_close(controls[450, 2:], torch.tensor([np.cos(6.5), np.sin(6.5)], dtype=torch.float32))
    torch.testing.assert_close(trial['issued_controls'][90], controls[88], atol=0, rtol=0)
    torch.testing.assert_close(trial['issued_controls'][92], controls[92], atol=0, rtol=0)
    again = evaluation.controlled_trials(seed)
    for first, second in zip(trials, again):
        assert first['phases'] == second['phases']
        torch.testing.assert_close(first['issued_controls'], second['issued_controls'], atol=0, rtol=0)


def test_raw_four_frame_execution_and_fk_reencoding_failure_preservation(factual_seed, tmp_path):
    from motionbricks.data.native_mb_observed_history import reencode_played_history

    rep, world = factual_seed
    seed = world[:, :4].clone()
    trial = evaluation.controlled_trials(seed)[1]
    trial = dict(trial, frames=12)
    received, predictions = [], []

    def controller(history, command, scene):
        received.append(history.clone())
        if len(received) == 3:
            raise ValueError('intentional third-call failure')
        output = world.clone()
        output[:, :4, :3] += 100.  # Decoded past must never enter played history.
        output[:, 4:8, :3] = history[:, -1:, :3] + torch.tensor([0., 0., .04]) * torch.arange(1, 5)[:, None]
        predictions.append(output)
        return dict(world=output)

    raw = evaluation.run_rollout(controller, rep, seed, {}, trial, reencode=reencode_played_history,
                                synchronize=lambda: None, observe=lambda: {})
    assert not raw['completed'] and len(raw['failures']) == 1
    assert raw['failures'][0]['emitted_frames'] == 8
    assert raw['world'].shape == (1, 12, 366)
    torch.testing.assert_close(raw['world'][:, :4], seed, atol=0, rtol=0)
    torch.testing.assert_close(raw['world'][:, 4:8], predictions[0][:, 4:8], atol=0, rtol=0)
    torch.testing.assert_close(received[1], reencode_played_history(predictions[0][:, 4:8], rep), atol=0, rtol=0)
    torch.testing.assert_close(received[2], reencode_played_history(predictions[1][:, 4:8], rep), atol=0, rtol=0)
    assert len(raw['native_forecasts']) == 2 and len(raw['conditioned_histories']) == 3
    assert all(not history.requires_grad for history in received)
    torch.save(raw, tmp_path / 'partial.pt')
    restored = torch.load(tmp_path / 'partial.pt', weights_only=False)
    torch.testing.assert_close(restored['world'], raw['world'], atol=0, rtol=0)
    assert restored['failures'] == raw['failures']
    fk = rep.inverse(raw['world'], False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    assert fk.shape == (1, 12, 30, 3)
    metrics = evaluation.motion_metrics(raw['world'], fk, trial, 0., skeleton=rep.skeleton)
    assert metrics['body_relative_joint_step_m']['maximum'] < 2e-5
    assert metrics['phases'][0]['root_height_m']['mean'] == pytest.approx(1.7)


def test_nonfinite_forecast_kept_without_emission_or_tail_hold(factual_seed):
    rep, world = factual_seed
    broken = world.clone()
    broken[0, 7, 1] = float('nan')
    trial = evaluation.controlled_trials(world[:, :4])[0]
    raw = evaluation.run_rollout(lambda *args: dict(world=broken), rep, world[:, :4], {}, trial,
                                reencode=lambda motion, rep: motion.clone(), synchronize=lambda: None,
                                observe=lambda: {})
    assert not raw['completed'] and raw['world'].shape[1] == 4
    assert len(raw['native_forecasts']) == 1
    assert torch.isnan(raw['native_forecasts'][0]).any()
    assert len(raw['failures']) == 1


def test_heading_uses_displayed_fk_and_matches_reencoded_history(factual_seed):
    from motionbricks.data.native_mb_observed_history import reencode_played_history
    from motionbricks.motionlib.core.motion_reps.tools.heading import calc_heading_from_joints_pos

    rep, original = factual_seed
    seed = reencode_played_history(original[:, :4], rep)
    trial = evaluation.controlled_trials(seed)[0]
    world = original[:, :1].expand(1, 484, 366).clone()
    actual = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    actual_heading = calc_heading_from_joints_pos(actual, rep.skeleton)
    wrong_heading = actual_heading + .7
    world[..., 3:5] = torch.stack((wrong_heading.cos(), wrong_heading.sin()), -1)
    displayed = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    torch.testing.assert_close(displayed, actual, atol=0, rtol=0)
    metric = evaluation.motion_metrics(world, displayed, trial, 0., skeleton=rep.skeleton)
    phase = metric['phases'][0]
    assert phase['absolute_heading_error_rad']['maximum'] < 1e-5
    assert phase['raw_feature_heading_error_rad']['p95'] == pytest.approx(.7, abs=1e-5)
    assert phase['raw_vs_fk_heading_disagreement_rad']['p95'] == pytest.approx(.7, abs=1e-5)
    assert evaluation.control_screen(metric, trial)['passed']
    observed = reencode_played_history(world[:, 4:8], rep)
    observed_heading = torch.atan2(observed[..., 4], observed[..., 3])
    torch.testing.assert_close(observed_heading, actual_heading[:, 4:8], atol=1e-5, rtol=0)

    # Conversely, correct redundant heading must not hide a rotated display.
    rotated = rep.change_first_heading(original[:, :1].expand(1, 484, 366),
                                       float(actual_heading[0, 0]) + .4, False, False)
    rotated[..., 3:5] = seed[:, -1:, 3:5]
    displayed = rep.inverse(rotated, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    metric = evaluation.motion_metrics(rotated, displayed, trial, 0., skeleton=rep.skeleton)
    assert metric['phases'][0]['absolute_heading_error_rad']['p95'] == pytest.approx(.4, abs=1e-5)
    assert metric['phases'][0]['raw_feature_heading_error_rad']['p95'] < 1e-5
    assert not evaluation.control_screen(metric, trial)['passed']


def test_metrics_signed_direction_stop_clock_drift_and_join_step(factual_seed):
    rep, initial = factual_seed
    trial = evaluation.controlled_trials(initial[:, :4])[1]
    controls = trial['requested_controls'].double()
    world = initial[:, :1].double().expand(1, 484, 366).clone()
    world[0, 4:, [0, 2]] += controls[:, :2].cumsum(0) / 30
    world[0, 4:, 3:5] = controls[:, 2:]
    joints = body_at_root(world, rep.skeleton)
    joints[0, 4, 1, 0] += .2  # Include the first emitted boundary in joint steps.
    metrics = evaluation.motion_metrics(world, joints, trial, 1.2, skeleton=rep.skeleton)
    idle, forward, stop = metrics['phases']
    assert idle['idle_endpoint_displacement_m'] == 0
    assert idle['body_relative_joint_step_m']['maximum'] == pytest.approx(.2)
    assert forward['mean_signed_velocity_mps'] == pytest.approx(1.2)
    assert forward['orthogonal_error_m']['maximum'] < 1e-12
    assert forward['root_height_above_plane_m']['mean'] == pytest.approx(.5)
    assert forward['start_threshold']['onset_seconds'] == pytest.approx(1 / 30)
    assert forward['start_threshold']['confirmed_seconds'] == pytest.approx(6 / 30)
    assert stop['stop_delay']['onset_seconds'] == pytest.approx(1 / 30)
    assert stop['idle_endpoint_displacement_m'] == 0
    assert evaluation.control_screen(metrics, trial)['passed']
    # A failed start is a missing threshold crossing, never a zero-second success.
    frozen = world.clone()
    frozen[..., [0, 2]] = initial[0, 0, [0, 2]].double()
    failed = evaluation.motion_metrics(frozen, body_at_root(frozen, rep.skeleton), trial, 1.2, skeleton=rep.skeleton)
    assert failed['phases'][1]['start_threshold']['onset_seconds'] is None
    assert failed['phases'][1]['mean_signed_velocity_mps'] == 0
    assert not evaluation.control_screen(failed, trial)['passed']


def test_reverse_and_strafe_metrics_use_command_axis(factual_seed):
    rep, initial = factual_seed
    trial = evaluation.controlled_trials(initial[:, :4])[2]
    controls = trial['requested_controls'].double()
    world = initial[:, :1].double().expand(1, 544, 366).clone()
    world[0, 4:, [0, 2]] += controls[:, :2].cumsum(0) / 30
    world[0, 4:, 3:5] = controls[:, 2:]
    metrics = evaluation.motion_metrics(world, body_at_root(world, rep.skeleton), trial, 1.2, skeleton=rep.skeleton)
    for phase in metrics['phases'][:4]:
        assert phase['mean_signed_velocity_mps'] == pytest.approx(1.2)
        assert phase['orthogonal_error_m']['maximum'] < 1e-12
    assert evaluation.control_screen(metrics, trial)['passed']
    assert metrics['phases'][4]['heading_change_rad'] == pytest.approx(1.8 * 89 / 30)
    assert metrics['phases'][4]['idle_max_displacement_m'] == 0
    assert metrics['phases'][4]['absolute_heading_error_rad']['maximum'] < 1e-7
    partial = evaluation.motion_metrics(world[:, :100], body_at_root(world[:, :100], rep.skeleton), trial, 1.2,
                                        skeleton=rep.skeleton)
    assert partial['emitted_frames'] == 96
    assert len(partial['phases']) == 2 and not partial['phases'][1]['complete']
    assert not evaluation.control_screen(partial, trial)['passed']


def test_drifting_idle_and_late_stop_fail_research_screen(factual_seed):
    rep, seed = factual_seed
    trial = evaluation.controlled_trials(seed[:, :4])[0]
    world = seed[:, :1].double().expand(1, 484, 366).clone()
    world[0, 4:, 0] += torch.arange(1, 481) * .001
    metrics = evaluation.motion_metrics(world, body_at_root(world, rep.skeleton), trial, 0., skeleton=rep.skeleton)
    assert metrics['phases'][0]['idle_endpoint_displacement_m'] == pytest.approx(.48)
    screen = evaluation.control_screen(metrics, trial)
    assert not screen['passed']
    assert any(check['criterion'] == 'idle16_endpoint_le_0.05m' for check in screen['failures'])
    trial = evaluation.controlled_trials(seed[:, :4])[1]
    commands = trial['requested_controls'].double().clone()
    # Keep moving for 20 frames after the authored stop, then genuinely stop.
    commands[240:260, :2] = commands[239, :2]
    world[0, 4:, [0, 2]] = seed[0, 0, [0, 2]].double() + commands[:, :2].cumsum(0) / 30
    world[0, 4:, 3:5] = commands[:, 2:]
    metrics = evaluation.motion_metrics(world, body_at_root(world, rep.skeleton), trial, 0., skeleton=rep.skeleton)
    screen = evaluation.control_screen(metrics, trial)
    assert not screen['passed']
    assert any(check['phase'] == 'stop' and 'stop_confirmed' in check['criterion']
               for check in screen['failures'])


@pytest.mark.parametrize('defect,criterion', [('too_fast', 'signed_mean'),
                                           ('lateral_drift', 'orthogonal_max'),
                                           ('heading', 'fixed_heading_p95')])
def test_other_control_gates_with_full_native_shapes(factual_seed, defect, criterion):
    rep, seed = factual_seed
    trial = evaluation.controlled_trials(seed[:, :4])[1]
    commands = trial['requested_controls'].double().clone()
    if defect == 'too_fast':
        commands[60:240, :2] *= 1.3
    world = seed[:, :1].double().expand(1, 484, 366).clone()
    world[0, 4:, [0, 2]] += commands[:, :2].cumsum(0) / 30
    world[0, 4:, 3:5] = commands[:, 2:]
    if defect == 'lateral_drift':
        forward = commands[60, :2] / commands[60, :2].norm()
        lateral = torch.stack((forward[1], -forward[0]))
        world[0, 64:244, [0, 2]] += torch.arange(1, 181)[:, None] * .002 * lateral
    if defect == 'heading':
        angles = torch.atan2(commands[60:240, 3], commands[60:240, 2]) + np.deg2rad(20)
        world[0, 64:244, 3:5] = torch.stack((angles.cos(), angles.sin()), -1)
    metric = evaluation.motion_metrics(world, body_at_root(world, rep.skeleton), trial, 0., skeleton=rep.skeleton)
    screen = evaluation.control_screen(metric, trial)
    assert not screen['passed']
    assert any(criterion in check['criterion'] for check in screen['failures'])


def test_idle_selection_and_factual_plane_fail_closed():
    def row(domain, split, category, index):
        return dict(cache_index=0, row_index=index, category=category,
                    record=dict(domain=domain, split=split, quarter=0))
    selected = row('bones', 'validation', 'idle', 7)
    selection = dict(validation=[row('pfnn', 'validation', 'idle', 0),
                                 row('bones', 'train', 'idle', 1), selected,
                                 row('bones', 'validation', 'idle', 8)])
    assert evaluation.select_idle(selection) == selected
    with pytest.raises(ValueError, match='no BONES'):
        evaluation.select_idle(dict(validation=selection['validation'][:2]))
    scene = dict(heightfield_y_m=torch.full((3, 4), 2.3), dx_m=torch.tensor(.2))
    assert evaluation.plane_height(scene) == pytest.approx(2.3)
    scene['heightfield_y_m'][1, 1] += 1e-5
    with pytest.raises(ValueError, match='constant'):
        evaluation.plane_height(scene)


@pytest.mark.parametrize('changed', [dict(updates=63), dict(updates=True), dict(exposures=511),
                                     dict(schema='other')])
def test_fixed_final_endpoint_schema_and_hashes(changed):
    contract = dict(schema=evaluation.PILOT_SCHEMA, parent_sha256='parent', bundle_sha256='bundle',
                    arm='source_history', eligible_for_automatic_export=False)
    saved = dict(schema=evaluation.PILOT_SCHEMA, contract=contract, updates=64, exposures=512)
    assert evaluation.validate_endpoint(saved, 'parent', 'bundle') == contract
    with pytest.raises(ValueError, match='fixed-64/512'):
        evaluation.validate_endpoint(dict(saved, **changed), 'parent', 'bundle')
    with pytest.raises(ValueError, match='fixed-64/512'):
        evaluation.validate_endpoint(saved, 'wrong_parent', 'bundle')
    with pytest.raises(ValueError, match='fixed-64/512'):
        evaluation.validate_endpoint(saved, 'parent', 'wrong_bundle')


def test_full_skin_gate_and_missing_thresholds_remain_failures(factual_seed):
    clean = dict(vertices_per_frame=18056, swept_substeps=8, unknown_body_fraction=0.,
                 uncertain_inside_samples=0, body_prism_max_m=.019)
    assert evaluation.skin_failures(clean) == []
    for changed in (dict(body_prism_max_m=.021), dict(unknown_body_fraction=.001),
                    dict(uncertain_inside_samples=1), dict(vertices_per_frame=100),
                    dict(swept_substeps=4), dict(body_prism_max_m=None)):
        assert evaluation.skin_failures(dict(clean, **changed))
    rep, world = factual_seed
    trial = evaluation.controlled_trials(world[:, :4])[1]
    motion = world[:, :1].expand(1, 484, 366).clone()
    metric = evaluation.motion_metrics(motion, body_at_root(motion, rep.skeleton), trial, 0., skeleton=rep.skeleton)
    base = dict(model='parent', trial=trial['name'], completed=True, motion=metric)
    candidate = deepcopy(base)
    candidate['model'] = 'generated_history'
    result = evaluation.comparison([base, candidate])[0]
    assert result['phases'][1]['candidate_minus_parent']['start_threshold'] == dict(
        parent_seconds=None, candidate_seconds=None, delta_seconds=None)
