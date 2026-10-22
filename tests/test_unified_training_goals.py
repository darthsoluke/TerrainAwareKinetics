from types import SimpleNamespace
import json

import pytest
import torch

from motionbricks.data.unified_training_goals import source_terminal_goal, validate_source_terminal_goal
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.native_mb_online_conditions import VELOCITY_PROTOCOL
from scripts.pilot_unified_goal_alignment import freeze_for_root_comparison, source_windows, training_objective
import scripts.pilot_unified_goal_alignment as fit


def source():
    root = torch.zeros(2, 33, 5, dtype=torch.float64)
    root[..., 3] = 1
    root[:, 31, [0, 2]] = torch.tensor([2., 3.], dtype=root.dtype)
    root[:, 32, [0, 2]] = torch.tensor([2.1, 3.2], dtype=root.dtype)
    root[:, 31, 3:5] = torch.tensor([0., 1.], dtype=root.dtype)
    return root


def test_interval_and_heading_match_native_terminal_indices():
    root = source()
    result = source_terminal_goal(root)
    torch.testing.assert_close(result, root.new_tensor([[3., 6., 0., 1.]]).expand(2, -1))
    root[:, 32, 3:5] = root.new_tensor([-1., 0.])
    torch.testing.assert_close(source_terminal_goal(root), result, atol=0, rtol=0)


def test_no_height_leak_or_mutation_and_no_target_gradient():
    root = source().requires_grad_(True)
    original = root.detach().clone()
    result = source_terminal_goal(root)
    assert not result.requires_grad
    torch.testing.assert_close(root, original, atol=0, rtol=0)
    modified = original.clone()
    modified[..., 1] = torch.randn_like(modified[..., 1]) * 100
    torch.testing.assert_close(source_terminal_goal(modified), result, atol=0, rtol=0)


def test_translation_and_yaw_covariance():
    root = source()
    translated = root.clone()
    translated[..., [0, 2]] += root.new_tensor([120., -90.])
    torch.testing.assert_close(source_terminal_goal(translated), source_terminal_goal(root), atol=1e-10, rtol=0)
    rotated = root.clone()
    rotated[..., 0], rotated[..., 2] = root[..., 2], -root[..., 0]
    rotated[..., 3], rotated[..., 4] = -root[..., 4], root[..., 3]
    torch.testing.assert_close(source_terminal_goal(rotated), root.new_tensor([[6., -3., -1., 0.]]).expand(2, -1))


def test_rejects_padding_wrong_clock_and_mismatched_proxy():
    root = source()
    with pytest.raises(ValueError):
        source_terminal_goal(root[:, :32])
    with pytest.raises(ValueError):
        source_terminal_goal(root, fps=60)
    wrong = source_terminal_goal(root)
    wrong[:, :2] = .75
    with pytest.raises(ValueError, match='differs'):
        validate_source_terminal_goal(root, wrong)


def test_training_guard_checks_actual_forward_controls_before_loss():
    root = source()
    model = SimpleNamespace(inference=SimpleNamespace(global_motion_rep=SimpleNamespace(extract_root=lambda x: x)))
    result = dict(goal_protocol=VELOCITY_PROTOCOL['version'], input_controls=torch.zeros(2, 4, dtype=root.dtype))
    with pytest.raises(ValueError, match='differs'):
        UnifiedTerrainController.native_losses(model, result, root, root[:, :4])
    with pytest.raises(ValueError, match='actual forward'):
        UnifiedTerrainController.native_losses(model, {}, root, root[:, :4])


def test_priority_one_freezes_everything_except_root_environment_modules():
    model = torch.nn.Module()
    model.root = torch.nn.Linear(4, 4)
    model.pose = torch.nn.Linear(4, 4)
    model.codec = torch.nn.Linear(4, 4)
    model.adaptation = torch.nn.ModuleDict({name: torch.nn.Linear(4, 4) for name in
        ('encoder', 'pose', 'decoder', 'height_encoder', 'root_stage1', 'root_stage2')})
    names = freeze_for_root_comparison(model)
    assert len(names) == 4
    assert all(name.startswith(('adaptation.root_stage1.', 'adaptation.root_stage2.')) for name in names)
    assert not any(p.requires_grad for p in model.adaptation['encoder'].parameters())
    assert not any(p.requires_grad for p in model.pose.parameters())
    assert not any(p.requires_grad for p in model.codec.parameters())


def test_dense_coverage_keeps_real_interval_endpoints_and_archived_control():
    assert source_windows(81, 24) == [0, 24, 48]
    assert source_windows(81, 1) == list(range(49))
    assert all(start + 33 <= 81 for start in source_windows(81, 1))
    with pytest.raises(ValueError):
        source_windows(32, 1)


def test_staged_root_objective_keeps_native_weighting_without_joint_gradient():
    root = torch.tensor(2., requires_grad=True)
    pose = torch.tensor(3., requires_grad=True)
    parts = dict(native_root=root.square(), native_total=root.square() + pose.square())
    value = training_objective(parts, 'root')
    value.backward()
    assert root.grad == 4 and pose.grad is None
    assert training_objective(parts, 'joint') is parts['native_total']
    with pytest.raises(ValueError):
        training_objective(parts, 'scaled')


def test_height_readout_updates_only_y_row_of_native_root_output():
    model = torch.nn.Module()
    model.root = torch.nn.Module()
    model.root._conv_output = torch.nn.Module()
    layer = torch.nn.Conv1d(4, 5, 3, padding=1)
    model.root._conv_output.model = torch.nn.ModuleList([layer])
    model.pose = torch.nn.Linear(4, 4)
    model.codec = torch.nn.Linear(4, 4)
    model.adaptation = torch.nn.ModuleDict({name: torch.nn.Linear(4, 4) for name in
        ('encoder', 'pose', 'decoder', 'height_encoder', 'root_stage1', 'root_stage2')})
    freeze_for_root_comparison(model, height_readout=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    weight, bias = layer.weight.detach().clone(), layer.bias.detach().clone()
    layer(torch.ones(1, 4, 8)).sum().backward()
    optimizer.step()
    keep = [0, 2, 3, 4]
    torch.testing.assert_close(layer.weight[keep], weight[keep], atol=0, rtol=0)
    torch.testing.assert_close(layer.bias[keep], bias[keep], atol=0, rtol=0)
    assert not torch.equal(layer.weight[1], weight[1])
    assert not torch.equal(layer.bias[1], bias[1])
    assert not any(p.requires_grad for p in model.pose.parameters())


def test_real_prefix_extension_preserves_review_sources_and_original_frames(tmp_path, monkeypatch):
    dataset_path = tmp_path / 'dataset'
    dataset_path.mkdir()
    (dataset_path / 'native_manifest.json').write_text('{}')
    cases_path = tmp_path / 'cases.pt'
    cases_path.write_bytes(b'source identity')
    world = torch.zeros(105, 5)
    world[:, 1] = 1
    world[:, 3] = 1
    world[:, 2] = torch.arange(105) * .01
    scene = dict(heightfield_y_m=torch.zeros(9, 9), min_point_world_xyz_m=[-4., 0., -4.], dx_m=1.)
    cases = [dict(name='stairs', label=name, row=dict(source_id=name),
                  start_frame=24, world=world[24:].clone(), scene=scene) for name in ('up', 'down')]
    originals = [case['world'].clone() for case in cases]
    fake = SimpleNamespace(records=[dict(source_id=name) for name in ('up', 'down')],
                           full_reference_item=lambda _i: dict(world=world.clone()))
    monkeypatch.setattr(fit, 'TerrainLearningDataset', lambda *_args: fake)
    monkeypatch.setattr(fit, 'exact_skin_metrics', lambda *_args: dict(
        body_prism_max_m=.01, unknown_body_fraction=0., uncertain_inside_samples=0))
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(dict(source_cases_sha256=fit.sha256(cases_path),
        dataset_manifest_sha256=fit.sha256(dataset_path / 'native_manifest.json'),
        rows=[dict(label=name, eligible_for_flat_approach_extension=True, added_real_prefix_frames=24,
                   old_start=24, new_start=0) for name in ('up', 'down')])))
    args = SimpleNamespace(approach_dataset=dataset_path, approach_audit=audit_path, cases=cases_path,
                           arm='terminal', objective='root', window_stride=1, height_readout=False, device='cpu')
    controller = SimpleNamespace(inference=SimpleNamespace(global_motion_rep=None))
    result, checks = fit.extend_real_approaches(cases, args, controller, None)
    assert len(checks) == 2
    for index, case in enumerate(cases):
        assert case['start_frame'] == 24 and len(case['world']) == 81
        torch.testing.assert_close(case['world'], originals[index], atol=0, rtol=0)
        torch.testing.assert_close(result[index]['world'][24:], originals[index], atol=0, rtol=0)
        assert result[index]['start_frame'] == 0 and len(result[index]['world']) == 105
    cases[0]['world'][0, 1] += .001
    with pytest.raises(AssertionError):
        fit.extend_real_approaches(cases, args, controller, None)
