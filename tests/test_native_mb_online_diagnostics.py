"""Bounded live viewer diagnostics contract tests."""

import json
from types import SimpleNamespace

import torch
from torch import nn

from motionbricks.data import native_mb_online_live as live
from motionbricks.data.native_mb_online_conditions import ONLINE_PROTOCOL


class _GlobalRep:
    indices = {'all': list(range(5)), 'root': list(range(5))}

    @staticmethod
    def unnormalize(values):
        scale = values.new_tensor([2.0, 1.0, 3.0, 1.0, 1.0])
        return values * scale


class _LocalRep:
    indices = {
        'root': [0, 1, 2, 3],
        'ric_data': [0, 1],
        'global_rot_data': [2],
    }


class _DualRep:
    @staticmethod
    def global_to_local(history, *_args, **_kwargs):
        return history[..., :4]


class _RootBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        frames = torch.arange(32, dtype=torch.float32)
        trace = torch.zeros(1, 32, 5)
        trace[0, :, 0] = frames * 0.25
        trace[0, :, 1] = 2.0 + frames * 0.01
        trace[0, :, 2] = -frames * 0.5
        self.trace = trace

    def forward(self):
        return {'pred_global_root_values': self.trace}


class _FakeInference:
    EXTERNAL_POSE_FEATURE_MODE = 'joint_positions_and_rotations'

    def __init__(self, with_root_backbone=True):
        self.global_motion_rep = _GlobalRep()
        self.local_motion_rep = _LocalRep()
        self.motion_rep = SimpleNamespace(dual_rep=_DualRep())
        root_model = SimpleNamespace(args={'online_conditioning': ONLINE_PROTOCOL['version']})
        if with_root_backbone:
            root_model.backbone_net = _RootBackbone()
        self._root_model = root_model
        self._pose_model = SimpleNamespace(args={'online_conditioning': ONLINE_PROTOCOL['version']})
        self.observations = []
        self.prediction = torch.zeros(1, 32, 5)
        frames = torch.arange(32, dtype=torch.float32)
        self.prediction[0, :, 0] = 100.0 + frames
        self.prediction[0, :, 1] = 1.0 + frames * 0.02
        self.prediction[0, :, 2] = -100.0 - frames
        self.count = torch.tensor([[8]])

    def predict(self, *_args, terrain_observation, **_kwargs):
        self.observations.append(terrain_observation)
        if hasattr(self._root_model, 'backbone_net'):
            self._root_model.backbone_net()
        return self.prediction, self.count


def _history():
    history = torch.zeros(1, 4, 5)
    history[0, :, 0] = torch.tensor([2.5, 1.5, 0.7, 0.0])
    history[0, :, 1] = torch.tensor([1.0, 1.1, 1.15, 1.2])
    history[0, :, 2] = torch.tensor([-3.0, -2.0, -1.0, 0.0])
    yaw = 0.7
    history[..., 3] = torch.cos(torch.tensor(yaw))
    history[..., 4] = torch.sin(torch.tensor(yaw))
    return history


def _scene():
    heights = torch.arange(101 * 101, dtype=torch.float32).reshape(101, 101) * 0.0001
    valid = torch.ones_like(heights, dtype=torch.bool)
    heights[50, 50] = float('nan')
    valid[50, 50] = False
    return {
        'heightfield_y_m': heights,
        'heightfield_valid': valid,
        'min_point_world_xyz_m': [-5.0, 0.0, -5.0],
        'dx_m': 0.1,
    }


def _goal():
    heading = 0.8
    return torch.tensor([[1.5, -0.7, torch.cos(torch.tensor(heading)),
                          torch.sin(torch.tensor(heading))]])


def test_diagnostics_capture_actual_observation_and_both_root_stages_read_only(monkeypatch):
    inference = _FakeInference()
    history, goal, scene = _history(), _goal(), _scene()
    history_before, goal_before = history.clone(), goal.clone()
    root_backbone = inference._root_model.backbone_net
    hooks_before = len(root_backbone._forward_hooks)
    diagnostics = {}
    sampled = []
    extract = live.extract_local_terrain_observation

    def capture_observation(*args, **kwargs):
        observation = extract(*args, **kwargs)
        sampled.append((observation, observation.points.clone(), observation.valid.clone()))
        return observation

    monkeypatch.setattr(live, 'extract_local_terrain_observation', capture_observation)

    prediction, count = live.predict_planar_terrain(
        inference, history, goal, scene, num_tokens=8, diagnostics=diagnostics)

    assert prediction is inference.prediction
    assert count is inference.count
    assert len(root_backbone._forward_hooks) == hooks_before
    torch.testing.assert_close(history, history_before)
    torch.testing.assert_close(goal, goal_before)
    observation = inference.observations[-1]
    sampled_observation, sampled_points, sampled_valid = sampled[-1]
    assert observation['points'] is sampled_observation.points
    assert observation['valid'] is sampled_observation.valid
    torch.testing.assert_close(observation['points'], sampled_points)
    torch.testing.assert_close(observation['valid'], sampled_valid)
    assert diagnostics['version'] == 'soma_motion_debug_v1'
    assert diagnostics['observation_source'] == 'current_played_history'
    assert diagnostics['input_intent']['height_source'] == 'not_conditioned'
    assert diagnostics['input_intent']['horizon_seconds'] == (4 * 8 - 4) / 30
    assert diagnostics['input_intent']['start_world_xyz_m'] == history[0, -1, :3].tolist()
    assert diagnostics['observation_root_world_xyz_m'] == history[0, -1, :3].tolist()
    assert diagnostics['input_intent']['endpoint_world_xz_m'] == goal[0, :2].tolist()

    root_path = torch.tensor(diagnostics['root_stage_world_xyz_m'])
    expected_root = root_backbone.trace[0, 4:28, :3] * torch.tensor([2.0, 1.0, 3.0])
    expected_root[:, 0] += history[0, 0, 0]
    expected_root[:, 2] += history[0, 0, 2]
    torch.testing.assert_close(root_path, expected_root)
    assert root_path[0, 0].item() == 4.5
    assert root_path[0, 2].item() == -9.0

    decoded_path = torch.tensor(diagnostics['decoded_world_xyz_m'])
    torch.testing.assert_close(decoded_path, inference.prediction[0, 4:28, :3])
    assert [patch['name'] for patch in diagnostics['terrain_patches']] == ['near', 'intent']
    near = diagnostics['terrain_patches'][0]
    assert near['shape'] == [9, 9]
    assert len(near['world_xyz_m']) == len(near['valid']) == 81
    torch.testing.assert_close(torch.tensor(near['valid']).reshape(9, 9), sampled_valid[0, 0])
    torch.testing.assert_close(
        torch.tensor(near['world_xyz_m']).reshape(9, 9, 3)[..., [0, 2]],
        sampled_observation.world_query_xz[0, 0])
    assert near['valid'][4 * 9 + 4] is False
    assert near['world_xyz_m'][4 * 9 + 4][1] == history[0, -1, 1].item()
    valid_index = near['valid'].index(True)
    sample_x, sample_y, sample_z = near['world_xyz_m'][valid_index]
    cell_x = int(torch.floor(torch.tensor((sample_x + 5.0) / 0.1 + 0.5)).item())
    cell_z = int(torch.floor(torch.tensor((sample_z + 5.0) / 0.1 + 0.5)).item())
    assert sample_y == scene['heightfield_y_m'][cell_z, cell_x].item()
    assert all(torch.isfinite(torch.tensor(point)).all() for patch in diagnostics['terrain_patches']
               for point in patch['world_xyz_m'])
    assert 'not newly observed or resampled on cached blocks' in diagnostics['observation_snapshot_note']
    json.dumps(diagnostics, allow_nan=False)

    plain_prediction, plain_count = live.predict_planar_terrain(
        inference, history, goal, scene, num_tokens=8)
    assert plain_prediction is inference.prediction
    assert plain_count is inference.count
    assert len(root_backbone._forward_hooks) == hooks_before


def test_wide_survey_is_real_but_never_passed_as_near_motion_coordinates():
    from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY
    inference=_FakeInference()
    inference._root_model.args['observation_range_policy']=INTENT_RANGE_POLICY
    inference._pose_model.args['observation_range_policy']=INTENT_RANGE_POLICY
    history=_history();goal=_goal()
    scene={'heightfield_y_m':torch.zeros(601,601),
           'min_point_world_xyz_m':[-30,0,-30],'dx_m':.1}
    observations=[]
    for extent in (3.4,20.):
        debug={}
        live.predict_planar_terrain(inference,history,goal,scene,num_tokens=8,
            query_controls=torch.tensor([[extent,6.,1.,.4]]),diagnostics=debug)
        observations.append(inference.observations[-1])
        assert debug['motion_observation_extent_m']<=3.400001
        assert debug['motion_observation_halfwidth_m']<=.600001
        survey=debug['terrain_patches'][-1]
        assert survey['name']=='survey' and survey['role']=='survey_only_not_model_input'
        assert survey['valid'].count(True)==81
        points=torch.tensor(survey['world_xyz_m'])
        assert points[:,1].abs().max()==0
        assert (points[:,[0,2]]-history[0,3,[0,2]]).norm(dim=-1).max()>extent-.01
    torch.testing.assert_close(observations[0]['points'],observations[1]['points'],atol=0,rtol=0)
    assert 'survey_points' not in observations[1]


def test_missing_root_backbone_never_fabricates_root_stage_trace():
    inference = _FakeInference(with_root_backbone=False)
    diagnostics = {}
    live.predict_planar_terrain(
        inference, _history(), _goal(), _scene(), num_tokens=8, diagnostics=diagnostics)

    assert diagnostics['root_stage_world_xyz_m'] == []
    assert len(diagnostics['decoded_world_xyz_m']) == 24


def test_backend_debug_paths_advance_with_buffer_and_replace_on_command_change(monkeypatch):
    inference = SimpleNamespace(
        device='cpu',
        global_motion_rep=SimpleNamespace(indices={'root': list(range(5))}),
    )
    history = _history()
    generations = []

    def generate(_inference, observed, _goal, _scene, *, diagnostics, **_kwargs):
        generation = len(generations)
        generations.append(observed.clone())
        prediction = torch.full((1, 32, 5), float(generation * 1000))
        path = [[float(generation * 1000 + index), 1.0, -float(index)] for index in range(24)]
        diagnostics.update({
            'version': 'soma_motion_debug_v1',
            'root_stage_world_xyz_m': [row[:] for row in path],
            'decoded_world_xyz_m': [row[:] for row in path],
            'terrain_patches': [{'name': 'near', 'world_xyz_m': [[generation, 2.0, 3.0]],
                                 'valid': [True]}],
            'observation_source': 'current_played_history',
            'observation_snapshot_note': 'Generation-time snapshot; not resampled on cached blocks.',
        })
        return prediction, torch.tensor([[8]])

    monkeypatch.setattr(live, 'predict_planar_terrain', generate)
    backend = live.NativeTerrainOnlineBackend(inference, history, {})

    first = backend.predict_plan([[0.0, 1.0]], [0.0])
    assert len(first['debug']['root_stage_world_xyz_m']) == 24
    backend.history = first['planned_motion'][:, -4:].clone()
    second = backend.predict_plan([[0.0, 1.0]], [0.0])
    assert len(second['debug']['root_stage_world_xyz_m']) == 16
    assert second['debug']['root_stage_world_xyz_m'][0][0] == 8.0
    assert second['debug']['decoded_world_xyz_m'][0][0] == 8.0
    assert second['debug']['terrain_patches'] == first['debug']['terrain_patches']
    assert second['debug']['observation_snapshot_note'] == first['debug']['observation_snapshot_note']

    backend.history = second['planned_motion'][:, -4:].clone()
    third = backend.predict_plan([[0.0, 1.0]], [0.0])
    assert len(third['debug']['root_stage_world_xyz_m']) == 8
    assert third['debug']['root_stage_world_xyz_m'][0][0] == 16.0

    changed = backend.predict_plan([[1.0, 0.0]], [0.5])
    assert len(generations) == 2
    assert changed['debug']['root_stage_world_xyz_m'][0][0] == 1000.0
    assert changed['debug']['decoded_world_xyz_m'][0][0] == 1000.0

    backend.history = backend.history + 1.0
    incompatible = backend.predict_plan([[1.0, 0.0]], [0.5])
    assert len(generations) == 3
    assert incompatible['debug']['root_stage_world_xyz_m'][0][0] == 2000.0
