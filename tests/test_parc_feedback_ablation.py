import torch
import json
import numpy as np
import pytest

from scripts.review_parc_feedback_ablation import terrain_gap, mean_or_none, evaluation_windows
from scripts.train_native_mb import sha256


@pytest.mark.parametrize('offset', [0, 8, 16, 24])
def test_velocity_goal_uses_dual_rep_and_measured_world_planar_velocity(offset):
    from types import SimpleNamespace
    from test_native_mb_stage_training import _motion_and_representation
    from scripts.review_parc_feedback_ablation import evaluation_goal
    from motionbricks.data.native_mb_online_conditions import VELOCITY_PROTOCOL
    rep, batch = _motion_and_representation(65)
    infer = SimpleNamespace(motion_rep=rep, global_motion_rep=rep.dual_rep.global_motion_rep)
    assert infer.global_motion_rep.dual_rep is None
    world = rep.unnormalize(batch['motion'])
    seconds = torch.arange(65) / 30.
    world[..., 0] = 1.25 * seconds
    world[..., 2] = -.5 * seconds
    world[..., 1] += .3 * seconds.square()
    world[..., 3], world[..., 4] = .6, .8
    goal = evaluation_goal(infer, world, offset, VELOCITY_PROTOCOL['version'])
    torch.testing.assert_close(goal, torch.tensor([[1.25, -.5, .6, .8]]), atol=2e-5, rtol=0)


def test_obstacle_gap_uses_own_vertex_cell_and_retains_unknown():
    scene = {'heightfield_y_m': torch.tensor([[0., .4], [.8, float('nan')]]),
             'min_point_world_xyz_m': [0., 0., 0.], 'dx_m': 1.}
    y = torch.tensor([[0., .6, 1., 2., 0.]])
    xz = torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 0.]]])
    gap, valid = terrain_gap(y, xz, scene)
    assert valid.tolist() == [[True, True, True, False, False]]
    torch.testing.assert_close(gap[valid], torch.tensor([0., .2, .2]))
    assert mean_or_none(gap, torch.zeros_like(valid)) is None
    raised = dict(scene, heightfield_y_m=scene['heightfield_y_m'] + 3.)
    other, known = terrain_gap(y + 3., xz, raised)
    assert torch.equal(known, valid)
    torch.testing.assert_close(other[known], gap[valid])


def test_skill_windows_are_test_only_and_hash_bound(tmp_path):
    records, artifacts = [], {}
    for split in ('train', 'validation', 'test'):
        path = tmp_path / (split + '.npz')
        np.savez(path, global_features=np.zeros((80, 366), dtype=np.float32))
        artifacts[path.name] = sha256(path)
        records.append({'split': split, 'shard': path.name, 'terrain_family': 'stairs', 'source_group': split})
    accepted = tmp_path / 'accepted.json'
    accepted.write_text(json.dumps(records))
    artifacts['accepted.json'] = sha256(accepted)
    (tmp_path / 'native_manifest.json').write_text(json.dumps({'artifact_sha256': artifacts}))
    _, windows, excluded, _ = evaluation_windows(tmp_path, 'test', 8)
    assert len(windows) == 3 and not excluded
    assert all(w['shard'] == 'test.npz' and w['split'] == 'test' for w in windows)
    assert [w['start'] for w in windows] == [0, 7, 15]
    (tmp_path / 'test.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        evaluation_windows(tmp_path, 'test', 8)
