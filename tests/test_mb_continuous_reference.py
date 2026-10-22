from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest

from scripts.isolate_mb_continuous_reference import reference_from_poses
from scripts.review_mb_continuous_reference import root_metrics
from test_mb_trajectory_reference import kinematics, poses
from test_mb_sonic_tracker_wiring import initialized, scene_model
from motionbricks.mb_sonic.runtime import PhysicsRecorder, _numeric_snapshot


def test_baked_reference_preserves_poses_and_has_consistent_world_velocity(kinematics, monkeypatch):
    monkeypatch.setattr(mujoco, 'mj_step', lambda *args: pytest.fail('Reference construction advanced physics'))
    qpos = poses(51)
    times = np.arange(51)/50
    qpos[:, 0] += times*.5
    source = SimpleNamespace(scene_id='flat', command_id='fixed', provenance={})
    adapter = SimpleNamespace(kinematics=kinematics, calibration={})
    reference = reference_from_poses(adapter, times, qpos, source, 'test_sha')
    np.testing.assert_array_equal(reference.qpos, qpos)
    np.testing.assert_allclose(reference.qvel[:, :3], np.tile([.5, 0., 0.], (51, 1)), atol=1e-12)
    assert reference.provenance['reference_mode'].endswith('no_online_switches')
    assert not reference.qpos.flags.writeable


def test_initial_and_warmup_physics_snapshots_are_losslessly_json_recordable(scene_model, tmp_path):
    physics, _, _ = initialized(scene_model)
    path = tmp_path/'initial.jsonl'
    snapshot = physics.snapshot()
    with PhysicsRecorder(path, {'schema': 'test'}) as recorder:
        recorder.write(dict(kind='explicit_initial_reset', actual=_numeric_snapshot(snapshot)))
    actual = json.loads(path.read_text().splitlines()[-1])['actual']
    np.testing.assert_array_equal(actual['qpos'], snapshot.qpos)
    assert actual['state_version'] == 0


def test_straight_root_has_correct_progress_without_fabricated_high_frequency_motion():
    times = np.arange(481)/30
    root = np.c_[np.zeros(481), -1.2*times, np.ones(481)]
    report = root_metrics(times, root)
    assert report['progress_negative_y_m'] == pytest.approx(19.2)
    assert report['mean_forward_mps'] == pytest.approx(1.2)
    assert report['spectrum']['energy_8_14'] < 1e-20
