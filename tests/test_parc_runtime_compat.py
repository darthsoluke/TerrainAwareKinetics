"""Exact regressions for the local PARC vendor compatibility patch."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1] / 'out/vendor/PARC'
if not REPO.is_dir():
    pytest.skip('official PARC checkout is required', allow_module_level=True)
pytest.importorskip('trimesh')
pytest.importorskip('wandb')
sys.path.insert(0, str(REPO))
sys.modules.setdefault('parc', importlib.import_module('PARC'))
from parc.motion_synthesis.procgen import mdm_path
from parc.motion_generator import gen_util
from parc.motion_generator.diffusion_util import MDMKeyType
from parc.util.motion_util import MotionFrames


def frames(batch, start, count):
    positions = torch.zeros(batch, count, 3)
    positions[:, :, 0] = torch.arange(start, start + count)
    rotations = torch.zeros(batch, count, 4)
    rotations[..., 3] = 1
    return MotionFrames(root_pos=positions, root_rot=rotations,
                        joint_rot=rotations[:, :, None].clone(),
                        contacts=torch.zeros(batch, count, 2))


def completion_run(monkeypatch, done_schedule, max_seconds=10.):
    batch = len(done_schedule[0])
    terrain = SimpleNamespace(torch_copy=lambda: terrain, set_device=lambda device: None)
    char = SimpleNamespace(_device=torch.device('cpu'))
    model = SimpleNamespace(_device='cpu', _seq_len=15, _num_prev_states=2, _sequence_fps=30)
    settings = mdm_path.MDMPathSettings()
    settings.mdm_batch_size, settings.max_motion_length = batch, max_seconds
    monkeypatch.setattr(mdm_path.geom_util, 'get_char_point_samples', lambda _: [])
    monkeypatch.setattr(mdm_path, 'gen_mdm_motion_at_path_start', lambda **kwargs: frames(batch, 0, 15))
    remaining = iter(done_schedule)

    def generate(**kwargs):
        start = int(kwargs['prev_frames'].root_pos[0, 0, 0])
        return frames(batch, start, 15), torch.tensor(next(remaining), dtype=torch.bool)

    monkeypatch.setattr(mdm_path, 'generate_frames_along_path', generate)
    monkeypatch.setattr(mdm_path, 'compute_motion_loss', lambda *args: {
        'total_loss': torch.zeros(1), 'contact_loss': torch.zeros(1), 'pen_loss': torch.zeros(1)})
    return mdm_path.generate_frames_until_end_of_path(torch.zeros(2, 3), terrain, char,
        model, None, settings, gen_util.MDMGenSettings(), False, False, False)


def test_all_done_before_break_preserves_history_and_no_failure_penalty(monkeypatch):
    motions, _, info = completion_run(monkeypatch, [[True, True]])
    assert info['completed'].tolist() == [True, True]
    assert info['losses'].tolist() == [0., 0.]
    assert info['frame_counts'].tolist() == [9, 9]
    for motion in motions:
        torch.testing.assert_close(motion.root_pos[0, :, 0], torch.arange(9).float())


def test_staggered_done_stops_after_each_candidate_has_reached(monkeypatch):
    motions, _, info = completion_run(monkeypatch, [[True, False], [False, True]])
    lengths = {int(idx): int(count) for idx, count in zip(info['candidate_ids'], info['frame_counts'])}
    assert lengths == {0: 9, 1: 16}
    assert info['completed'].all()
    for motion in motions:
        torch.testing.assert_close(motion.root_pos[0, :, 0], torch.arange(motion.root_pos.shape[1]).float())


def test_timeout_retains_last_frame_and_has_one_dimensional_scores(monkeypatch):
    motions, _, info = completion_run(monkeypatch, [[False]], max_seconds=.3)
    assert info['completed'].tolist() == [False]
    assert info['losses'].shape == (1,)
    assert info['losses'].item() == 100.
    assert info['frame_counts'].tolist() == [22]
    assert motions[0].root_pos[0, -1, 0].item() == 21.


def test_real_batched_cfg_runs_and_matches_batch_one(monkeypatch):
    scripts = REPO.parents[2] / 'scripts'
    sys.path.insert(0, str(scripts))
    try:
        import run_parc_local_terrain_pilot as pilot
        import run_parc_official_correction as correction
    finally:
        sys.path.pop(0)
    if not pilot.DEFAULT_CHECKPOINT.is_file():
        pytest.skip('released small checkpoint is required')
    torch.set_num_threads(4)
    model = pilot.load_official_model(REPO, pilot.DEFAULT_CHECKPOINT, 'cpu')
    from parc.util import terrain_util
    terrain, nodes = pilot.build_stairs(terrain_util, 'up')
    first = mdm_path.gen_mdm_motion_at_path_start(nodes, terrain, model._kin_char_model,
        model, gen_util.MDMGenSettings(), None, 1)
    prev = first.get_slice(slice(5, 7))
    original = model.ddim_inference
    mutable_flags = []

    def fixed_noise(conds, stride, **kwargs):
        batch = conds[MDMKeyType.PREV_STATE_KEY].shape[0]
        for key in (MDMKeyType.PREV_STATE_FLAG_KEY, MDMKeyType.PREV_STATE_NOISE_IND_KEY):
            flag = conds[key]
            assert batch == 1 or flag.stride(0) != 0
            mutable_flags.append(flag.clone())
        noise = torch.zeros(batch, model._seq_len, model._motion_frame_dof)
        return original(conds, noise=noise, stride=stride)

    monkeypatch.setattr(model, 'ddim_inference', fixed_noise)
    single = gen_util.gen_mdm_motion(nodes[-1:], prev, terrain, model,
                                    model._kin_char_model, gen_util.MDMGenSettings())
    batched = gen_util.gen_mdm_motion(nodes[-1:].expand(2, -1), prev.expand_first_dim(2),
        terrain, model, model._kin_char_model, gen_util.MDMGenSettings())
    assert len(mutable_flags) == 4
    for name in ('root_pos', 'root_rot', 'joint_rot', 'contacts'):
        for row in (0, 1):
            torch.testing.assert_close(getattr(batched, name)[row], getattr(single, name)[0],
                                       atol=2e-4, rtol=2e-4)
    points = mdm_path.geom_util.get_char_point_samples(model._kin_char_model)
    direct = mdm_path.compute_motion_loss(batched, None, terrain, model._kin_char_model,
                                         points, .1, .1, 1., verbose=False)
    config = {'mdm_path': {'w_contact': .1, 'w_pen': .1}}
    split = correction.score_candidate_batches(batched, terrain, model._kin_char_model,
                                                points, config, batch_size=1)
    for key in direct:
        torch.testing.assert_close(split[key], direct[key], atol=2e-5, rtol=2e-5)
