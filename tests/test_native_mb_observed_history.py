import torch

from test_native_mb_stage_training import _motion_and_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history


def test_fk_observation_repairs_redundant_positions_without_changing_displayed_motion():
    rep, batch = _motion_and_representation(frame_count=25)
    history = rep.unnormalize(batch['motion'][:, :4]).clone()
    history = rep.change_first_heading(history, torch.tensor([.7]), False, False)
    history[..., rep.indices['global_root_pos_2d']] += history.new_tensor([3., -2.])
    history[..., rep.indices['global_root_pos'][1]] += 4.
    # These decoder channels are independent predictions, not authoritative FK.
    history[..., rep.indices['ric_data'][1::3]] -= 2.
    history[..., rep.indices['local_vel']] = 17.
    original = history.clone()
    displayed = rep.inverse(history, False, return_all=False, joint_positions_from='global_rot_data')
    info = {}
    observed = reencode_played_history(history, rep, info)
    torch.testing.assert_close(history, original, atol=0, rtol=0)
    from_rotations = rep.inverse(observed, False, return_all=False, joint_positions_from='global_rot_data')
    from_positions = rep.inverse(observed, False, return_all=False, joint_positions_from='ric_data')
    torch.testing.assert_close(from_rotations['posed_joints'], displayed['posed_joints'], atol=2e-5, rtol=0)
    torch.testing.assert_close(from_positions['posed_joints'], displayed['posed_joints'], atol=2e-5, rtol=0)
    torch.testing.assert_close(observed[..., rep.indices['global_root_pos']],
                               history[..., rep.indices['global_root_pos']], atol=2e-5, rtol=0)
    assert info['original_ric_fk_mean_error_m'] > .5
    assert info['physical_state_reencoding_max_error_m'] < 2e-5
    assert not torch.equal(observed[..., rep.indices['local_vel']], original[..., rep.indices['local_vel']])
