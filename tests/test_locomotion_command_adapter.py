import pytest
import torch
from types import SimpleNamespace

from motionbricks.data.locomotion_command_adapter import LocomotionCommandCondition, desired_motion_features, install_locomotion_command
from motionbricks.data.bones_controller_episode import episode_controls, rotate_controls


def test_new_command_path_is_exact_identity_before_training():
    module = LocomotionCommandCondition(12)
    query = torch.randn(2, 8, 12)
    control = torch.tensor([[4., -3., 0., 1., 1., 0., 28/30], [2., 7., 0., 0., 1., 0., 28/30]])
    torch.testing.assert_close(module(query, control), query, atol=0, rtol=0)
    module(query, control).square().mean().backward()
    assert module.update[-1].weight.grad.norm() > 0


def test_stop_style_never_uses_the_stale_release_position():
    control = torch.tensor([[0., 0., 0., 0., 1., 0., 28/30]])
    shifted = control.clone(); shifted[:, :2] = torch.tensor([100., -100.])
    assert torch.equal(desired_motion_features(control), desired_motion_features(shifted))
    assert desired_motion_features(control)[0, -1] == 0
    shifted[:, 3] = .5
    assert desired_motion_features(shifted)[0, -1] == 1
    with pytest.raises(ValueError): desired_motion_features(torch.zeros(1, 8))


def test_episode_schedule_reads_future_xz_not_future_body_or_height():
    world = torch.zeros(97, 12); world[:, 1] = 1; world[:, 3] = 1
    world[16:, 2] = torch.arange(81)/30
    original = episode_controls(world, 'start', 9)
    changed = world.clone(); changed[4:, 1] += 30; changed[4:, 3:] = torch.randn(93, 9)
    assert torch.equal(original, episode_controls(changed, 'start', 9))
    assert original[:, 1].min() > .9
    with pytest.raises(ValueError): episode_controls(world[:96], 'start', 9)


def test_rigid_rotation_preserves_speed_and_facing_offset():
    value = torch.tensor([[0., 1.6, 1., 0.], [0., 0., 1., 0.]])
    rotated = rotate_controls(value, torch.pi/2)
    torch.testing.assert_close(rotated, torch.tensor([[1.6, 0., 0., 1.], [0., 0., 0., 1.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotate_controls(rotated, -torch.pi/2), value, atol=1e-6, rtol=0)


def test_start_intent_does_not_use_unrelated_later_source_travel():
    world=torch.zeros(97,12);world[:,1]=1;world[:,3]=1
    world[16:,0]=torch.arange(81)*.05
    reference=torch.zeros(300,12);reference[:97]=world;reference[-1,2]=-100
    actual=episode_controls(world,'start',9,reference)
    torch.testing.assert_close(actual,episode_controls(world,'start',9),atol=0,rtol=0)
    assert (actual[:,0]>1).all() and torch.count_nonzero(actual[:,1])==0
    torch.testing.assert_close(actual[:,2:],torch.tensor([[0.,1.]]).expand(9,2),atol=1e-6,rtol=0)


def test_hooks_use_latest_command_at_both_stages_without_changing_base():
    model = torch.nn.Module()
    model.inference = SimpleNamespace(device='cpu')
    model.control = torch.nn.Linear(7, 8)
    model.pose = torch.nn.Module(); model.pose._args = {'n_embd':8}
    model.pose._transformer_model = torch.nn.Identity()
    model.codec = torch.nn.Module(); model.codec.code_dim = 8
    model.codec.decoder = torch.nn.Identity()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    install_locomotion_command(model)
    with pytest.raises(ValueError): install_locomotion_command(model)
    for module in (model.locomotion_pose_condition,model.locomotion_decoder_condition):
        with torch.no_grad():
            module.update[0].weight.zero_(); module.update[0].bias.zero_()
            module.update[0].weight[0,-1] = 1
            module.update[-1].weight[0,0] = 1
    query = torch.zeros(1,4,8)
    stopped = torch.tensor([[0.,0.,0.,0.,1.,0.,28/30]])
    model.control(stopped)
    assert torch.count_nonzero(model.pose._transformer_model(query)) == 0
    move = stopped.clone(); move[:,3] = 1
    model.control(move)
    pose = model.pose._transformer_model(query)
    decoder = model.codec.decoder(query.transpose(1,2)).transpose(1,2)
    assert (pose[:,:,0] > .8).all()
    torch.testing.assert_close(pose,decoder)
    for key,value in before.items(): assert torch.equal(model.state_dict()[key],value)
