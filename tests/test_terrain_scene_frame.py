import pytest
import torch

from motionbricks.data.terrain_scene_frame import HorizontalSceneFrame,known_history_centers
from motionbricks.data.terrain_model_v3 import multiscale_observation,pack_scene_chunks


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_rigid_frame_preserves_fk_and_roundtrip_and_prediction_gradient():
    from test_native_mb_stage_training import _motion_and_representation
    rep,batch=_motion_and_representation(33)
    world=rep.unnormalize(batch['motion'][:1]).clone()
    frame=HorizontalSceneFrame(torch.tensor([1.13]),torch.tensor([[.17,-.23]]))
    rotated=frame.motion(rep,world)
    roundtrip=frame.motion(rep,rotated,inverse=True)
    torch.testing.assert_close(roundtrip,world,atol=2e-5,rtol=1e-5)
    joints=rep.inverse(world,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
    transformed=rep.inverse(rotated,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
    expected=joints.clone();expected[...,[0,2]]=frame.points(joints[...,[0,2]])
    torch.testing.assert_close(transformed,expected,atol=2e-5,rtol=1e-5)
    predicted=rotated.clone().requires_grad_()
    physical=frame.motion(rep,predicted,inverse=True)
    physical[...,:3].square().mean().backward()
    assert predicted.grad.isfinite().all() and predicted.grad[...,:3].abs().sum()>0


def test_observed_nonflat_scene_does_not_rotate_away_from_the_motion():
    from test_native_mb_stage_training import _motion_and_representation
    rep,batch=_motion_and_representation(33)
    history=rep.unnormalize(batch['motion'][:1,:4]).clone()
    history[:,:,0]+=.123;history[:,:,2]+=.237
    task=torch.tensor([[.7,2.,.3,1.4,1.,0.]])
    hf=torch.zeros(161,161);hf[85:90,70:90]=.5;hf[90:95,70:90]=1.
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-8.,0.,-8.],'dx_m':.1}
    before=multiscale_observation(history,task,scene,8,far_distance=3.4)
    frame=HorizontalSceneFrame(torch.tensor([1.57]),torch.tensor([[.11,-.31]]))
    augmented_history=frame.motion(rep,history);augmented_task=frame.task(task)
    after=multiscale_observation(frame.motion(rep,augmented_history,inverse=True),
        frame.task(augmented_task,inverse=True),scene,8,far_distance=3.4)
    for key in ('valid','far_valid'):assert torch.equal(before[key],after[key])
    for key in ('points','far_points','far_min_y'):
        torch.testing.assert_close(before[key],after[key],atol=2e-4,rtol=1e-5)
    assert torch.equal(scene['heightfield_y_m'],hf)


def test_batch_frames_rotate_velocities_and_requests_together():
    frame=HorizontalSceneFrame(torch.tensor([torch.pi/2,-torch.pi/2]),torch.zeros(2,2))
    task=torch.tensor([[0.,2.,0.,1.,1.,0.],[0.,2.,0.,1.,1.,0.]])
    changed=frame.task(task)
    torch.testing.assert_close(changed[:,:2],torch.tensor([[2.,0.],[-2.,0.]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(changed[:,2:4],torch.tensor([[1.,0.],[-1.,0.]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(frame.task(changed,inverse=True),task,atol=1e-6,rtol=0)


def test_history_validity_uses_real_map_cells_and_matches_the_observation():
    history=torch.zeros(3,4,8);history[:,:,1]=1.;history[:,:,3]=1.
    history[1,:,0]=10.;history[2,:,2]=.1
    hf=torch.zeros(21,21);hf[11,10]=float('nan')
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-1.,0.,-1.],'dx_m':.1}
    scenes=[scene]*3;packed=pack_scene_chunks(scenes,'cpu')
    known=known_history_centers(history,packed)
    observed=multiscale_observation(history,torch.tensor([[0.,1.,0.,1.,1.,0.]]).expand(3,-1),scenes,8,packed=packed)
    assert known.tolist()==[True,False,False]
    assert torch.equal(known,observed['valid'][:,0,4,4])


def test_episode_distance_crop_preserves_interior_surface_values_and_gradients():
    from collections import OrderedDict
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective
    objective=JointGeometryObjective.__new__(JointGeometryObjective)
    objective.volumes=OrderedDict();objective.max_volumes=8
    hf=torch.zeros(161,161);hf[65:95,80:83]=2.
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-8.,0.,-8.],'dx_m':.1}
    short=torch.zeros(1,33,8);episode=torch.zeros(1,129,8)
    episode[0,:,2]=torch.linspace(0,3,129)
    points=torch.tensor([[-.01,.8,0.],[.6,.3,.3],[-.6,.8,1.]],requires_grad=True)
    a,ka=objective.volume(scene,short,'short').query(points)
    b,kb=objective.volume(scene,episode,'episode').query(points)
    assert torch.equal(ka,kb) and ka.all()
    torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-5)
    ga=torch.autograd.grad(a.sum(),points,retain_graph=True)[0]
    gb=torch.autograd.grad(b.sum(),points)[0]
    torch.testing.assert_close(ga,gb,atol=2e-5,rtol=1e-5)
