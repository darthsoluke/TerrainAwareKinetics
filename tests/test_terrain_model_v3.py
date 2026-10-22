import pytest
import torch

from motionbricks.data.terrain_model_v3 import FarSceneResidual,straight_through_codes
from motionbricks.data.native_mb_online_conditions import sparse_soft_intent_conditions,SOFT_INTENT_PROTOCOL,validate_online_inputs
from motionbricks.data.terrain_distance_volume import TerrainDistanceVolume,swept_surface_points


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_soft_intent_has_no_known_future_motion_and_cannot_override_root():
    root=torch.zeros(2,4,5);root[...,3]=1
    velocity=torch.zeros(2,4,4);pose=torch.zeros(2,4,10)
    c=sparse_soft_intent_conditions(root,velocity,pose)
    validate_online_inputs(c['global_root_values'],c['has_global_root_values'],c['global_root_component_mask'],
        c['local_root_values'],c['has_local_root_values'],c['local_poses'],c['has_local_poses'],{},
        version=SOFT_INTENT_PROTOCOL['version'],local_component_mask=c['local_root_component_mask'])
    assert not c['global_root_component_mask'][:,4:].any()
    assert not c['local_root_component_mask'][:,3:].any()
    c['global_root_component_mask'][:,7,1]=True;c['has_global_root_values']=c['global_root_component_mask'].any(-1)
    with pytest.raises(ValueError):
        validate_online_inputs(c['global_root_values'],c['has_global_root_values'],c['global_root_component_mask'],
            c['local_root_values'],c['has_local_root_values'],c['local_poses'],c['has_local_poses'],{},
            version=SOFT_INTENT_PROTOCOL['version'],local_component_mask=c['local_root_component_mask'])


def test_soft_code_gradient_keeps_the_actual_hard_forward_codeword():
    torch.manual_seed(1)
    book=torch.randn(2,8,4)
    logits=torch.randn(3,6,2,8,requires_grad=True)
    ids=logits.detach().argmax(-1)
    hard=torch.stack([book[h][ids[:,:,h]] for h in range(2)],2).flatten(-2)
    value=straight_through_codes(logits,hard,book)
    assert torch.equal(value,hard)
    value.square().mean().backward()
    assert logits.grad.isfinite().all() and logits.grad.abs().sum()>0


def test_learned_far_residual_is_flat_invariant_and_nonflat_sensitive():
    torch.manual_seed(3)
    module=FarSceneResidual(16);module.gate.data.fill_(.7)
    query=torch.randn(2,6,16)
    near=torch.zeros(2,2,9,9,3);near[...,1]=-1
    for extent in (1.,3.4,20.):
        far=torch.randn(2,9,9,3)*extent;far[...,1]=-1
        obs={'points':near,'far_points':far,'far_valid':torch.ones(2,9,9,dtype=torch.bool),
             'speed_mps':torch.ones(2),'horizon_seconds':torch.ones(2)}
        assert torch.equal(module(query,obs),query)
    obs['far_points']=far.clone();obs['far_points'][:,4:,:,1]+=.8;obs.pop('_far_cache',None)
    output=module(query,obs)
    assert (output-query).abs().max()>1e-5
    output.square().mean().backward()
    assert module.encoder[0].weight.grad.abs().sum()>0


def test_distance_volume_wall_gradient_and_swept_collision():
    hf=torch.zeros(41,41);hf[:,20:22]=2
    volume=TerrainDistanceVolume({'heightfield_y_m':hf,'min_point_world_xyz_m':[-2,0,-2],'dx_m':.1})
    p=torch.tensor([[-.01,.8,0.]],requires_grad=True)
    gap,valid=volume.query(p)
    assert valid.all() and gap.item()<-.02
    (-gap.sum()).backward();assert p.grad[0,0]>0
    endpoints=torch.tensor([[[-.3,.8,0.]],[[.4,.8,0.]]])
    before,_=volume.query(endpoints)
    during,_=volume.query(swept_surface_points(endpoints,8))
    assert before.min()>0 and during.min()<0


@pytest.mark.parametrize('extent',[3.4,12.,20.])
def test_far_scene_sees_thin_wall_and_pit_between_sparse_query_points(extent):
    from motionbricks.data.terrain_model_v3 import multiscale_observation
    history=torch.zeros(1,4,8);history[:,:,1]=1;history[:,:,3]=1
    intent=torch.tensor([[0.,2.,0.,1.,1.,0.]])
    hf=torch.zeros(501,501);hf[267:269]=2.;hf[275:277]=-1.
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-25.,0.,-25.],'dx_m':.1}
    observed=multiscale_observation(history,intent,scene,8,far_distance=extent)
    assert observed['far_points'][...,1].max()==1.
    assert observed['far_min_y'].min()==-2.


def test_refinement_cannot_forget_a_wall_observed_along_the_input_intention():
    from motionbricks.data.terrain_model_v3 import retain_intent_observation
    initial={'far_points':torch.zeros(1,9,9,3),'far_valid':torch.ones(1,9,9,dtype=torch.bool),
             'far_min_y':torch.zeros(1,9,9)}
    initial['far_points'][:,4,:,1]=2.
    refined={key:value.clone() for key,value in initial.items()}
    refined['far_points'][...,1]=0
    combined=retain_intent_observation(initial,refined)
    assert combined['far_points'].shape==(1,18,9,3)
    assert combined['far_points'][...,1].max()==2.
    assert torch.equal(combined['far_points'][:,:9],initial['far_points'])
    assert torch.equal(combined['far_points'][:,9:],refined['far_points'])
    combined.update(points=torch.zeros(1,2,9,9,3),speed_mps=torch.ones(1),horizon_seconds=torch.ones(1))
    network=FarSceneResidual(16);network.gate.data.fill_(1.)
    query=torch.randn(1,6,16)
    assert not torch.equal(network(query,combined),query)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_final_fk_gradients_reach_root_pose_decoder_but_not_labels_or_vocabulary(batch_size):
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    inference,_=online_infer()
    model=JointTerrainController(inference,{'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))})
    model.set_learning_phase(False)
    _,batch=_motion_and_representation(33)
    source=inference.global_motion_rep.unnormalize(batch['motion'][:1]).repeat(batch_size,1,1).requires_grad_()
    history=source[:,:4].detach()
    velocity=(source[:,31,[0,2]]-source[:,3,[0,2]])/(28/30)
    intent=torch.cat((source[:,31,[0,2]],velocity,source[:,31,3:5]),-1).detach()
    scene={'heightfield_y_m':torch.zeros(81,81),'min_point_world_xyz_m':[-4,0,-4],'dx_m':.1}
    result=model(history,intent,[scene]*batch_size,source=source,num_tokens=8,route_refinement=False,straight_through=True)
    native=model.native_losses(result,source)
    assert torch.isfinite(native['native_total'])
    predicted=inference.global_motion_rep.inverse(result['world'][:,4:12],False,
        joint_positions_from='global_rot_data',return_all=False)['posed_joints']
    truth=inference.global_motion_rep.inverse(source.detach()[:,4:12],False,
        joint_positions_from='global_rot_data',return_all=False)['posed_joints']
    (predicted-truth).square().mean().backward()
    assert model.pose._proj_pose_output_logit.weight.grad.abs().sum()>0
    assert model.codec.decoder.model[-1].weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.root.parameters())
    assert model.control[-1].weight.grad.abs().sum()>0
    assert source.grad is None
    assert all(p.grad is None and not p.requires_grad for p in model.codec.encoder.parameters())


def test_adapter_warmup_trains_new_height_columns_and_keeps_pretrained_columns_exact():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    inference,_=online_infer()
    model=JointTerrainController(inference,{'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))})
    model.set_learning_phase(True)
    linear=model.codec.decoder.external_cond_blocks[0]
    before=linear.weight.detach().clone()
    x=torch.randn(2,4,linear.in_features)
    linear(x).square().mean().backward()
    # The copied hidden/planar weights are nonzero; all new height columns start at zero.
    old=before.abs().sum(0)>0
    assert linear.weight.grad[:,old].count_nonzero()==0
    assert linear.weight.grad[:,~old].abs().sum()>0
    torch.optim.Adam([linear.weight],lr=1e-3).step()
    assert torch.equal(linear.weight[:,old],before[:,old])
    assert not torch.equal(linear.weight[:,~old],before[:,~old])
    assert not model.codec.decoder.model[-1].weight.requires_grad
    assert not model.pose._proj_pose_output_logit.weight.requires_grad


def test_training_actually_reaches_weak_pair_stop_tail_and_keeps_source_clock():
    from motionbricks.data.terrain_learning_data import sample_aligned_world_windows
    class Draws:
        def random(self):return 0.
        def integers(self,maximum):return min(11,maximum-1)
    world=torch.arange(129).float()[:,None].expand(-1,7).clone()
    items=[{'world':world,'override':torch.ones(6)}, {'world':world+1000,'override':None}]
    window,starts,count=sample_aligned_world_windows(items,41,Draws())
    assert starts==[88,11] and count==1
    assert window[0,-1,0]==128  # The old prefix never reached these stopping frames.
    assert torch.equal(window[0,8:41],world[96:129])
    assert torch.equal(window[1,8:41],items[1]['world'][19:52])
    assert torch.equal(world[:,0],torch.arange(129).float())


def test_contact_loss_does_not_teach_a_misplaced_foot_to_climb_a_wall():
    from collections import OrderedDict
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective
    objective=JointGeometryObjective.__new__(JointGeometryObjective)
    objective.sole_count=1;objective.volumes=OrderedDict();objective.max_volumes=4
    objective.geometry=lambda motion:(motion[...,:3,None].transpose(-1,-2),motion[...,:3,None].transpose(-1,-2))
    source=torch.zeros(1,16,8);source[:,:,0]=-.3
    prediction=source.clone();prediction[:,:,0]=.05;prediction[:,:,1]=.1;prediction.requires_grad_()
    hf=torch.zeros(41,41);hf[:,20:23]=2.
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-2.,0.,-2.],'dx_m':.1}
    loss=objective.losses(prediction,source,source[:,:4],[scene],['wall-correspondence'])
    assert loss['collision_m2']>0
    loss['support_gap_m'].backward()
    assert prediction.grad[...,1].sum()>0  # Descend toward source support, not the two-meter wall top.
    assert prediction.grad[...,0].count_nonzero()==0


def test_source_penetration_is_reported_and_does_not_license_predicted_penetration():
    from collections import OrderedDict
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective
    objective=JointGeometryObjective.__new__(JointGeometryObjective)
    objective.sole_count=1;objective.volumes=OrderedDict();objective.max_volumes=4
    objective.geometry=lambda motion:(motion[...,:3,None].transpose(-1,-2),motion[...,:3,None].transpose(-1,-2))
    source=torch.zeros(1,16,8);source[:,:,1]=-.1
    prediction=source.clone().requires_grad_()
    scene={'heightfield_y_m':torch.zeros(41,41),'min_point_world_xyz_m':[-2.,0.,-2.],'dx_m':.1}
    loss=objective.losses(prediction,source,source[:,:4],[scene],['source-residual'])
    assert loss['source_collision_p99_m']>=.099
    assert loss['collision_m2']>0
    loss['collision_m2'].backward()
    assert prediction.grad[...,1].sum()<0
