import torch

from motionbricks.data.terrain_model_v3 import multiscale_observation
from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS


def test_auto_survey_speed_gain_base_and_width_change_actual_queries():
    torch.set_num_threads(2)
    history=torch.zeros(1,4,8);history[:,:,1]=1;history[:,:,3]=1
    intent=torch.tensor([[0.,2.,0.,1.6,1.,0.]])
    scene={'heightfield_y_m':torch.zeros(501,501),'min_point_world_xyz_m':[-25.,0.,-25.],'dx_m':.1}
    a=multiscale_observation(history,intent,scene,8,query_controls=torch.tensor([[-1.,.6,1.,.4]]))
    b=multiscale_observation(history,intent,scene,8,query_controls=torch.tensor([[-1.,2.,3.,1.]]))
    assert b['far_points'][...,2].max()>a['far_points'][...,2].max()+2
    assert b['far_points'][...,0].abs().max()>a['far_points'][...,0].abs().max()+1
    torch.testing.assert_close(a['points'],b['points'],atol=0,rtol=0)


def test_horizon_denoising_refinement_and_refresh_reach_real_model_and_buffer():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
    from motionbricks.data.terrain_control import SoftCommandState
    torch.set_num_threads(2)
    inference,_=online_infer()
    model=JointTerrainController(inference,{'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))})
    model.requires_grad_(False);model.eval()
    _,batch=_motion_and_representation(49)
    history=inference.global_motion_rep.unnormalize(batch['motion'][:1,:4])
    scene={'heightfield_y_m':torch.zeros(161,161),'min_point_world_xyz_m':[-8.,0.,-8.],'dx_m':.1}
    backend=JointTerrainOnlineBackend(model,history,scene)
    settings={**JOINT_RUNTIME_DEFAULTS,'num_tokens':12,'inference_steps':2,'path_refinements':2,
        'refresh_frames':24,'lookahead_m':5.,'lateral_halfwidth_m':1.1}
    backend.configure_runtime(settings)
    calls={'root':0,'pose':0}
    def count_root(*_):calls['root']+=1
    def count_pose(*_):calls['pose']+=1
    root_hook=model.root.register_forward_hook(count_root);pose_hook=model.pose.register_forward_hook(count_pose)
    # SomaLabSession supplies one batched command, including a list heading.
    with torch.inference_mode():first=backend.predict_plan([[0.,1.6]],[0.])
    root_hook.remove();pose_hook.remove()
    assert calls=={'root':3,'pose':6}
    with torch.inference_mode():
        task=SoftCommandState().intent(history,[0.,1.6],0.,12)
        reference=model(history,task,[scene],num_tokens=12,inference_steps=2,path_refinements=2,
            query_controls=history.new_tensor([[5.,1.1,1.,.4]]),packed_scene=backend.packed_scene)['world']
        torch.testing.assert_close(first['planned_motion'],reference[:,4:12],atol=0,rtol=0)
        backend.history=first['published_history']
        second=backend.predict_plan([[0.,1.6]],[0.])
        torch.testing.assert_close(second['planned_motion'],reference[:,12:20],atol=0,rtol=0)
    assert backend.last_generation['neural_generation'] is False
    assert backend.last_generation['runtime_controls']==settings
    assert backend.last_generation['remaining_frames']==8
    changed={**settings,'lookahead_m':8.}
    backend.configure_runtime(changed)
    assert backend._future_buffer is None
    with torch.inference_mode():backend.predict_plan([0.,1.6],0.)
    assert backend.last_generation['neural_generation'] is True
    assert backend.last_generation['runtime_controls']['lookahead_m']==8.


def test_input_only_refinement_has_no_fabricated_route_patch():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
    torch.set_num_threads(2)
    inference,_=online_infer()
    model=JointTerrainController(inference,{'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))})
    model.requires_grad_(False);model.eval()
    _,batch=_motion_and_representation(33)
    history=inference.global_motion_rep.unnormalize(batch['motion'][:1,:4])
    scene={'heightfield_y_m':torch.zeros(81,81),'min_point_world_xyz_m':[-4.,0.,-4.],'dx_m':.1}
    backend=JointTerrainOnlineBackend(model,history,scene)
    backend.configure_runtime({**JOINT_RUNTIME_DEFAULTS,'path_refinements':0,'inference_steps':1})
    with torch.inference_mode():result=backend.predict_plan([0.,1.6],0.)
    assert len(result['debug']['terrain_patches'])==3
    assert result['debug']['path_refinements']==0
