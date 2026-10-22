import torch
import pytest


def test_pure_parc_stage_precedes_inherited_leg_and_large_batch_metadata():
    from motionbricks.data.joint_terrain_live import joint_checkpoint_progress
    checkpoint={'step':86,'phase':'episode','exposure':{'window:up':160,'window:down':160,'episode:up':72,'episode:down':96},
        'contract':{'pure_parc_pilot':{'window_updates':80,'episode_updates':6,'window_batch':4},
                    'leg_head_fit':{'actual_updates':20},'joint_sdf':{'global_batch':384}}}
    result=joint_checkpoint_progress(checkpoint)
    assert result['primitive_examples']==488 and result['source_autoencoder_examples']==344
    assert result['trained_updates']==86 and result['generated_history_examples']==168


def test_live_progress_distinguishes_actual_updates_from_reference_sample_budget():
    from motionbricks.data.joint_terrain_live import joint_checkpoint_progress
    checkpoint={'step':5709,'contract':{'sample_budget':{'inherited_reference_updates':5000,
        'retained_examples':288000,'anchor_update':4000,'global_batch':768,'reference_batch':32}}}
    assert joint_checkpoint_progress(checkpoint)=={'trained_updates':10709,'optimizer_updates_in_stage':5709,
        'primitive_examples':1600512,'reference_batch':32,'reference_batch_equivalent_updates':50016.}
    assert joint_checkpoint_progress({'step':2000,'contract':{}})=={
        'trained_updates':2000,'optimizer_updates_in_stage':2000}
    checkpoint['step']=4167;checkpoint['contract']['joint_sdf']={'global_batch':384}
    stage=joint_checkpoint_progress(checkpoint)
    assert stage['optimizer_updates_in_stage']==4167 and stage['primitive_examples']==1600128
    assert stage['reference_batch_equivalent_updates']==50004
    assert stage['prior_training_reported_separately']


def test_leg_episode_counts_do_not_inherit_old_large_batch_clock():
    from motionbricks.data.joint_terrain_live import joint_checkpoint_progress
    checkpoint={'step':20,'contract':{'joint_sdf':{'global_batch':384},'leg_head_fit':{
        'actual_updates':20,'actor_examples':860,'source_autoencoder_examples':80,
        'inherited_optimizer_updates':1000,'inherited_fit_operations':1}}}
    stage=joint_checkpoint_progress(checkpoint)
    assert stage['primitive_examples']==860 and stage['optimizer_updates_in_stage']==20
    assert stage['reference_batch_equivalent_updates']==26.875
    assert stage['inherited_fit_operations']==1 and stage['prior_training_reported_separately']


def test_live_backend_emits_exact_model_prefix_and_uses_shared_rolling_controls():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    from motionbricks.data.terrain_control import SoftCommandState
    from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        inference,_=online_infer()
        model=JointTerrainController(inference,{'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))})
        model.requires_grad_(False);model.eval()
        _,batch=_motion_and_representation(33)
        history=inference.global_motion_rep.unnormalize(batch['motion'][:1,:4])
        scene={'heightfield_y_m':torch.zeros(81,81),'min_point_world_xyz_m':[-4.,0.,-4.],'dx_m':.1}
        backend=JointTerrainOnlineBackend(model,history,scene)
        with torch.inference_mode():
            controls=SoftCommandState().intent(history,[0.,1.6],0.,8)
            reference=model(history,controls,[scene],packed_scene=backend.packed_scene)['world']
            result=backend.predict_plan([0.,1.6],0.)
        torch.testing.assert_close(result['planned_motion'],reference[:,4:12],atol=0,rtol=0)
        assert result['debug']['input_intent']['velocity_world_xz_mps']==[0.,torch.tensor(1.6).item()]
        assert backend._future_buffer is None
        assert result['published_history'].shape==history.shape
        assert len(result['debug']['terrain_patches'])==4
        assert len(result['debug']['decoded_world_xyz_m'])==24
    finally:torch.set_num_threads(previous)


def test_committed_prefetch_is_identical_to_serial_and_waits_for_playback_commit():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        infer,_=online_infer()
        model=JointTerrainController(infer,{'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))})
        model.requires_grad_(False);model.eval()
        _,batch=_motion_and_representation(33)
        initial=infer.global_motion_rep.unnormalize(batch['motion'][:1,:4])
        scene={'heightfield_y_m':torch.zeros(81,81),'min_point_world_xyz_m':[-4.,0.,-4.],'dx_m':.1}
        serial=JointTerrainOnlineBackend(model,initial,scene)
        ahead=JointTerrainOnlineBackend(model,initial,scene);ahead.committed_history_prefetch=True
        first=serial.predict_plan([0.,1.],0.)
        published=ahead.predict_plan([0.,1.],0.)
        torch.testing.assert_close(first['planned_motion'],published['planned_motion'],atol=0,rtol=0)
        serial.history=first['published_history'].clone()
        expected=serial.predict_plan([.4,1.2],.2)
        prior_goal=ahead.command_state.stop_goal.clone()
        candidate=ahead.prepare_next_plan([.4,1.2],.2,initial,published['published_history'],overlap_frames=8)
        torch.testing.assert_close(ahead.history,initial,atol=0,rtol=0)
        torch.testing.assert_close(ahead.command_state.stop_goal,prior_goal,atol=0,rtol=0)
        torch.testing.assert_close(candidate['planned_motion'],expected['planned_motion'],atol=0,rtol=0)
        with pytest.raises(ValueError,match='actual playback'):
            ahead.commit_prepared_plan(candidate)
        ahead.history=published['published_history'].clone()
        ahead.commit_prepared_plan(candidate)
        ahead.history=candidate['published_history'].clone();serial.history=expected['published_history'].clone()
        actual=ahead.predict_plan([.4,1.2],.2)
        third=serial.predict_plan([.4,1.2],.2)
        torch.testing.assert_close(actual['planned_motion'],third['planned_motion'],atol=0,rtol=0)
    finally:torch.set_num_threads(old_threads)


def test_explicit_new_stage_metadata_overrides_parent_training_clock():
    from motionbricks.data.joint_terrain_live import joint_checkpoint_progress
    result=joint_checkpoint_progress({'step':8,'contract':{'pure_parc_pilot':{},'leg_head_fit':{},
        'stair_training_stage':{'schema':'motionbricks.training_stage.v1','name':'continuity',
        'optimizer_updates':8,'actor_primitives':224,'source_autoencoder_examples':32,'inherited_optimizer_updates':86}}})
    assert result['primitive_examples']==224 and result['inherited_optimizer_updates']==86
