import io

import torch

from motionbricks.data.terrain_learning_episode import TerrainLearningEpisode


def test_mid_rollout_checkpoint_restores_draws_source_clock_actual_history_and_commands():
    from test_native_mb_stage_training import _motion_and_representation
    rep,batch=_motion_and_representation(129)
    world=rep.unnormalize(batch['motion'][:1])[0]
    world[:,2]=torch.arange(129)/30
    scene={'heightfield_y_m':torch.zeros(101,101),'min_point_world_xyz_m':torch.tensor([-5.,0.,-5.]),'dx_m':torch.tensor(.1)}
    class Data:
        def __getitem__(self,request):
            index,draw=request
            return {'world':world.clone(),'scene':scene,'row':{'control_category':'start','source_id':str(index)},
                'override':None,'sampling_index':index,'sampling_draw':draw}
    data=Data();items=[data[(3,19)],data[(8,21)]]
    episode=TerrainLearningEpisode(items,[11,7],32,5,rep,'cpu',yaw=[1.1,-.7],far=7.,width=1.3)
    episode.position=2
    history=episode.source[:,16:20].clone();history[:,:,0]+=.017
    episode.history=history
    episode.command_state.intent(history,[[0.,1.6],[1.,0.]],torch.tensor([.1,.2]),8)
    with io.BytesIO() as buffer:
        torch.save(episode.state_dict(),buffer);buffer.seek(0)
        stored=torch.load(buffer,weights_only=False)
    restored=TerrainLearningEpisode.restore(stored,data,rep,'cpu')
    assert restored.position==2 and restored.starts==[11,7]
    assert restored.far==7. and restored.width==1.3
    assert stored['requests']==[(3,19),(8,21)]
    for key in ('history','source','physical_source'):
        torch.testing.assert_close(getattr(restored,key),getattr(episode,key),atol=0,rtol=0)
    assert torch.equal(restored.command_state.stop_goal,episode.command_state.stop_goal)
    for left,right in zip(episode.window(False)[:3],restored.window(False)[:3]):
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    # Restarting from a source prefix would erase this factual generated offset.
    assert not torch.equal(restored.history,restored.source[:,16:20])


def test_unknown_generated_history_restarts_training_at_same_source_time_only():
    from test_native_mb_stage_training import _motion_and_representation
    rep,batch=_motion_and_representation(65)
    world=rep.unnormalize(batch['motion'][:1])[0]
    scene={'heightfield_y_m':torch.zeros(21,21),'min_point_world_xyz_m':torch.tensor([-1.,0.,-1.]),'dx_m':torch.tensor(.1)}
    item={'world':world,'scene':scene,'row':{'source_id':'a'},'sampling_index':0,'sampling_draw':0}
    episode=TerrainLearningEpisode([item],[0],32,3,rep,'cpu')
    episode.position=1
    bad=episode.history.clone();bad[:,:,0]+=10
    episode.history=bad
    _,physical,observed,resets=episode.window(False)
    assert resets==1 and torch.equal(observed,world[None,8:12])
    assert torch.equal(physical,world[None,8:41])
    assert episode.position==1  # Source time was neither skipped nor replayed.
    assert bad[0,0,0]==10  # The earlier bad prediction was not projected/clamped.
