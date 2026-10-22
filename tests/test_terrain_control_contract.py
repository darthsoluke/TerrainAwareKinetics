import pytest
import torch

from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls
from motionbricks.data.terrain_learning_data import source_task_intent


def motion():
    value=torch.zeros(2,65,8);value[:,:,1]=1.;value[:,:,3]=1.
    value[0,:,2]=-torch.arange(65)/30
    value[1,:,0]=torch.arange(65)/30
    return value


@pytest.mark.parametrize('tokens',[6,8,16])
def test_training_and_online_use_identical_explicit_requests_with_actual_history(tokens):
    source=motion()[:,:tokens*4+1];history=source[:,:4].clone()
    history[:,:,0]+=3.5;history[:,:,2]-=2.
    controls=demonstration_controls(source,tokens*4)
    train=source_task_intent(source,tokens*4,history=history)
    goal=source[:,tokens*4-1,[0,2]]
    live=SoftCommandState().intent(history,controls[:,:2],torch.atan2(controls[:,3],controls[:,2]),tokens,goal_xz=goal)
    torch.testing.assert_close(train,live,atol=1e-6,rtol=0)
    torch.testing.assert_close(train[:,:2],goal)
    assert train[0,3]<0 and train[0,4]==1  # Backwards speed never flips body facing.


def test_prediction_drift_cannot_move_the_requested_reference_goal():
    source=motion()[:1,:33];history=source[:,:4].clone()
    first=source_task_intent(source,32,history=history)
    history[:,:,0]+=1.
    second=source_task_intent(source,32,history=history)
    assert torch.equal(first,second)
    before=first[:,:2]-source[:,3,[0,2]]
    after=second[:,:2]-history[:,3,[0,2]]
    torch.testing.assert_close(after-before,torch.tensor([[-1.,0.]]))


def test_free_joystick_request_is_causal_and_does_not_enforce_its_goal():
    history=motion()[:1,:4];before=history.clone()
    task=SoftCommandState().intent(history,[0.,1.6],0.,8)
    torch.testing.assert_close(task[:,:2]-history[:,3,[0,2]],torch.tensor([[0.,1.6*28/30]]))
    assert torch.equal(history,before)
    with pytest.raises(ValueError):SoftCommandState().intent(history,[0.,1.6],0.,8,goal_xz=torch.zeros(1,3))


def test_weak_controls_roll_with_current_state_and_cannot_supply_fixed_world_goal():
    source=motion()[:1,:33];control=torch.tensor([[0.,1.6,-1.,0.]])
    state=SoftCommandState();first=source_task_intent(source,32,override=control,state=state)
    shifted=source[:,:4].clone();shifted[:,:,2]+=10
    second=source_task_intent(source,32,override=control,history=shifted,state=state)
    assert second[0,1]-first[0,1]==10
    assert torch.equal(second[:,2:],first[:,2:])
    with pytest.raises(ValueError):source_task_intent(source,32,override=torch.zeros(1,6))


def test_stop_anchor_only_updates_on_command_transition_and_is_not_an_output_override():
    history=motion()[:1,:4].clone();state=SoftCommandState()
    state.intent(history,[0,1.6],0.,8)
    history[:,:,2]+=1
    first_stop=state.intent(history,[0,0],0.,8)
    history[:,:,2]+=.2
    second_stop=state.intent(history,[0,0],0.,8)
    assert torch.equal(first_stop,second_stop)
    assert second_stop[0,1]!=history[0,-1,2]


def test_start_from_quiet_history_receives_cruise_request_before_acceleration():
    value=motion()[:1];value[:,:20,2]=0.;value[:,20:,2]=torch.arange(45)*1.6/30
    control=demonstration_controls(value[:,:33],32,category='start',reference=value)
    assert control[0,1]>1.5
    assert torch.count_nonzero(value[:,:4,2])==0


def test_control_construction_has_no_label_gradients_or_future_height_dependency():
    source=motion().requires_grad_()
    before=source_task_intent(source,64)
    changed=source.detach().clone();changed[:,4:,1]+=100
    changed[:,4:,5:]+=100
    after=source_task_intent(changed,64)
    assert torch.equal(before,after) and not before.requires_grad


def test_clip_level_stop_does_not_label_its_cruising_approach_as_zero_command():
    source=motion()[:1,:33]
    control=demonstration_controls(source,32,category='stop')
    torch.testing.assert_close(control[0,:2],torch.tensor([0.,-1.]))
    stopped=source.clone();stopped[:,24:,2]=stopped[:,24:25,2]
    control=demonstration_controls(stopped,32,category='stop')
    assert torch.equal(control[0,:2],torch.zeros(2))
