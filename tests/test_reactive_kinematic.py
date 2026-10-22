import numpy as np
import pytest

from motionbricks.data.reactive_kinematic import ReactivePlayback


def intent(seq, velocity=(0.,1.)):
    return dict(seq=seq,velocity_xz=velocity,heading=.3,client_input_ms=seq*10.)


def forecast(start):
    return np.arange(start,start+24,dtype=np.float32)[:,None]*np.ones((1,6),np.float32)


def started(step=4):
    state=ReactivePlayback(np.zeros((4,6),np.float32),step_frames=step)
    state.receive_intent(intent(1),1)
    state.started=True
    return state


def publish(state, request, motion):
    plan=state.publish(request,motion)
    state.accept(state.episode_id,plan,request.origin_frame+1)
    return plan


def test_publishing_future_never_advances_played_history():
    state=started()
    request=state.capture()
    original=state.history.copy()
    plan=publish(state,request,forecast(1))
    np.testing.assert_array_equal(state.history,original)
    state.acknowledge(1,1,plan)
    np.testing.assert_array_equal(state.history[-1],np.ones(6))
    np.testing.assert_array_equal(request.history,original)
    assert not request.history.flags.writeable


@pytest.mark.parametrize('step',[4,2,1])
def test_requested_cadence_uses_actual_played_frames(step):
    state=started(step)
    plan=publish(state,state.capture(),forecast(1))
    assert not state.due()
    for frame in range(1,step):
        state.acknowledge(1,frame,plan)
        assert not state.due()
    state.acknowledge(1,step,plan)
    assert state.due()
    request=state.capture()
    assert request.played_frame==step and request.origin_frame==step+3
    assert request.played_history[-1,0]==step
    assert request.history[-1,0]==request.origin_frame
    assert request.prefix_plan_ids==(plan,)*3


def test_latest_intent_is_independent_of_inflight_forecast():
    state=started()
    old=state.capture()
    state.receive_intent(intent(2,(1.,0.)),2)
    state.receive_intent(intent(3,(-1.,0.)),3)
    pending=state.publish(old,forecast(1))
    assert state.capture() is None
    state.accept(1,pending,1)
    assert state.capture() is None
    for frame in range(1, 5):
        state.acknowledge(1, frame, pending)
    fresh=state.capture()
    assert fresh.input_seq==3 and fresh.velocity==(-1.,0.)
    assert old.input_seq==1 and old.origin_frame==0 and fresh.played_frame==4
    assert fresh.origin_frame==7


def test_forecast_replacement_keeps_old_played_frames_and_absolute_time():
    state=started(1)
    old=publish(state,state.capture(),forecast(1))
    state.acknowledge(1,1,old)
    state.receive_intent(intent(2))
    fresh=state.capture()
    assert fresh.origin_frame==4
    before=state.history.copy()
    new=publish(state,fresh,forecast(105))
    np.testing.assert_array_equal(state.history,before)
    for frame in (2,3,4):state.acknowledge(1,frame,old)
    with pytest.raises(ValueError,match='schedule'):state.acknowledge(1,5,old)
    state.acknowledge(1,5,new)
    assert state.history[-1,0]==105
    np.testing.assert_array_equal(state.history[-3:-1,0],[3.,4.])


def test_queued_prefix_uses_the_accepted_plan_at_each_future_frame():
    state=started(1)
    first=publish(state,state.capture(),forecast(1))
    state.acknowledge(1,1,first)
    state.receive_intent(intent(2))
    second=publish(state,state.capture(),forecast(105))
    state.acknowledge(1,2,first)
    state.receive_intent(intent(3))
    third=state.capture()
    assert third.origin_frame==5 and third.prefix_plan_ids==(first,first,second)
    np.testing.assert_array_equal(third.history[:,0],[2.,3.,4.,105.])
    np.testing.assert_array_equal(third.played_history[:,0],[0.,0.,1.,2.])


def test_shorter_measured_lead_keeps_previously_committed_prefix():
    state=started(1)
    first=publish(state,state.capture(),forecast(1))
    state.acknowledge(1,1,first)
    state.receive_intent(intent(2))
    second=publish(state,state.capture(),forecast(105))
    state.configure(1)
    state.observe_latency(10.)
    state.acknowledge(1,2,first)
    state.receive_intent(intent(3))
    request=state.capture()
    assert request.origin_frame==5 and request.prefix_plan_ids==(first,first,second)
    replacement=publish(state,request,forecast(206))
    for frame in (3,4):state.acknowledge(1,frame,first)
    state.acknowledge(1,5,second)
    with pytest.raises(ValueError,match='schedule'):state.acknowledge(1,6,second)
    state.acknowledge(1,6,replacement)


def test_ack_rejects_duplicate_skip_unknown_and_wrong_episode():
    state=started()
    plan=publish(state,state.capture(),forecast(1))
    for args in ((1,2,plan),(2,1,plan),(1,1,999)):
        with pytest.raises(ValueError): state.acknowledge(*args)
    state.acknowledge(1,1,plan)
    with pytest.raises(ValueError): state.acknowledge(1,1,plan)
    for frame in range(2,25):state.acknowledge(1,frame,plan)
    with pytest.raises(ValueError):state.acknowledge(1,25,plan)


def test_late_result_is_discarded_without_skipping_its_initial_frames():
    state=started()
    request=state.capture()
    plan=publish(state,request,forecast(1))
    for frame in range(1,25):state.acknowledge(1,frame,plan)
    before=state.history.copy()
    assert state.publish(request,forecast(1)) is None
    np.testing.assert_array_equal(state.history,before)
    assert state.due()


def test_compute_lead_is_independent_from_requested_update_interval():
    state=started(1)
    state.observe_latency(65.)
    assert state.lead_frames==2 and state.step_frames==1
    state.configure(4)
    assert state.lead_frames==2 and state.step_frames==4
    state.observe_latency(120.)
    assert state.lead_frames==4
    for _ in range(12):state.observe_latency(10.)
    assert state.lead_frames==0
    state.observe_latency(1000.)
    assert state.lead_frames==6


def test_delivery_miss_adds_measured_scheduling_margin_without_advancing_history():
    state=started(1)
    state.observe_latency(10.)
    first=publish(state,state.capture(),forecast(1))
    state.acknowledge(1,1,first)
    request=state.capture()
    pending=state.publish(request,forecast(102))
    state.acknowledge(1,2,first)
    before=state.history.copy()
    state.reject(1,pending)
    state.observe_latency(5.)
    assert state.lead_frames==1 and state.step_frames==1
    np.testing.assert_array_equal(before,state.history)


def test_unconfirmed_result_cannot_feed_the_next_generation_or_playback():
    state=started()
    plan=state.publish(state.capture(),forecast(1))
    assert state.capture() is None
    with pytest.raises(ValueError):state.acknowledge(1,1,plan)
    state.reject(1,plan)
    assert state.due() and not state.accepted
    state.receive_intent(intent(2))
    assert state.due()


def test_continuous_analog_input_cannot_shorten_selected_execution_interval():
    state = started(4)
    old = publish(state, state.capture(), forecast(1))
    seq = 1
    for frame in range(1, 5):
        state.acknowledge(1, frame, old)
        for _ in range(10):
            seq += 1
            state.receive_intent(intent(seq, (.5 + seq * .0001, 1.)))
        assert state.due() == (frame == 4)
    fresh = state.capture()
    assert fresh.input_seq == seq
    new = publish(state, fresh, forecast(fresh.origin_frame + 1))
    state.observe_latency(0.)
    for frame in range(5, 9):
        source = old if frame <= fresh.origin_frame else new
        state.acknowledge(1, frame, source)
    next_request = state.capture()
    assert next_request.origin_frame >= fresh.origin_frame + 4
