from motionbricks.data.stop_transition_adapter import CommandTransitionState


def test_continuous_move_is_not_a_transition():
    state = CommandTransitionState()
    assert not state.update([3.,0.],2.)
    assert not state.update([3.,0.],.05)
    assert not state.update([3.,0.],2.)


def test_restart_retains_learned_branch_until_motion_is_established():
    state = CommandTransitionState()
    assert not state.update([3.,0.],2.)
    assert state.update([0.,0.],2.)
    assert state.update([0.,0.],.05)
    assert state.update([3.,0.],.05)
    assert state.update([3.,0.],.4)
    assert not state.update([3.,0.],.4)


def test_cold_start_and_release_are_causal():
    state = CommandTransitionState()
    assert state.update([3.,0.],.02)
    assert state.update([0.,0.],.4)
    assert state.resume_plans == 0
