from dataclasses import fields

import pytest
import torch

from motionbricks.scene_control_v3 import (
    ContactEvent, ContactLifecycleState, contact_features, observe_contacts,
)


@pytest.fixture(autouse=True)
def cpu_two_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def observation(time_s, *, batch_size=1, slots=15, active=True, valid=True, object_id=7, material_id=3):
    shape = (batch_size, slots)
    return {
        "observation_time_s": torch.full((batch_size,), time_s, dtype=torch.float64),
        "active": torch.full(shape, active, dtype=torch.bool),
        "valid": torch.full(shape, valid, dtype=torch.bool),
        "object_id": torch.full(shape, object_id, dtype=torch.int64),
        "material_id": torch.full(shape, material_id, dtype=torch.int64),
        "anchor_ego_m": torch.full((*shape, 3), 0.25),
        "normal_ego": torch.tensor([0.0, 1.0, 0.0]).expand(*shape, 3).clone(),
        "surface_velocity_ego_mps": torch.full((*shape, 3), 0.1),
    }


def assert_state_equal(left, right):
    for item in fields(left):
        torch.testing.assert_close(getattr(left, item.name), getattr(right, item.name),
                                   equal_nan=True, rtol=0, atol=0)


def test_lifecycle_uses_observation_seconds_not_probability_or_call_count():
    state = ContactLifecycleState.init(1)
    acquired = observe_contacts(state, **observation(1000000.0))
    sustained = observe_contacts(acquired, **observation(1000000.1))
    assert (acquired.event == ContactEvent.ACQUIRED).all()
    assert (sustained.event == ContactEvent.NONE).all()
    torch.testing.assert_close(sustained.last_event_time_s, acquired.last_event_time_s, rtol=0, atol=0)
    assert (acquired.active_age_s == 0).all()
    torch.testing.assert_close(sustained.active_age_s, torch.full((1, 15), 0.1))
    packed, valid = contact_features(sustained, torch.zeros(1, 15), torch.ones(1, 15))
    assert packed.shape == (1, 15, 16) and valid.all()
    assert (packed[..., 0] == 1).all() and (packed[..., 1] == 0).all() and (packed[..., 2] == 1).all()
    assert sustained.actual_contact.all() and sustained.observed_valid.all()
    torch.testing.assert_close(packed[..., 3], sustained.active_age_s)
    assert_state_equal(sustained, observe_contacts(sustained, **observation(1000000.1)))
    assert_state_equal(sustained, observe_contacts(sustained, **observation(999999.0, active=False)))


@pytest.mark.parametrize("changed", ["object_id", "material_id"])
def test_object_and_material_rebinding_reset_interval_even_with_constant_contact(changed):
    state = observe_contacts(ContactLifecycleState.init(1), **observation(2.0))
    state = observe_contacts(state, **observation(4.0))
    next_observation = observation(5.0)
    next_observation[changed] += 1
    rebound = observe_contacts(state, **next_observation)
    assert rebound.actual_contact.all()
    assert (rebound.event == ContactEvent.REBOUND).all()
    assert (rebound.active_age_s == 0).all()
    torch.testing.assert_close(rebound.last_event_time_s, torch.full((1, 15), 5.0, dtype=torch.float64))


def test_release_and_reacquisition_are_factual_transitions():
    state = observe_contacts(ContactLifecycleState.init(1), **observation(0.0))
    released = observe_contacts(state, **observation(0.2, active=False))
    free = observe_contacts(released, **observation(0.5, active=False))
    assert not released.actual_contact.any()
    assert (released.event == ContactEvent.RELEASED).all()
    assert (free.event == ContactEvent.NONE).all()
    assert (released.object_id == -1).all() and (released.material_id == -1).all()
    torch.testing.assert_close(free.inactive_age_s, torch.full((1, 15), 0.3))
    assert (free.anchor_ego_m == 0).all() and (free.active_age_s == 0).all()
    acquired = observe_contacts(free, **observation(0.6))
    assert (acquired.event == ContactEvent.ACQUIRED).all() and (acquired.active_age_s == 0).all()


def test_unknown_does_not_invent_release_or_extend_age_and_nan_inputs_are_masked():
    state = observe_contacts(ContactLifecycleState.init(1), **observation(0.0))
    state = observe_contacts(state, **observation(0.1))
    unknown = observation(1.0, valid=False)
    for key in ("anchor_ego_m", "normal_ego", "surface_velocity_ego_mps"):
        unknown[key].fill_(torch.nan)
    lost = observe_contacts(state, **unknown)
    assert lost.actual_contact.all() and not lost.observed_valid.any()
    assert (lost.event == ContactEvent.OBSERVATION_LOST).all()
    torch.testing.assert_close(lost.active_age_s, state.active_age_s)
    for key in ("anchor_ego_m", "normal_ego", "surface_velocity_ego_mps"):
        assert torch.isfinite(getattr(lost, key)).all()
    packed, valid = contact_features(lost, torch.full((1, 15), torch.nan), torch.full((1, 15), torch.nan))
    assert not valid.any() and (packed == 0).all()
    recovered = observe_contacts(lost, **observation(2.0))
    assert (recovered.event == ContactEvent.ACQUIRED).all() and (recovered.active_age_s == 0).all()


def test_unknown_initial_state_is_explicit_and_all_prediction_values_remain_separate():
    state = ContactLifecycleState.init(2)
    proposed = torch.ones(2, 15)
    release = torch.zeros(2, 15)
    packed, valid = contact_features(state, proposed, release)
    assert not state.actual_contact.any() and not valid.any() and (packed == 0).all()
    assert (state.object_id == -1).all()
    proposed.zero_()
    assert not state.actual_contact.any()


@pytest.mark.parametrize("key", ["anchor_ego_m", "normal_ego", "surface_velocity_ego_mps", "observation_time_s"])
def test_cpu_admission_rejects_nonfinite_known_evidence(key):
    obs = observation(0.0)
    obs[key].fill_(torch.nan)
    with pytest.raises(ValueError, match="finite"):
        observe_contacts(ContactLifecycleState.init(1), **obs)


@pytest.mark.parametrize("key", ["object_id", "material_id"])
def test_cpu_admission_rejects_unknown_or_noninteger_active_binding(key):
    obs = observation(0.0)
    obs[key].fill_(-1)
    with pytest.raises(ValueError, match="binding IDs"):
        observe_contacts(ContactLifecycleState.init(1), **obs)
    obs[key] = obs[key].float()
    with pytest.raises(ValueError, match="binding IDs"):
        observe_contacts(ContactLifecycleState.init(1), **obs)


def test_newer_unknown_observation_clears_event_without_inventing_release():
    acquired = observe_contacts(ContactLifecycleState.init(1), **observation(1.0))
    lost = observe_contacts(acquired, **observation(2.0, valid=False))
    still_unknown = observe_contacts(lost, **observation(3.0, valid=False))
    assert (lost.event == ContactEvent.OBSERVATION_LOST).all()
    assert (still_unknown.event == ContactEvent.NONE).all() and still_unknown.actual_contact.all()
    torch.testing.assert_close(still_unknown.last_event_time_s, lost.last_event_time_s, rtol=0, atol=0)


def test_independent_slots_batch_times_no_mutation_or_growing_state():
    state = ContactLifecycleState.init(2)
    before = ContactLifecycleState(**{item.name: getattr(state, item.name).clone() for item in fields(state)})
    count = sum(getattr(state, item.name).numel() for item in fields(state))
    obs = observation(0.0, batch_size=2)
    obs["observation_time_s"][1] = 5.0
    obs["valid"][0, 1:] = False
    obs_before = {key: value.clone() for key, value in obs.items()}
    updated = observe_contacts(state, **obs)
    assert_state_equal(state, before)
    for key, value in obs.items():
        torch.testing.assert_close(value, obs_before[key], rtol=0, atol=0)
    assert updated.observed_valid[0].sum() == 1 and updated.observed_valid[1].all()
    for step in range(1, 101):
        obs["observation_time_s"] += 1.0 / 30.0
        updated = observe_contacts(updated, **obs)
        assert sum(getattr(updated, item.name).numel() for item in fields(updated)) == count
    torch.testing.assert_close(updated.active_age_s[1], torch.full((15,), 100.0 / 30.0))
