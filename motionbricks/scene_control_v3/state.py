"""Fixed-size factual contact lifecycle, separate from actor proposals."""

from dataclasses import dataclass

import torch

from .contracts import ContactEvent


@dataclass(frozen=True)
class ContactLifecycleState:
    actual_contact: torch.Tensor
    observed_valid: torch.Tensor
    object_id: torch.Tensor
    material_id: torch.Tensor
    anchor_ego_m: torch.Tensor
    normal_ego: torch.Tensor
    surface_velocity_ego_mps: torch.Tensor
    observation_time_s: torch.Tensor
    last_observed_time_s: torch.Tensor
    interval_start_s: torch.Tensor
    last_event_time_s: torch.Tensor
    event: torch.Tensor
    active_age_s: torch.Tensor
    inactive_age_s: torch.Tensor

    @classmethod
    def init(cls, batch_size, *, contact_slots=15, device=None, dtype=torch.float32):
        shape = (batch_size, contact_slots)
        floating = torch.zeros(shape, device=device, dtype=dtype)
        vector = torch.zeros((*shape, 3), device=device, dtype=dtype)
        timestamp = torch.full(shape, -torch.inf, device=device, dtype=torch.float64)
        integer = torch.full(shape, -1, device=device, dtype=torch.int64)
        boolean = torch.zeros(shape, device=device, dtype=torch.bool)
        return cls(
            boolean, boolean.clone(), integer, integer.clone(), vector, vector.clone(), vector.clone(),
            timestamp, timestamp.clone(), timestamp.clone(), timestamp.clone(),
            torch.zeros_like(integer), floating, floating.clone(),
        )


def observe_contacts(state, *, observation_time_s, active, valid, object_id, material_id,
                     anchor_ego_m, normal_ego, surface_velocity_ego_mps):
    """Admit CPU observations; unknown breaks observed continuity.

    material_id identifies a persistent material point/contact-event anchor, not
    a material type. Newer continuous observations emit NONE; last_event_time_s
    retains the last transition time. Repeated or older observations leave each
    slot unchanged, including its event and observation_time_s. Consume events
    once per (batch, slot, observation_time_s). Unknown samples preserve the last
    factual contact but mark it unobserved and freeze its age.
    """
    observation = (observation_time_s, active, valid, object_id, material_id,
                   anchor_ego_m, normal_ego, surface_velocity_ego_mps)
    if any(value.device.type != "cpu" for value in observation) or state.actual_contact.device.type != "cpu":
        raise ValueError("Contact observation admission requires CPU tensors")
    if not torch.isfinite(observation_time_s).all():
        raise ValueError("Contact observation timestamps must be finite")
    known_active = valid & active
    for identity in (object_id, material_id):
        if identity.dtype != torch.int64 or (identity[known_active] < 0).any():
            raise ValueError("Known active contacts require nonnegative int64 object/material-point binding IDs")
    for geometry in (anchor_ego_m, normal_ego, surface_velocity_ego_mps):
        if not torch.isfinite(geometry[known_active]).all():
            raise ValueError("Known active contact anchors, normals and surface velocities must be finite")

    time = observation_time_s.to(dtype=torch.float64).unsqueeze(-1).expand_as(state.observation_time_s)
    newer = time > state.observation_time_s
    known = newer & valid
    same_binding = (object_id == state.object_id) & (material_id == state.material_id)
    continuous = state.observed_valid & valid & (active == state.actual_contact) & (~active | same_binding)
    restart = known & ~continuous
    start = torch.where(restart, time, state.interval_start_s)
    observed_age = (time - start).to(state.active_age_s.dtype)
    active_age = torch.where(known, torch.where(active, observed_age, 0.0), state.active_age_s)
    inactive_age = torch.where(known, torch.where(~active, observed_age, 0.0), state.inactive_age_s)

    event = torch.zeros_like(state.event)
    event = torch.where(restart & active, ContactEvent.ACQUIRED, event)
    rebound = restart & active & state.observed_valid & state.actual_contact & ~same_binding
    event = torch.where(rebound, ContactEvent.REBOUND, event)
    event = torch.where(restart & ~active, ContactEvent.OBSERVED_FREE, event)
    released = restart & ~active & state.observed_valid & state.actual_contact
    event = torch.where(released, ContactEvent.RELEASED, event)
    lost = newer & ~valid & state.observed_valid
    event = torch.where(lost, ContactEvent.OBSERVATION_LOST, event)
    changed = event != ContactEvent.NONE

    def observed_vector(value, previous):
        clean = torch.where((valid & active).unsqueeze(-1), value, 0.0)
        return torch.where(known.unsqueeze(-1), clean, previous)

    return ContactLifecycleState(
        actual_contact=torch.where(known, active, state.actual_contact),
        observed_valid=torch.where(newer, valid, state.observed_valid),
        object_id=torch.where(known, torch.where(active, object_id, -1), state.object_id),
        material_id=torch.where(known, torch.where(active, material_id, -1), state.material_id),
        anchor_ego_m=observed_vector(anchor_ego_m, state.anchor_ego_m),
        normal_ego=observed_vector(normal_ego, state.normal_ego),
        surface_velocity_ego_mps=observed_vector(surface_velocity_ego_mps, state.surface_velocity_ego_mps),
        observation_time_s=torch.where(newer, time, state.observation_time_s),
        last_observed_time_s=torch.where(known, time, state.last_observed_time_s),
        interval_start_s=start,
        last_event_time_s=torch.where(changed, time, state.last_event_time_s),
        event=torch.where(newer, event, state.event),
        active_age_s=active_age,
        inactive_age_s=inactive_age,
    )


def contact_features(state, proposed_contact_probability, predicted_release_probability):
    """Pack 16 channels without letting predictions change factual state."""
    dtype = state.anchor_ego_m.dtype
    known = state.observed_valid
    elapsed = torch.where(known, state.last_observed_time_s - state.last_event_time_s, 0.0).to(dtype)
    scalar = torch.stack((
        state.actual_contact.to(dtype), proposed_contact_probability, predicted_release_probability,
        state.active_age_s, state.inactive_age_s, elapsed,
    ), dim=-1)
    features = torch.cat((scalar, state.anchor_ego_m, state.normal_ego,
                          state.surface_velocity_ego_mps, known.to(dtype).unsqueeze(-1)), dim=-1)
    return torch.where(known.unsqueeze(-1), features, 0.0), known.clone()
