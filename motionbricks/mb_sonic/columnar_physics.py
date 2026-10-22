"""Lossless columnar storage for an ordered batch of actual physics states."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .runtime import PhysicsState


SCHEMA = "motionbricks.physics_state_batch.v1"
CONTACT_BATCH_KIND = "motionbricks.contact_batch.v1"
DENSE_FIELDS = (
    "qpos", "qvel", "joint_positions", "joint_velocities",
    "body_positions_world", "body_quat_wxyz", "actuator_controls",
    "actuator_forces", "generalized_actuator_forces", "root_qfrc_actuator",
    "qfrc_applied", "xfrc_applied",
)
WARNING_NAMES = (
    "mjWARN_INERTIA", "mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL",
    "mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL",
)
CONTACT_ARRAY_SHAPES = {
    "position_world": (3,), "frame_contact_from_world": (3, 3),
    "wrench_contact": (6,),
}
CONTACT_COLUMN_SHAPES = dict(
    geom_ids=(2,), body_ids=(2,), **CONTACT_ARRAY_SHAPES, distance=(),
)


def _stack(values, *, empty_shape=()):
    if not values:
        return np.empty((0, *empty_shape), dtype=np.float64)
    first = values[0]
    if any(value.dtype != first.dtype or value.shape != first.shape for value in values):
        raise ValueError("Column entries must have identical dtype and shape; no promotion is allowed")
    return np.stack(values, dtype=first.dtype)


def _legacy_contact_columns(contacts):
    columns = {
        "geom_ids": np.asarray([contact.geom_ids for contact in contacts], dtype=np.int64).reshape(-1, 2),
        "body_ids": np.asarray([contact.body_ids for contact in contacts], dtype=np.int64).reshape(-1, 2),
        "distance": np.asarray([contact.distance for contact in contacts], dtype=np.float64),
    }
    for name, shape in CONTACT_ARRAY_SHAPES.items():
        column = _stack([getattr(contact, name) for contact in contacts], empty_shape=shape)
        if column.shape != (len(contacts), *shape):
            raise ValueError(f"Invalid contact column shape: {name}")
        columns[name] = column
    return columns


def _contact_columns(groups):
    if all(isinstance(group, tuple) for group in groups):
        return _legacy_contact_columns(tuple(contact for group in groups for contact in group))
    parts = []
    for group in groups:
        if isinstance(group, tuple):
            parts.append(_legacy_contact_columns(group))
        elif getattr(group, "kind", None) == CONTACT_BATCH_KIND:
            parts.append({name: getattr(group, name) for name in CONTACT_COLUMN_SHAPES})
        else:
            raise TypeError("Contacts must be a ContactForce tuple or motionbricks.contact_batch.v1")
    columns = {}
    for name, shape in CONTACT_COLUMN_SHAPES.items():
        values = [part[name] for part in parts]
        dtype = values[0].dtype
        if any(value.shape != (len(group), *shape) for group, value in zip(groups, values)):
            raise ValueError(f"Invalid contact column shape: {name}")
        if any(value.dtype != dtype for value in values):
            raise ValueError("Contact columns must have identical dtype; no promotion is allowed")
        columns[name] = np.concatenate(values, dtype=dtype)
    return columns


def pack_physics_states(states: tuple["PhysicsState", ...]) -> dict:
    """Own all columns, preserving every sample and floating-point bit pattern."""
    warning_names = tuple(states[0].solver_warnings) if states else WARNING_NAMES
    if set(warning_names) != set(WARNING_NAMES):
        raise ValueError("Physics state must contain all seven MuJoCo solver warnings")
    warning_values = np.empty((len(states), 7, 2), dtype=np.int64)
    offsets = np.zeros(len(states) + 1, dtype=np.int64)
    for index, state in enumerate(states):
        if set(state.solver_warnings) != set(warning_names):
            raise ValueError("Solver warning columns differ within the batch")
        warning_values[index] = [
            (state.solver_warnings[name]["number"], state.solver_warnings[name]["lastinfo"])
            for name in warning_names
        ]
        offsets[index + 1] = offsets[index] + len(state.contacts)
    contact_columns = dict(offsets=offsets, **_contact_columns([state.contacts for state in states]))
    return {
        "schema": SCHEMA,
        **{name: _stack([getattr(state, name) for state in states]) for name in DENSE_FIELDS},
        "time_s": np.asarray([state.time_s for state in states], dtype=np.float64),
        "captured_at_monotonic_ns": np.asarray(
            [state.captured_at_monotonic_ns for state in states], dtype=np.int64),
        "state_version": np.asarray([state.state_version for state in states], dtype=np.int64),
        "root_ground_known": np.asarray([state.root_ground_known for state in states], dtype=np.bool_),
        "root_ground_height_m": np.asarray([
            np.nan if state.root_ground_height_m is None else state.root_ground_height_m
            for state in states
        ], dtype=np.float64),
        "root_ground_height_none": np.asarray([
            state.root_ground_height_m is None for state in states
        ], dtype=np.bool_),
        "solver_warning_names": list(warning_names),
        "solver_warning_values": warning_values,
        "terminal_reason": [state.terminal_reason for state in states],
        "contacts": contact_columns,
    }


def unpack_physics_states(batch: dict) -> tuple["PhysicsState", ...]:
    """Reconstruct states from an NPZ reader's decoded columnar dictionary."""
    from .runtime import ContactForce, PhysicsState

    if batch["schema"] != SCHEMA:
        raise ValueError("Unsupported physics state batch schema")
    columns = batch["contacts"]
    warning_names = batch["solver_warning_names"]
    states = []
    for index in range(len(batch["time_s"])):
        start, stop = columns["offsets"][index:index + 2]
        contacts = tuple(ContactForce(
            geom_ids=tuple(int(value) for value in columns["geom_ids"][row]),
            body_ids=tuple(int(value) for value in columns["body_ids"][row]),
            **{name: columns[name][row].copy() for name in CONTACT_ARRAY_SHAPES},
            distance=float(columns["distance"][row]),
        ) for row in range(int(start), int(stop)))
        states.append(PhysicsState(
            **{name: batch[name][index].copy() for name in DENSE_FIELDS},
            time_s=float(batch["time_s"][index]),
            captured_at_monotonic_ns=int(batch["captured_at_monotonic_ns"][index]),
            state_version=int(batch["state_version"][index]),
            root_ground_known=bool(batch["root_ground_known"][index]),
            root_ground_height_m=(None if batch["root_ground_height_none"][index]
                                  else float(batch["root_ground_height_m"][index])),
            solver_warnings={name: dict(number=int(values[0]), lastinfo=int(values[1]))
                             for name, values in zip(warning_names, batch["solver_warning_values"][index])},
            contacts=contacts,
            terminal_reason=batch["terminal_reason"][index],
        ))
    return tuple(states)
