"""Owned contact columns read from MuJoCo, without per-contact Python copies."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactBatch:
    kind = "motionbricks.contact_batch.v1"
    geom_ids: np.ndarray
    body_ids: np.ndarray
    position_world: np.ndarray
    frame_contact_from_world: np.ndarray
    wrench_contact: np.ndarray
    distance: np.ndarray

    def __len__(self):
        return len(self.distance)

    def __getitem__(self, index):
        from .runtime import ContactForce

        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return ContactForce(tuple(map(int, self.geom_ids[index])), tuple(map(int, self.body_ids[index])),
            self.position_world[index], self.frame_contact_from_world[index],
            self.wrench_contact[index], float(self.distance[index]))

    def __iter__(self):
        return (self[index] for index in range(len(self)))


def capture_contacts(model, data):
    """Match mj_contactForce, including the native sequential pyramid summation.

    Reference: MuJoCo3.8 engine_util_misc.c:mju_decodePyramid. This reads solver
    results; it does not change contact generation, constraint forces or state.
    """
    if mujoco.__version__ != "3.8.0":
        raise RuntimeError("Contact-column equivalence is scoped to MuJoCo3.8.0")
    source = data.contact
    count = data.ncon
    geom = np.array(source.geom, copy=True).reshape(count, 2)
    force = np.zeros((count, 6), dtype=np.float64)
    addresses = np.asarray(source.efc_address)
    dimensions = np.asarray(source.dim)
    active = addresses >= 0
    if not ((dimensions == 1) | (dimensions == 3) | (dimensions == 4) | (dimensions == 6)).all():
        raise ValueError("Unsupported MuJoCo contact dimension")
    if model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
        friction = np.asarray(source.friction)
        for dim in (1, 3, 4, 6):
            ids = np.flatnonzero(active & (dimensions == dim))
            if not len(ids):
                continue
            start = addresses[ids]
            if dim == 1:
                force[ids, 0] = data.efc_force[start]
                continue
            # Keep the exact C accumulation order; pairwise/reduction sums change bits.
            normal = np.zeros(len(ids), dtype=np.float64)
            for index in range(2*(dim-1)):
                normal += data.efc_force[start+index]
            force[ids, 0] = normal
            for index in range(dim-1):
                force[ids, index+1] = ((data.efc_force[start+2*index]-data.efc_force[start+2*index+1])
                                      * friction[ids, index])
    elif model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        for dim in (1, 3, 4, 6):
            ids = np.flatnonzero(active & (dimensions == dim))
            if len(ids):
                force[ids, :dim] = data.efc_force[addresses[ids, None]+np.arange(dim)]
    else:
        raise ValueError("Unsupported MuJoCo contact cone")
    return ContactBatch(geom, np.array(model.geom_bodyid[geom], copy=True),
        np.array(source.pos, copy=True).reshape(count, 3),
        np.array(source.frame, copy=True).reshape(count, 3, 3), force,
        np.array(source.dist, copy=True).reshape(count))
