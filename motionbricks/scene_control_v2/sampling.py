"""Fixed quotas with explicit causal roles for body/contact/time guidance."""

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .geometry import SceneQuery


SAMPLING_POLICY = "scene_control_v2.global_body_time_contact.fixed_quota.v1"


@dataclass(frozen=True)
class QueryBudget:
    global_points: int = 128
    body_points: int = 256
    contact_points: int = 128
    body_radius_m: float = 0.15
    contact_radius_m: float = 0.04

    def __post_init__(self):
        counts = (self.global_points, self.body_points, self.contact_points)
        if (any(not isinstance(n, int) or n < 0 for n in counts) or sum(counts) <= 0
                or not np.isfinite([self.body_radius_m, self.contact_radius_m]).all()
                or min(self.body_radius_m, self.contact_radius_m) < 0):
            raise ValueError("Query quotas must be nonnegative integers with positive total")

    @property
    def total(self) -> int:
        return self.global_points + self.body_points + self.contact_points


@dataclass(frozen=True)
class QueryGuidance:
    role: Literal["factual_history", "model_prediction", "offline_teacher"]
    available_at_s: float
    body_positions_world_m: np.ndarray
    body_valid: np.ndarray
    contacts: Mapping


def offline_teacher_guidance(sample: Mapping) -> QueryGuidance:
    """Explicit offline oracle opt-in. Actor sampling rejects this role."""
    return QueryGuidance("offline_teacher", float(sample["timestamps_s"][-1]),
                         sample["body_positions_world_m"], sample["body_valid"], sample["contacts"])


def _cycle_candidates(candidates, count: int, rng) -> np.ndarray:
    # Round-robin body/contact channels; cycle each channel's available times.
    groups = rng.permutation(np.unique(candidates[:, 1]))
    tracks = [candidates[candidates[:, 1] == group][rng.permutation((candidates[:, 1] == group).sum())]
              for group in groups]
    return np.asarray([tracks[i % len(tracks)][(i // len(tracks)) % len(tracks[i % len(tracks)])]
                       for i in range(count)])


def sample_scene_queries(scene_query: SceneQuery, sample: Mapping, *,
                         guidance: QueryGuidance, mode: Literal["actor", "offline"],
                         as_of_s: float, budget: QueryBudget = QueryBudget(),
                         seed: int = 2333) -> dict:
    """Global/body/contact quotas never borrow missing guidance from label arrays.

    Missing guidance reserves masked slots in its own quota; it does not silently
    change the sampling policy. Sampling is deterministic for a fixed seed.
    Paired future labels are read only by `offline_teacher_guidance`, never here.
    """
    if mode not in ("actor", "offline") or guidance.role not in (
            "factual_history", "model_prediction", "offline_teacher"):
        raise ValueError("Explicit mode and guidance role are required")
    if not np.isfinite([as_of_s, guidance.available_at_s]).all():
        raise ValueError("Availability timestamps must be finite")
    if mode == "actor" and (guidance.role == "offline_teacher" or guidance.available_at_s > as_of_s):
        raise ValueError("Actor guidance must be factual history or predictions available as of actor time")
    times = np.asarray(sample["timestamps_s"])
    frames, body = len(times), len(sample["body_part_names"])
    positions = np.asarray(guidance.body_positions_world_m, np.float32)
    body_valid = np.asarray(guidance.body_valid, bool)
    contacts = guidance.contacts
    if positions.shape != (frames, body, 3) or body_valid.shape != (frames, body):
        raise ValueError("Guidance must match the explicit sample time/body table")
    if not np.isfinite(positions[body_valid]).all():
        raise ValueError("Valid body guidance must be finite")
    future = times > guidance.available_at_s + 1e-9
    contact_valid = np.asarray(contacts["valid"], bool)
    anchor_valid = np.asarray(contacts["anchor_valid"], bool)
    if contact_valid.ndim != 2 or contact_valid.shape[0] != frames or anchor_valid.shape != contact_valid.shape:
        raise ValueError("Guidance contact masks must have aligned [T,C] shape")
    if guidance.role == "factual_history" and (
            body_valid[future].any() or contact_valid[future].any() or anchor_valid[future].any()):
        raise ValueError("Factual guidance cannot expose future ground-truth body/contact labels")

    # Bind the queried geometry tracks to the serialized paired scene tracks.
    scene = sample["scene"]
    lookup = {int(value): i for i, value in enumerate(scene["object_ids"])}
    if set(lookup) != {obj.object_id for obj in scene_query.objects}:
        raise ValueError("SceneQuery objects differ from paired sample scene binding")
    for obj in scene_query.objects:
        i = lookup[obj.object_id]
        if (not np.array_equal(obj.object_to_world, scene["object_to_world"][:, i])
                or not np.array_equal(obj.valid, scene["object_valid"][:, i])):
            raise ValueError("SceneQuery transforms/validity differ from paired sample")
    if mode == "actor":
        tracks = scene["object_track_provenance"]
        if len(tracks) != len(lookup):
            raise ValueError("Every actor object track needs explicit provenance")
        for obj in scene_query.objects:
            track = tracks[lookup[obj.object_id]]
            if not track["source"] or track["available_at_s"] > as_of_s:
                raise ValueError("Object track provenance must be available at actor time")
            role = track["role"]
            if role == "static":
                if not np.allclose(obj.object_to_world, obj.object_to_world[0], atol=1e-7, rtol=0):
                    raise ValueError("Static object track contains motion")
            elif role not in ("factual_history", "model_prediction", "known_schedule"):
                raise ValueError("Actor scene transforms cannot use offline teacher tracks")

    rng = np.random.default_rng(seed)
    n = budget.total
    points = np.zeros((n, 3), np.float32)
    time_index = np.zeros(n, np.int32)
    part = np.full(n, -1, np.int32)
    contact_index = np.full(n, -1, np.int32)
    target_object = np.full(n, -1, np.int32)
    region = np.repeat(np.arange(3, dtype=np.int32),
                       [budget.global_points, budget.body_points, budget.contact_points])
    sample_valid = np.zeros(n, bool)
    time_pool = np.flatnonzero(~future) if guidance.role == "factual_history" else np.arange(frames)
    if not len(time_pool):
        raise ValueError("No query times are available for this guidance")
    coarse = slice(0, budget.global_points)
    points[coarse] = rng.uniform(scene_query.valid_bounds_world_m[0],
                                 scene_query.valid_bounds_world_m[1], size=(budget.global_points, 3))
    time_index[coarse] = np.resize(rng.permutation(time_pool), budget.global_points)
    sample_valid[coarse] = True
    candidates = np.argwhere(body_valid)
    start, end = budget.global_points, budget.global_points + budget.body_points
    if len(candidates) and budget.body_points:
        selected = _cycle_candidates(candidates, budget.body_points, rng)
        t, b = selected.T
        points[start:end] = positions[t, b] + rng.uniform(
            -budget.body_radius_m, budget.body_radius_m, size=(budget.body_points, 3))
        time_index[start:end], part[start:end] = t, b
        sample_valid[start:end] = True
    candidates = np.argwhere(contact_valid & anchor_valid & (np.asarray(contacts["state"]) > 0.5))
    if len(candidates) and budget.contact_points:
        selected = _cycle_candidates(candidates, budget.contact_points, rng)
        t, c = selected.T
        anchors = scene_query.anchors_world(np.asarray(contacts["object_id"])[t, c], t,
                                            np.asarray(contacts["anchor_local_m"])[t, c],
                                            np.asarray(contacts["normal_local"])[t, c])
        points[end:] = anchors["world_points_m"] + rng.uniform(
            -budget.contact_radius_m, budget.contact_radius_m, size=(budget.contact_points, 3))
        time_index[end:], contact_index[end:] = t, c
        part[end:] = np.asarray(contacts["body_part_id"])[c]
        target_object[end:] = np.asarray(contacts["object_id"])[t, c]
        sample_valid[end:] = anchors["valid"]
    if mode == "actor":
        for track in tracks:
            if track["role"] == "factual_history" and np.any(
                    times[time_index[sample_valid]] > track["available_at_s"] + 1e-9):
                raise ValueError("Future queries require predicted or known-schedule scene transforms")
    result = scene_query.query_world(points, time_index)
    result["valid"] &= sample_valid
    result["distance_m"][~result["valid"]] = 0
    result["normal_world"][~result["valid"]] = 0
    result["distance_gradient_world"][~result["valid"]] = 0
    result["distance_gradient_valid"] &= result["valid"]
    result["closest_point_world_m"][~result["valid"]] = 0
    result["closest_point_valid"] &= result["valid"]
    result["object_id"][~result["valid"]] = -1
    result.update(policy=SAMPLING_POLICY, body_part_id=part, contact_index=contact_index,
                  target_object_id=target_object, region_id=region, sample_valid=sample_valid,
                  time_s=times[time_index].copy(), guidance_role=guidance.role,
                  guidance_available_at_s=guidance.available_at_s, mode=mode, as_of_s=as_of_s)
    return result
