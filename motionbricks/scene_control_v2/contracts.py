"""Shared serialized contact IDs, CPU batches and resolved run configuration."""

from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path

import numpy as np
import torch


RUN_SCHEMA = "motionbricks.scene_control_v2.run.v1"
BATCH_SCHEMA = "motionbricks.scene_control_v2.paired_torch_batch.v1"
PAIRED_SCHEMA = "motionbricks.scene_control_v2.paired.v1"


class TorchSceneQuery:
    """Device-preserving point VJP over synchronous CPU BVH distance queries.

    Mesh sign uses Warp winding classification. Unique nearest-point derivatives
    are signed (p-nearest)/distance, including edges/corners. At zero distance
    the chosen subgradient is zero; ties select the same first object as SceneQuery.
    Meshes/transforms are fixed labels or accepted actor observations, not variables.
    """

    def __init__(self, scene_query, *, scene_metadata, mode, as_of_s):
        from .geometry import MeshField

        if mode not in ("actor", "offline") or not np.isfinite(as_of_s):
            raise ValueError("Explicit actor/offline mode and finite query time required")
        if any(not isinstance(obj.field, MeshField) for obj in scene_query.objects):
            raise TypeError("TorchSceneQuery supports explicit triangle meshes only")
        ids = list(scene_metadata["object_ids"])
        if set(ids) != {obj.object_id for obj in scene_query.objects}:
            raise ValueError("Torch query and paired scene object identities differ")
        for obj in scene_query.objects:
            index = ids.index(obj.object_id)
            if (not np.array_equal(obj.object_to_world, scene_metadata["object_to_world"][:, index])
                    or not np.array_equal(obj.valid, scene_metadata["object_valid"][:, index])):
                raise ValueError("Torch query must use the explicitly bound scene tracks")
        if mode == "actor":
            tracks = scene_metadata["object_track_provenance"]
            if len(tracks) != len(ids):
                raise ValueError("Every actor object needs explicit track provenance")
            for obj in scene_query.objects:
                track = tracks[ids.index(obj.object_id)]
                if (not track["source"] or not np.isfinite(track["available_at_s"])
                        or track["available_at_s"] > as_of_s):
                    raise ValueError("Object track must be available at actor query time")
                if track["role"] == "static":
                    if not np.allclose(obj.object_to_world, obj.object_to_world[0], atol=1e-7, rtol=0):
                        raise ValueError("Static object track contains motion")
                elif track["role"] not in ("factual_history", "model_prediction", "known_schedule"):
                    raise ValueError("Actor cannot query future teacher scene tracks")
        self.scene_query = scene_query
        self.mode = mode
        self.scene_metadata = deepcopy(scene_metadata)
        self.as_of_s = as_of_s
        self.distance_semantics = scene_query.distance_semantics + ":nearest_point_first_derivative"

    def _query_flat(self, points, times, assigned_object_id):
        if assigned_object_id is None:
            return self.scene_query.query_world_torch(points, times)
        from .geometry import SceneQuery

        assigned = assigned_object_id.reshape(-1)
        result = {"distance_m": points.sum(-1) * 0,
                  "valid": torch.zeros(len(points), dtype=torch.bool, device=points.device),
                  "distance_gradient_valid": torch.zeros(len(points), dtype=torch.bool, device=points.device),
                  "object_id": torch.full((len(points),), -1, dtype=torch.int64, device=points.device)}
        for obj in self.scene_query.objects:
            selected = (assigned == obj.object_id).nonzero().flatten()
            if len(selected):
                query = SceneQuery((obj,), self.scene_query.valid_bounds_world_m,
                    validity_fn=self.scene_query.validity_fn).query_world_torch(points[selected], times[selected])
                for key in result:
                    result[key] = result[key].index_copy(0, selected, query[key].to(result[key]))
        return result

    def __call__(self, world_points_m, time_index, *, timestamps_s, assigned_object_id=None):
        if (world_points_m.dtype != torch.float32
                or time_index.device != world_points_m.device or time_index.dtype != torch.int64
                or time_index.shape != world_points_m.shape[:-1]):
            raise ValueError("Float32 points and aligned int64 time indices on the same device required")
        if assigned_object_id is not None and (assigned_object_id.dtype != torch.int64
                or assigned_object_id.device != world_points_m.device or assigned_object_id.shape != time_index.shape):
            raise ValueError("Assigned surfaces require aligned int64 persistent object IDs on the points device")
        time_cpu = time_index.detach().cpu().numpy()
        if np.any(time_cpu < 0) or np.any(time_cpu >= len(timestamps_s)):
            raise ValueError("Query time is outside the explicitly supplied timestamp track")
        queried_times = np.asarray(timestamps_s)[time_cpu]
        if self.mode == "actor":
            for track in self.scene_metadata["object_track_provenance"]:
                if track["role"] == "factual_history" and np.any(queried_times > track["available_at_s"] + 1e-9):
                    raise ValueError("Future scene transforms require predictions or a known schedule")
        query = self._query_flat(world_points_m.reshape(-1, 3), time_index.reshape(-1), assigned_object_id)
        shape = world_points_m.shape[:-1]
        return {"distance_m": query["distance_m"].reshape(shape), "valid": query["valid"].reshape(shape),
                "distance_gradient_valid": query["distance_gradient_valid"].reshape(shape),
                "object_id": query["object_id"].reshape(shape).long(),
                "valid_semantics": ("all_declared_components_known" if assigned_object_id is None
                                    else "assigned_object_component_known"),
                "query_scope": "scene" if assigned_object_id is None else "assigned_object",
                "distance_semantics": self.distance_semantics}


class ContactKind(IntEnum):
    SUPPORT = 0
    BRACE = 1
    GRASP = 2
    TOUCH = 3
    SLIDE = 4


@dataclass(frozen=True)
class NativeWindowFrame:
    """Native heading-retained frame: translate geometry, restore heading metadata.

    SOMA's native local velocities and global rotations retain world heading.
    Initial heading is restored to the integrated root heading channels only;
    rotating body geometry again would apply it twice. Vertical origin must be
    the observed terrain origin used to prepare this particular window.
    """

    translation_world_m: torch.Tensor  # [B,3]: first-frame XZ, observed terrain Y
    initial_heading_rad: torch.Tensor  # [B,1]

    @classmethod
    def from_global_history(cls, history, *, vertical_origin_m):
        translation = history[:, 0, :3].detach().clone()
        translation[:, 1] = torch.as_tensor(vertical_origin_m).to(history).reshape(-1)
        heading = torch.atan2(history[:, :1, 4], history[:, :1, 3]).detach()
        return cls(translation, heading)

    def points_to_window(self, points):
        offset = self.translation_world_m.reshape(len(points), *([1] * (points.ndim - 2)), 3)
        return points - offset

    def points_to_world(self, points):
        offset = self.translation_world_m.reshape(len(points), *([1] * (points.ndim - 2)), 3)
        return points + offset

    def center_global_motion(self, motion, global_rep):
        from motionbricks.data.native_mb_vertical_frame import shift_global_motion

        result = shift_global_motion(motion, global_rep, -self.translation_world_m[:, 1])
        result[..., [0, 2]] -= self.translation_world_m[:, None, [0, 2]]
        return result

    def restore_local_motion(self, local_motion, global_rep):
        from motionbricks.data.native_mb_vertical_frame import shift_global_motion

        if global_rep.removed_heading:
            raise ValueError("This frame requires the native heading-retained representation")
        result = global_rep.dual_rep.local_to_global(local_motion, True, False,
            lengths=torch.full((len(local_motion),), local_motion.shape[1], dtype=torch.long))
        result = shift_global_motion(result, global_rep, self.translation_world_m[:, 1])
        result[..., [0, 2]] += self.translation_world_m[:, None, [0, 2]]
        angle = torch.atan2(result[..., 4], result[..., 3]) + self.initial_heading_rad
        result[..., 3:5] = torch.stack((angle.cos(), angle.sin()), -1)
        return result


def paired_batch_in_window(batch, frame):
    """Translate world labels, queries and scene together into a native window.

    Does not create native 366-D global motion or Pose labels from body-region points.
    Native training world/history must separately use this same frame.
    """
    if batch["coordinate_frame"] != "paired_world_y_up_m":
        raise ValueError("Expected an uncentered paired-world batch")
    result = deepcopy(batch)
    result["body_positions_world_m"] = frame.points_to_window(batch["body_positions_world_m"])
    result["scene"]["object_to_world"][..., :3, 3] = frame.points_to_window(
        batch["scene"]["object_to_world"][..., :3, 3])
    result["scene"]["valid_bounds_world_m"] = frame.points_to_window(batch["scene"]["valid_bounds_world_m"])
    if "queries" in batch:
        result["queries"]["world_points_m"] = frame.points_to_window(batch["queries"]["world_points_m"])
        result["queries"]["closest_point_world_m"] = torch.where(
            batch["queries"]["closest_point_valid"][..., None],
            frame.points_to_window(batch["queries"]["closest_point_world_m"]), 0)
    result["coordinate_frame"] = "native_window_heading_retained_y_up_m"
    result["window_frame"] = {"translation_world_m": frame.translation_world_m.clone(),
                              "initial_heading_rad": frame.initial_heading_rad.clone(),
                              "geometry_rotation": "heading_retained_no_extra_yaw"}
    return result


def numpy_sample_to_torch_batch(sample, *, queries=None):
    """Copy one paired sample to CPU with a leading batch axis on numeric arrays.

    Labels remain labels: this adapter neither builds actor guidance nor invents
    native rotations/features from body positions. Strings/provenance stay metadata.
    Integer arrays become int64 for indexing, masks stay bool, floats stay float32.
    Persistent object_id values are unchanged; object_table_index is separate.
    """
    if sample["schema"] != PAIRED_SCHEMA:
        raise ValueError("Expected a scene_control_v2 paired sample")

    def convert(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "OUS":
                return value.tolist()
            tensor = torch.from_numpy(np.array(value, copy=True))
            if value.dtype.kind in "iu":
                tensor = tensor.to(torch.int64)
            elif value.dtype.kind == "f":
                tensor = tensor.to(torch.float32)
            return tensor.unsqueeze(0)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return deepcopy(value)

    batch = convert(sample)
    batch["sample_schema"] = batch["schema"]
    batch["schema"] = BATCH_SCHEMA
    batch["coordinate_frame"] = "paired_world_y_up_m"

    def object_indices(values):
        result = torch.full_like(values, -1)
        known = values == -1
        for index, object_id in enumerate(sample["scene"]["object_ids"]):
            selected = values == int(object_id)
            result[selected] = index
            known |= selected
        if not known.all():
            raise ValueError("Object IDs must refer to the paired scene table")
        return result

    contacts = batch["contacts"]
    contacts["object_table_index"] = object_indices(contacts["object_id"])
    batch["object_id_semantics"] = "persistent_scene_object_id"
    if queries is not None:
        batch["queries"] = convert(queries)
        batch["queries"]["object_table_index"] = object_indices(batch["queries"]["object_id"])
        batch["queries"]["target_object_table_index"] = object_indices(batch["queries"]["target_object_id"])
    return batch


def load_run_config(path, *, project_root=None):
    """Resolve the single JSON config without creating directories or run state."""
    path = Path(path).resolve(strict=True)
    root = Path(project_root).resolve(strict=True) if project_root else Path(__file__).resolve().parents[2]
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["schema"] != RUN_SCHEMA:
        raise ValueError("Unknown scene-control run schema")
    codec = config["codec"]
    if (codec["num_heads"] != 8 or codec["entries_per_head"] not in (10, 128, 256)
            or codec["code_dim"] != 256 or codec["index_layout"] != "B,L,H"):
        raise ValueError("Expected eight explicit 32-D codebooks with per-head indices")
    if codec["initialization"] not in ("legacy_exact", "new_codebook_untrained"):
        raise ValueError("Explicit legacy copy or new untrained codebook initialization required")
    if codec["initialization"] == "legacy_exact" and codec["entries_per_head"] != 10:
        raise ValueError("Legacy weights have ten entries per head")
    if config["contacts"]["kinds"] != {kind.name.lower(): int(kind) for kind in ContactKind}:
        raise ValueError("Contact kind IDs differ from the paired data schema")
    execution = config["execution"]
    if execution != {"mode": "preflight", "device": "cpu", "read_only": True,
                     "optimizer": None, "resume": False, "auto_expand": False}:
        raise ValueError("These integration profiles authorize read-only CPU preflight only")
    accounts = config["budget"]["accounts"]
    if any(value != 0 for account in accounts.values() for value in account.values()):
        raise ValueError("Diagnostic and formal allocations require a separate reviewed run config")
    if config["budget"]["legacy_completed_exposures"] != 38_400_000:
        raise ValueError("The completed legacy budget is immutable")
    if config["sampling"]["actor_guidance_sources"] != ["factual_history", "model_prediction"]:
        raise ValueError("Actor guidance cannot use paired future labels")
    if config["normalization"] != {"source": "architecture_bundle", "mean_key": "stats_mean",
            "std_key": "stats_std", "identity_key": "stats_tensor_sha256", "frozen": True}:
        raise ValueError("Native preflight uses the exact frozen bundle normalization")
    if config["quality_gates"]["quality_approved"] or config["quality_gates"]["fixture_can_approve"]:
        raise ValueError("Preflight cannot grant model quality approval")
    gate = config["quality_gates"]["fullskin"]
    if gate != {"vertices": 18056, "substeps": 8, "peak_penetration_m": 0.02,
                "required": True, "preflight_can_approve": False}:
        raise ValueError("Original fullskin acceptance is required")
    artifacts = config["artifacts"]
    for key in ("architecture_bundle", "checkpoint"):
        artifacts[key] = str((root / artifacts[key]).resolve(strict=True))
    config["resolved"] = {"project_root": str(root), "config_path": str(path)}
    return config
