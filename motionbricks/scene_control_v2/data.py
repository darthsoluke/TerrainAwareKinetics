"""Paired data contract, family split audit and bounded public metadata readers."""

from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path

import numpy as np
import trimesh

from .geometry import MeshField, SceneObject, SceneQuery, rigid_transform


PAIRED_SCHEMA = "motionbricks.scene_control_v2.paired.v1"
BODY_PART_NAMES = (
    "pelvis", "head", "left_palm", "right_palm", "left_forearm", "right_forearm",
    "left_knee", "right_knee", "left_sole", "right_sole", "left_shoulder",
    "right_shoulder", "back",
)


def empty_contacts(frames: int, channels: int = 0) -> dict:
    """Unknown contact slots, not supervised negatives."""
    shape = (frames, channels)
    return {
        "state": np.zeros(shape, np.float32), "actual_state": np.zeros(shape, np.float32),
        "valid": np.zeros(shape, bool), "actual_valid": np.zeros(shape, bool),
        "anchor_valid": np.zeros(shape, bool), "confidence": np.zeros(shape, np.float32),
        "anchor_local_m": np.zeros((*shape, 3), np.float32),
        "normal_local": np.zeros((*shape, 3), np.float32),
        "object_id": np.full(shape, -1, np.int32), "kind": np.full(shape, -1, np.int32),
        "body_part_id": np.full(channels, -1, np.int32),
        "provenance": np.full(channels, "unavailable", dtype="U128"),
    }


def validate_paired_sample(sample: Mapping) -> None:
    """Check the versioned training boundary; no data repair or inferred fields."""
    if sample["schema"] != PAIRED_SCHEMA or sample["split"] not in ("train", "val", "test"):
        raise ValueError("Expected v2 paired schema and train/val/test split")
    for key in ("source", "family", "recording_id", "scene_id", "skeleton"):
        if not sample[key]:
            raise ValueError(f"Missing sample binding: {key}")
    time = np.asarray(sample["timestamps_s"])
    frames, body = np.asarray(sample["body_valid"]).shape
    if (time.shape != (frames,) or not np.isfinite(time).all() or frames == 0
            or not np.isfinite(sample["fps"]) or sample["fps"] <= 0
            or not np.allclose(np.diff(time), 1 / sample["fps"], atol=1e-6)):
        raise ValueError("Timestamps must match explicit positive source FPS")
    positions = np.asarray(sample["body_positions_world_m"])
    if (positions.shape != (frames, body, 3) or len(sample["body_part_names"]) != body
            or np.asarray(sample["body_valid"]).dtype != np.bool_
            or not np.isfinite(positions[sample["body_valid"]]).all()):
        raise ValueError("Body coordinates and validity must share the body/time table")
    scene = sample["scene"]
    ids = np.asarray(scene["object_ids"])
    if len(ids) != len(set(ids.tolist())) or np.any(ids < 0):
        raise ValueError("Scene object IDs must be unique and nonnegative")
    transforms = np.asarray(scene["object_to_world"])
    object_valid = np.asarray(scene["object_valid"])
    if (transforms.shape != (frames, len(ids), 4, 4)
            or object_valid.shape != (frames, len(ids)) or object_valid.dtype != np.bool_):
        raise ValueError("Scene transform tracks must match motion frames and object IDs")
    for matrix in transforms[object_valid]:
        rigid_transform(matrix)
    contacts = sample["contacts"]
    channels = len(contacts["body_part_id"])
    for key in ("state", "actual_state", "valid", "actual_valid", "anchor_valid",
                "confidence", "object_id", "kind"):
        if np.asarray(contacts[key]).shape != (frames, channels):
            raise ValueError(f"contacts.{key} requires [T,C]")
    for key in ("valid", "actual_valid", "anchor_valid"):
        if np.asarray(contacts[key]).dtype != np.bool_:
            raise ValueError(f"contacts.{key} must be boolean")
    for key in ("anchor_local_m", "normal_local"):
        array = np.asarray(contacts[key])
        if array.shape != (frames, channels, 3) or not np.isfinite(array[contacts["anchor_valid"]]).all():
            raise ValueError(f"contacts.{key} requires finite valid [T,C,3] slots")
    if (len(contacts["provenance"]) != channels
            or np.any((contacts["body_part_id"] < 0) | (contacts["body_part_id"] >= body))):
        raise ValueError("Contact channels need explicit body region and provenance")
    for state_key, valid_key in (("state", "valid"), ("actual_state", "actual_valid")):
        values = np.asarray(contacts[state_key])[contacts[valid_key]]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Valid contact states must be finite probabilities")
    anchor_mask = contacts["anchor_valid"]
    if (not np.isin(contacts["object_id"][anchor_mask], ids).all()
            or not np.isin(contacts["kind"][contacts["valid"]], np.arange(5)).all()):
        raise ValueError("Valid contact anchors/kinds require known objects and canonical IDs")
    anchor_norms = np.linalg.norm(contacts["normal_local"][anchor_mask], axis=-1)
    if not np.allclose(anchor_norms, 1, atol=1e-5):
        raise ValueError("Valid contact anchor normals must be unit length")
    for key in ("source_uri", "source_version", "license", "parent_recording_ids",
                "scene_family", "actor_id", "original_frame_interval"):
        if key not in sample["provenance"]:
            raise ValueError(f"Missing provenance field: {key}")


def assert_family_disjoint(samples: Sequence[Mapping]) -> None:
    """All variants, parent recordings and scene families stay in one split."""
    seen = {}
    for sample in samples:
        source, split = sample["source"], sample["split"]
        p = sample["provenance"]
        keys = [(source, "family", sample["family"]),
                (source, "recording", sample["recording_id"]),
                (source, "scene_family", p["scene_family"])]
        keys += [(source, "recording", parent) for parent in p["parent_recording_ids"]]
        # Explicit globally stable lineage keys also join derivatives across datasets.
        keys += [("global", "lineage", key) for key in p.get("lineage_ids", ())]
        for key in keys:
            if key in seen and seen[key] != split:
                raise ValueError(f"Split leakage for {key}: {seen[key]} vs {split}")
            seen[key] = split


def make_synthetic_fixture(*, frames: int = 8, fps: float = 30.0) -> dict:
    """Low ceiling, overhang and moving object; synthetic offline oracle only."""
    if frames < 2 or fps <= 0:
        raise ValueError("Fixture requires at least two frames and positive FPS")
    body = len(BODY_PART_NAMES)
    points = np.zeros((frames, body, 3), np.float32)
    points[:, :, 2] = np.linspace(-0.6, 0.6, frames)[:, None]
    points[:, :, 1] = np.array([0.65, 0.95, 0.5, 0.5, 0.6, 0.6, 0.2, 0.2,
                               0.05, 0.05, 0.8, 0.8, 0.75])
    points[:, :, 0] = np.array([0, 0, -0.25, 0.25, -0.3, 0.3, -0.2, 0.2,
                               -0.2, 0.2, -0.25, 0.25, 0])
    transforms = np.tile(np.eye(4, dtype=np.float32), (frames, 4, 1, 1))
    transforms[:, 0, :3, 3] = [0, -0.1, 0]
    transforms[:, 1, :3, 3] = [0, 1.2, 0]
    transforms[:, 2, :3, 3] = [1.2, 1.0, 0]
    transforms[:, 3, :3, 3] = np.column_stack((np.linspace(0.8, 1.1, frames),
                                              np.full(frames, 0.4), np.full(frames, 1.2)))
    contacts = empty_contacts(frames, 2)
    contacts.update(
        state=np.ones((frames, 2), np.float32), actual_state=np.ones((frames, 2), np.float32),
        valid=np.ones((frames, 2), bool), actual_valid=np.ones((frames, 2), bool),
        anchor_valid=np.ones((frames, 2), bool), confidence=np.ones((frames, 2), np.float32),
        object_id=np.tile(np.array([0, 3], np.int32), (frames, 1)),
        kind=np.tile(np.array([0, 1], np.int32), (frames, 1)),
        body_part_id=np.array([8, 2], np.int32),
        provenance=np.array(["synthetic_declared_support", "synthetic_declared_brace"]),
    )
    contacts["anchor_local_m"][:, 0] = [-0.2, 0.1, 0]
    contacts["anchor_local_m"][:, 1] = [0, 0.2, 0]
    contacts["normal_local"][:, :, 1] = 1
    # Put the declared synthetic body regions at their declared synthetic anchors.
    points[:, 8] = contacts["anchor_local_m"][:, 0] + transforms[:, 0, :3, 3]
    points[:, 2] = contacts["anchor_local_m"][:, 1] + transforms[:, 3, :3, 3]
    sample = {
        "schema": PAIRED_SCHEMA, "source": "synthetic", "family": "low_ceiling_overhang_v1",
        "split": "train", "fps": fps, "skeleton": "synthetic_regions13_not_SOMA",
        "recording_id": "synthetic_oracle_000", "scene_id": "synthetic_scene_000",
        "timestamps_s": np.arange(frames, dtype=np.float64) / fps,
        "body_positions_world_m": points, "body_valid": np.ones((frames, body), bool),
        "body_part_names": BODY_PART_NAMES, "contacts": contacts,
        "scene": {
            "geometry_kind": "triangle_mesh", "object_ids": np.arange(4, dtype=np.int32),
            "object_names": ("floor", "low_ceiling", "overhang", "moving_brace"),
            "object_to_world": transforms, "object_valid": np.ones((frames, 4), bool),
            "valid_bounds_world_m": np.array([[-3, -0.5, -3], [3, 2, 3]], np.float32),
            "transform_role": "offline_teacher", "available_at_s": 0.0,
            "object_track_provenance": [
                {"role": "static", "available_at_s": 0.0, "source": "synthetic_declared_static"}
                for _ in range(3)
            ] + [{"role": "offline_teacher", "available_at_s": (frames - 1) / fps,
                  "source": "synthetic_oracle_motion_track"}],
            "provenance": "synthetic_mesh_boxes_declared_not_captured",
        },
        "provenance": {
            "source_uri": "synthetic://scene_control_v2/low_ceiling_overhang_v1",
            "source_version": "1", "license": "synthetic_test_fixture",
            "parent_recording_ids": [], "scene_family": "synthetic_low_ceiling_overhang",
            "actor_id": "synthetic", "original_frame_interval": [0, frames],
            "synthetic": True, "purpose": "offline_oracle_diagnostic",
        },
    }
    return sample


def make_synthetic_scene(sample: Mapping) -> SceneQuery:
    if sample["source"] != "synthetic":
        raise ValueError("Synthetic geometry constructor only accepts synthetic fixtures")
    scene = sample["scene"]
    extents = ([6, 0.2, 6], [1.8, 0.2, 2], [1.0, 0.2, 2], [0.6, 0.4, 0.6])
    objects = tuple(SceneObject(int(object_id), scene["object_names"][i],
                                MeshField(trimesh.creation.box(extents=extents[i])),
                                scene["object_to_world"][:, i], scene["object_valid"][:, i])
                    for i, object_id in enumerate(scene["object_ids"]))
    return SceneQuery(objects, scene["valid_bounds_world_m"])


PUBLIC_SPECS = {
    "LINGO": {
        "source_url": "https://github.com/mileret/lingo-release",
        "license_status": "dataset_terms_require_review",
        "arrays": {"human_pose.npy": (63,), "human_orient.npy": (3,),
                   "transl_aligned.npy": (3,), "human_joints_aligned.npy": (28, 3),
                   "start_idx.npy": (3,), "end_idx.npy": (3,),
                   "left_hand_inter_frame.npy": (), "right_hand_inter_frame.npy": ()},
        "bindings": ("scene_name.pkl", "Scene", "language_motion_dict"),
        "contact_scope": "hand interaction frame events only; no full-body anchors",
    },
    "TRUMANS": {
        "source_url": "https://github.com/hygenie1228/TRUMANS",
        "license_status": "download_form_and_dataset_terms_require_review",
        "arrays": {"human_pose.npy": (63,), "human_orient.npy": (3,),
                   "human_transl.npy": (3,), "seg_name.npy": (), "frame_id.npy": (),
                   "scene_flag.npy": (), "scene_list.npy": (), "object_list.npy": (),
                   "object_flag.npy": (35,), "object_mat.npy": (5, 4, 4), "bad_frames.npy": ()},
        "bindings": ("Scene", "Scene_mesh", "Object_all", "smplx_result"),
        "contact_scope": "source contact definition must be audited separately",
    },
    "CIRCLE": {
        "source_url": "https://github.com/Stanford-TML/circle_dataset",
        "license_status": "CC-BY-NC-4.0; SMPL-X_and_asset_terms_separate",
        "arrays": {}, "bindings": (),
        "contact_scope": "motion/scene pairing; full-body contact anchors unavailable",
    },
}


def inspect_public_metadata(dataset: str, root: str | Path) -> dict:
    """Read headers/file presence only; no pickle execution, download or retarget.

    `root` is supplied by the caller. No guessed dataset installation paths.
    CIRCLE inspection is bounded to eight files per published format.
    """
    name = dataset.upper()
    spec = PUBLIC_SPECS[name]
    root = Path(root)
    report = {"dataset": name, "root": str(root.resolve()), **spec,
              "exists": root.is_dir(), "fields": {}, "missing": [],
              "entity_samples_loaded": 0, "paired_training_ready": False}
    for filename, tail in spec["arrays"].items():
        path = root / filename
        if not path.is_file():
            report["missing"].append(filename)
            continue
        # Parse only the NPY header, including object-dtype lists; never unpickle.
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            readers = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                raise ValueError(f"Unsupported NPY metadata header version: {version}")
            shape, _, dtype = readers[version](handle)
        report["fields"][filename] = {
            "shape": list(shape), "dtype": str(dtype),
            "shape_valid": len(shape) == len(tail) + 1 and shape[1:] == tail,
            "payload_loaded": False,
        }
    for filename in spec["bindings"]:
        if not (root / filename).exists():
            report["missing"].append(filename)
        else:
            report["fields"][filename] = {"present": True, "contents_loaded": False}
    if name == "CIRCLE":
        for suffix in ("bvh", "npz", "glb", "urdf"):
            paths = list(islice(root.rglob(f"*.{suffix}"), 8)) if root.is_dir() else []
            report["fields"][suffix] = [str(p.relative_to(root)) for p in paths]
            if not paths:
                report["missing"].append(f"*.{suffix}")
        report["native_frame"] = "Y-up,-Z-forward"
    report["unresolved"] = ["license_evidence", "recording_and_scene_family_split",
                            "metric_coordinate_binding", "target_skeleton_conversion",
                            "contact_anchor_validity"]
    return report


def adapt_public_window(dataset: str, *, native_fields: Mapping, body_positions_world_m,
                        body_valid, body_part_names, timestamps_s, fps: float,
                        skeleton: str, recording_id: str, family: str, split: str,
                        scene_id: str, scene: Mapping, provenance: Mapping,
                        contacts: Mapping) -> dict:
    """Bind an explicitly converted small window; retain published native fields.

    The caller owns coordinate/skeleton conversion, scene binding and licensing.
    LINGO hand events remain native metadata, not invented support labels.
    """
    name = dataset.upper()
    spec = PUBLIC_SPECS[name]
    required = {"LINGO": ("human_pose", "human_orient", "transl_aligned"),
                "TRUMANS": ("human_pose", "human_orient", "human_transl", "object_mat"),
                "CIRCLE": ("motion",)}[name]
    missing = set(required) - native_fields.keys()
    if missing:
        raise ValueError(f"Missing public native fields: {sorted(missing)}")
    frames = len(timestamps_s)
    for filename, tail in spec["arrays"].items():
        key = Path(filename).stem
        if key in native_fields and key in required:
            array = np.asarray(native_fields[key])
            if array.shape != (frames, *tail):
                raise ValueError(f"Native field {key} must match window frames and published shape")
    sample = {
        "schema": PAIRED_SCHEMA, "source": name, "family": family, "split": split,
        "recording_id": recording_id, "scene_id": scene_id, "fps": fps,
        "skeleton": skeleton, "timestamps_s": np.asarray(timestamps_s),
        "body_positions_world_m": np.asarray(body_positions_world_m, np.float32),
        "body_valid": np.asarray(body_valid, bool), "body_part_names": tuple(body_part_names),
        "scene": dict(scene), "contacts": dict(contacts), "provenance": dict(provenance),
        "native_fields": dict(native_fields), "public_schema_source": spec["source_url"],
    }
    validate_paired_sample(sample)
    return sample


def inspect_parc15_sidecar(path: str | Path) -> dict:
    """Inspect existing sidecars read-only; labels remain source-native."""
    with np.load(path, allow_pickle=False) as arrays:
        contacts = arrays["source_contacts"]
        names = arrays["source_contact_names"].astype(str)
        frames = arrays["source_frame_indices"]
        fps = float(arrays["fps"])
        if contacts.shape != (len(frames), 15) or names.shape != (15,) or fps <= 0:
            raise ValueError("Expected PARC15 source contact sidecar")
        return {"path": str(Path(path).resolve()), "frames": len(frames), "channels": 15,
                "names": names.tolist(), "fps": fps, "finite": bool(np.isfinite(contacts).all()),
                "value_min": float(contacts.min()), "value_max": float(contacts.max()),
                "provenance": "PARC_source_labels_unverified_on_target_skeleton",
                "certified_contact": False, "anchor_valid": False}
