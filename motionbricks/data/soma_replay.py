"""Recorded SOMA fixtures and explicit UE basis conversion; no model inference."""

import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.bones_seed import (FPS, SOLE_PROXY_GEOMETRIES, SOLE_PROXY_GEOMETRY, SOMA_REVISION,
                                          build_representation, official_sole_offsets)
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


UE_BASIS = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def motion_to_ue(positions, rotations):
    return np.asarray(positions) @ UE_BASIS.T * 100, UE_BASIS @ rotations @ UE_BASIS.T


def skeleton_manifest():
    skeleton = SOMASkeleton30()
    offsets, mesh_digest = official_sole_offsets(skeleton)
    # Every revision is published so a dataset built with an older proxy geometry can
    # still be verified against the geometry it was actually converted with.
    offsets_by_revision = {revision: official_sole_offsets(skeleton, revision)[0].tolist()
                           for revision in SOLE_PROXY_GEOMETRIES}
    joints = [{"name": name, "parent_index": int(parent), "neutral_position": position.tolist()}
              for name, parent, position in zip(skeleton.bone_order_names,
                                                skeleton.joint_parents,
                                                skeleton.neutral_joints)]
    fingerprint = hashlib.sha256(json.dumps(
        {"joints": joints, "reference_pose": "standard", "kimodo_revision": SOMA_REVISION},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    chains = [{"name": name, "start": start, "end": end} for name, start, end in (
        ("Spine", "Spine1", "Chest"), ("Neck", "Neck1", "Neck2"), ("Head", "Head", "Head"),
        ("LeftClavicle", "LeftShoulder", "LeftShoulder"), ("LeftArm", "LeftArm", "LeftHand"),
        ("LeftLeg", "LeftLeg", "LeftFoot"), ("LeftToe", "LeftToeBase", "LeftToeBase"),
        ("RightClavicle", "RightShoulder", "RightShoulder"), ("RightArm", "RightArm", "RightHand"),
        ("RightLeg", "RightLeg", "RightFoot"), ("RightToe", "RightToeBase", "RightToeBase"))]
    return {
        "schema_version": 1, "name": "soma30_kimodo_standard",
        "source_id": "kimodo:somaskel30:standard:" + fingerprint,
        "skeleton_sha256": fingerprint,
        "provenance": {"repository": "https://github.com/nv-tlabs/kimodo", "commit": SOMA_REVISION},
        "motionbricks_checkpoint_verified": False, "runtime_state": "source_replay",
        "reference_pose": "Kimodo official standard; identity local rotations",
        "coordinates": "RH_Y_UP_Z_FORWARD_X_LEFT_METERS", "root_bone": "Hips",
        "joints": joints, "retarget_chains": chains,
        "reference_ground_offset_m": float(-(skeleton.neutral_joints[skeleton.foot_joint_idx] + offsets)[:, 1].median()),
        "ground_anchor_bones": skeleton.foot_joint_names,
        "foot_channels": skeleton.foot_joint_names, "sole_offsets_m": offsets.tolist(),
        "sole_proxy_geometry": SOLE_PROXY_GEOMETRY, "sole_offsets_by_revision": offsets_by_revision,
        "sole_mesh_sha256": mesh_digest, "sole_status": "rigid mesh proxies; not manual contact truth",
    }


def replay_fixture(arrays, record, manifest, shard_sha256):
    representation = build_representation()
    skeleton = representation.skeleton
    local = torch.as_tensor(arrays["local_rot_mats"], dtype=torch.float32)
    root = torch.as_tensor(arrays["root_positions"], dtype=torch.float32)
    if local.shape != (len(root), 30, 3, 3) or root.shape != (len(root), 3) or len(root) < 2:
        raise ValueError("Invalid SOMA replay shape")
    values = [local, root]
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite replay data")
    matrices = local.numpy().astype(np.float64)
    if not np.allclose(matrices @ matrices.swapaxes(-1, -2), np.eye(3), atol=2e-5) or not np.allclose(np.linalg.det(matrices), 1, atol=2e-5):
        raise ValueError("Local matrices must be proper rotations")
    global_rotations, positions, _ = skeleton.fk(local, root)
    positions = positions.numpy()
    global_rotations = global_rotations.numpy()
    local_q = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(-1, 30, 4)
    global_q = Rotation.from_matrix(global_rotations.reshape(-1, 3, 3)).as_quat().reshape(-1, 30, 4)
    features = np.asarray(arrays["global_features"])
    if features.shape != (len(root), len(representation.indices["all"])):
        raise ValueError("MotionBricks global feature shape differs from the official representation")
    contacts = features[:, representation.indices["foot_contacts"]]
    confidence = np.asarray(arrays["contact_confidence"])
    if contacts.shape != (len(root), 4) or confidence.shape != contacts.shape:
        raise ValueError("Four contact/confidence channels are required")
    if not np.isin(contacts, [0, 1]).all() or not np.isin(confidence, [0, 1]).all():
        raise ValueError("Replay contacts must be binary automatic labels")
    return {
        "schema": "motionbricks.soma.replay.v1", "runtime_state": "source_replay",
        "model_inference": False, "source_id": manifest["source_id"],
        "coordinates": manifest["coordinates"], "rotation_space": "local", "quaternion_order": "xyzw",
        "fps": FPS, "joint_names": skeleton.bone_order_names, "parents": skeleton.joint_parents.tolist(),
        "neutral_positions_m": skeleton.neutral_joints.tolist(),
        "foot_channels": skeleton.foot_joint_names, "sole_offsets_m": manifest["sole_offsets_m"],
        "contact_status": "automatic confidence-masked labels; not manual ground truth",
        "clip": {key: record[key] for key in ("filename", "behavior", "source_sha256", "split")},
        "shard_sha256": shard_sha256, "preprocess_version": "soma60-planar-v7",
        "frames": [{"root_m": root[i].tolist(), "local_xyzw": local_q[i].tolist(),
                    "global_xyzw": global_q[i].tolist(), "positions_m": positions[i].tolist(),
                    "contacts": contacts[i].tolist(), "contact_confidence": confidence[i].tolist()}
                   for i in range(len(root))],
    }
