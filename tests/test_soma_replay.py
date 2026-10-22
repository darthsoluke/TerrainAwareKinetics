import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.soma_replay import UE_BASIS, motion_to_ue, replay_fixture, skeleton_manifest
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


def test_basis_axes_and_reflected_quaternion():
    positions, _ = motion_to_ue(np.eye(3), np.eye(3))
    np.testing.assert_allclose(positions, [[0, -100, 0], [0, 0, 100], [100, 0, 0]])
    rng = np.random.default_rng(4)
    rotations = Rotation.random(100, random_state=rng).as_matrix()
    point = rng.normal(size=(100, 3))
    converted_p, converted_r = motion_to_ue(point, rotations)
    np.testing.assert_allclose(np.einsum("tij,tj->ti", converted_r, converted_p),
                               np.einsum("tij,tj->ti", rotations, point) @ UE_BASIS.T * 100, atol=1e-10)
    q = Rotation.from_matrix(rotations).as_quat()
    ue_q = np.stack([-q[:, 2], q[:, 0], -q[:, 1], q[:, 3]], -1)
    np.testing.assert_allclose(Rotation.from_quat(ue_q).as_matrix(), converted_r, atol=1e-12)
    yaw = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    _, ue_yaw = motion_to_ue([0, 0, 0], yaw)
    np.testing.assert_allclose(ue_yaw @ [1, 0, 0], [0, -1, 0], atol=1e-12)


def test_official_reference_and_isolated_joint_fk_commute_with_ue_basis():
    skeleton = SOMASkeleton30()
    for joint in ("Hips", "LeftArm", "RightArm", "LeftLeg", "RightLeg", "Spine1"):
        local = torch.eye(3).repeat(1, 30, 1, 1)
        local[0, skeleton.bone_index[joint]] = torch.tensor(Rotation.from_euler("x", 35, degrees=True).as_matrix(), dtype=torch.float32)
        global_r, positions, _ = skeleton.fk(local, torch.tensor([[0.2, 0.9, -0.4]]))
        ue_p, ue_r = motion_to_ue(positions.numpy(), global_r.numpy())
        _, ue_local = motion_to_ue([0, 0, 0], local.numpy())
        ue_neutral, _ = motion_to_ue(skeleton.neutral_joints.numpy(), np.eye(3))
        computed_p, computed_r = np.zeros_like(ue_p), np.zeros_like(ue_r)
        for i, parent in enumerate(skeleton.joint_parents.tolist()):
            if parent < 0:
                computed_p[:, i], _ = motion_to_ue([[0.2, 0.9, -0.4]], np.eye(3))
                computed_r[:, i] = ue_local[:, i]
            else:
                computed_r[:, i] = computed_r[:, parent] @ ue_local[:, i]
                computed_p[:, i] = computed_p[:, parent] + np.einsum("tij,j->ti", computed_r[:, parent], ue_neutral[i] - ue_neutral[parent])
        np.testing.assert_allclose(computed_p, ue_p, atol=2e-5)
        np.testing.assert_allclose(computed_r, ue_r, atol=1e-6)


def test_replay_contract_does_not_claim_trained_inference():
    manifest = skeleton_manifest()
    assert manifest == skeleton_manifest()
    assert manifest["runtime_state"] == "source_replay"
    assert not manifest["motionbricks_checkpoint_verified"]
    assert "checkpoints" not in manifest
    assert manifest["source_id"].startswith("kimodo:somaskel30:standard:")
    assert manifest["joints"][0]["name"] == "Hips"
    assert len(manifest["retarget_chains"]) == 11
    assert manifest["reference_ground_offset_m"] > 0.8
    arrays = {"local_rot_mats": np.broadcast_to(np.eye(3), (4, 30, 3, 3)).copy(),
              "root_positions": np.tile([0, 0.9, 0], (4, 1)),
              "global_features": np.zeros((4, 366)), "contact_confidence": np.ones((4, 4))}
    record = dict(filename="test", behavior="idle", source_sha256="0" * 64, split="train")
    fixture = replay_fixture(arrays, record, manifest, "1" * 64)
    assert fixture["rotation_space"] == "local" and fixture["quaternion_order"] == "xyzw"
    assert not fixture["model_inference"]
    np.testing.assert_allclose(fixture["frames"][0]["positions_m"], SOMASkeleton30().neutral_joints.numpy() + [0, 0.9, 0], atol=1e-7)
    arrays["local_rot_mats"][0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="proper rotations"):
        replay_fixture(arrays, record, manifest, "1" * 64)
