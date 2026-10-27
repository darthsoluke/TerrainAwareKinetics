"""Runtime foot locking (two-bone IK), the shared motion metrics and the v6 contact-rate anchor."""

import numpy as np
import pytest

GROUND = {"center_xyz_m": [0, -0.1, 0], "half_extent_xyz_m": [20, 0.1, 20],
          "rotation_local_to_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


def _rep():
    pytest.importorskip("torch")
    pytest.importorskip("kimodo")
    from motionbricks.data.bones_seed import build_representation
    return build_representation()


def _random_rotations(torch, shape, scale, generator):
    from motionbricks.controller.foot_lock import axis_angle_matrix
    axis = torch.randn(*shape, 3, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.rand(*shape, generator=generator) * scale
    return axis_angle_matrix(axis, angle)


def frames_from(rep, rotations, roots):
    """Native world frames [B, F, 366] from local rotations [B, F, J, 3, 3] and roots [B, F, 3]."""
    import torch
    batch, frames = roots.shape[:2]
    canonical, origin = rep(dict(local_joint_rots=rotations, translation=roots, foot_contacts=torch.zeros(batch, frames, 4)),
                            to_normalize=False, return_init_heading_info=True,
                            lengths=torch.full((batch,), frames, dtype=torch.long))
    quat = origin["init_heading_quat"].reshape(batch, 4)
    yaw = 2 * torch.atan2(quat[:, 2], quat[:, 0])
    return rep.change_first_heading(canonical, yaw, False, False)


def test_two_bone_ik_reaches_reachable_targets_and_keeps_bone_lengths():
    torch = pytest.importorskip("torch")
    from motionbricks.controller.foot_lock import two_bone_ik
    g = torch.Generator().manual_seed(0)
    n = 512
    a = torch.randn(n, 3, generator=g, dtype=torch.float64)
    ga = _random_rotations(torch, (n,), 3.0, g).double()
    lb = _random_rotations(torch, (n,), 2.0, g).double()
    gb = ga @ lb
    ob, oc = torch.tensor([0, -0.45, 0.0], dtype=torch.float64), torch.tensor([0, -0.43, 0.02], dtype=torch.float64)
    b = a + ga @ ob
    c = b + gb @ oc
    target = a + torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.3
    new_a, new_b, reached = two_bone_ik(a, b, c, target, ga, gb, torch.tensor([0, 0, 1.0], dtype=torch.float64).expand(n, 3))
    lab, lcb = (b - a).norm(dim=-1), (c - b).norm(dim=-1)
    distance = (target - a).norm(dim=-1)
    reachable = (distance < (lab + lcb) * 0.998) & (distance > (lab - lcb).abs() + 1e-3)
    fk_c = (a + new_a @ ob) + new_b @ oc                                   # FK with the new rotations
    assert torch.allclose(fk_c, reached, atol=1e-9)
    assert (reached - target).norm(dim=-1)[reachable].max() < 1e-6
    assert torch.allclose(new_a @ new_a.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(n, 3, 3), atol=1e-9)
    # A straight leg has no bend plane; the hint supplies one.
    straight_c = b + ga @ (ob * 0.95)
    target = a + ga @ (ob * 1.5)
    hint = ga @ torch.tensor([0, 0, 1.0], dtype=torch.float64)
    new_a, new_b, reached = two_bone_ik(a, b, straight_c, target, ga, ga, hint)
    assert torch.isfinite(reached).all() and (reached - target).norm(dim=-1).max() < 1e-6


def test_foot_lock_fk_matches_the_representation():
    rep = _rep()
    import torch
    from motionbricks.controller.closed_loop import posed_joints
    from motionbricks.controller.foot_lock import FootLock
    g = torch.Generator().manual_seed(1)
    rotations = _random_rotations(torch, (2, 6, rep.nbjoints), 0.4, g)
    roots = torch.zeros(2, 6, 3)
    roots[..., 1] = 0.9
    roots[..., 2] = torch.arange(6) * 0.04
    frames = frames_from(rep, rotations, roots)
    decoded = rep.inverse(frames[:, -1:], False, return_all=False, joint_positions_from="global_rot_data")
    lock = FootLock(rep)
    joints, _ = lock.fk(decoded["local_joint_rots"][:, 0], frames[:, -1, :3])
    assert (joints - posed_joints(rep, frames[:, -1:])[:, 0]).norm(dim=-1).max() < 1e-4


def test_foot_lock_pins_a_sliding_stance_foot_and_releases():
    rep = _rep()
    import torch
    from motionbricks.controller.foot_lock import FootLock, FootLockConfig
    from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d
    g = torch.Generator().manual_seed(2)
    lock = FootLock(rep, FootLockConfig(release_frames=2))
    from motionbricks.controller.foot_lock import axis_angle_matrix
    rotations = _random_rotations(torch, (1, rep.nbjoints), 0.15, g)
    hip, knee, ankle, _ = lock.legs[0]
    x = torch.tensor([[1.0, 0.0, 0.0]])
    rotations[:, hip] = axis_angle_matrix(x, torch.tensor([-0.4]))       # rest legs point along -y: bend about x
    rotations[:, knee] = axis_angle_matrix(x, torch.tensor([0.8]))
    state = lock.initial_state(1, torch.device("cpu"))
    ankles, rot_out = [], []
    for t in range(10):
        contact = torch.tensor([[4.0, 4.0, -4.0, -4.0]]) if t < 7 else torch.tensor([[-4.0, -4.0, -4.0, -4.0]])
        pred = {"local_rot6d": matrix_to_cont6d(rotations), "root_next": torch.tensor([[0.0, 0.85, 0.01 * t]]),
                "contact_logits": contact}
        corrected, state, info = lock.apply(pred, state)
        from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix
        new_rot = cont6d_to_matrix(corrected["local_rot6d"])
        joints, _ = lock.fk(new_rot, pred["root_next"])
        ankles.append(joints[0, ankle])
        rot_out.append(new_rot)
    pinned = torch.stack(ankles[:7])
    assert (pinned - pinned[0]).norm(dim=-1).max() < 1e-3                 # the stance ankle stays put
    raw, _ = lock.fk(rotations, torch.tensor([[0.0, 0.85, 0.09]]))
    assert (ankles[9] - raw[0, ankle]).norm() < 1e-4                       # released and blended back
    untouched = [j for j in range(rep.nbjoints) if j not in (hip, knee, ankle)]
    assert torch.allclose(rot_out[3][0, untouched], rotations[0, untouched], atol=1e-5)
    assert not bool(state["locked"][0, 1])                                  # the unloaded side never locked


def test_rollout_frames_modes_and_progress_ratio():
    rep = _rep()
    import torch
    from motionbricks.controller.closed_loop import ClosedLoop, RolloutConfig
    from motionbricks.controller.foot_lock import FootLock, rollout_frames
    from motionbricks.controller.terrain_motion_transformer import TerrainMotionTransformer
    from motionbricks.controller.tokens import PartRayTable, StationProber
    from motionbricks.controller.training import motion_metrics
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler
    names = list(rep.skeleton.bone_order_names)
    sampler = SkeletonRaySampler(names, box_capacity=8)
    table, prober = PartRayTable(sampler.layout), StationProber()
    torch.manual_seed(0)
    model = TerrainMotionTransformer(rep, names, table.feature_dim, prober.feature_dim).eval()
    loop = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=4))
    g = torch.Generator().manual_seed(3)
    rotations = _random_rotations(torch, (2, 1, rep.nbjoints), 0.3, g).expand(2, 32, rep.nbjoints, 3, 3).clone()
    roots = torch.zeros(2, 32, 3)
    roots[..., 1] = 0.9
    roots[..., 2] = torch.arange(32) / 30
    frames = frames_from(rep, rotations, roots)
    packed = sampler.boxes([GROUND])
    batch = {"history": frames[:, :16], "history_mask": torch.ones(2, 16, dtype=torch.bool), "targets": frames[:, 16:],
             "commands": torch.tensor([0.0, 1.0, 1.0, 0.0]).expand(2, 16, 4),
             "box_centers": packed[0].expand(2, -1, -1), "box_half": packed[1].expand(2, -1, -1),
             "box_rotation": packed[2].expand(2, -1, -1, -1), "box_mask": packed[3].expand(2, -1),
             "contacts": torch.ones(2, 16, 4), "group": ["flat_control", "crouch"]}
    raw, info = rollout_frames(loop, batch, 4)
    with torch.no_grad():
        reference = loop.rollout(batch["history"], batch["history_mask"], batch["commands"][:, :4],
                                 (batch["box_centers"], batch["box_half"], batch["box_rotation"], batch["box_mask"]))
    assert info == {} and torch.allclose(raw, reference["history"][:, -4:], atol=1e-5)
    lock = FootLock(rep)
    for mode in ("display", "feedback"):
        shown, info = rollout_frames(loop, batch, 4, lock=lock, mode=mode)
        assert shown.shape == (2, 4, 366) and torch.isfinite(shown).all() and info["active"].shape == (2, 4, 2)
    perfect = motion_metrics(loop, batch, batch["targets"][:, :4], 4)
    assert perfect["closed_loop_root_drift_cm_p50"] < 1e-3 and perfect["moving_windows"] == 2
    assert perfect["closed_loop_progress_ratio_p50"] == pytest.approx(1.0, abs=1e-5)
    held = motion_metrics(loop, batch, batch["history"][:, -1:].expand(2, 4, 366), 4)
    assert held["closed_loop_progress_ratio_p50"] < 1e-3                   # standing still makes no progress
    assert set(held["by_group"]["crouch"]) >= {"progress_ratio_p50", "grounded_sole_speed_mps_p50"}


def test_contact_rate_anchor_is_phase_free_and_reaches_the_contact_head():
    rep = _rep()
    import torch
    from motionbricks.controller.closed_loop import ClosedLoop, RolloutConfig
    from motionbricks.controller.terrain_motion_transformer import TerrainMotionTransformer
    from motionbricks.controller.tokens import PartRayTable, StationProber
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler
    names = list(rep.skeleton.bone_order_names)
    sampler = SkeletonRaySampler(names, box_capacity=8)
    table, prober = PartRayTable(sampler.layout), StationProber()
    torch.manual_seed(0)
    model = TerrainMotionTransformer(rep, names, table.feature_dim, prober.feature_dim)
    g = torch.Generator().manual_seed(4)
    rotations = _random_rotations(torch, (2, 1, rep.nbjoints), 0.3, g).expand(2, 24, rep.nbjoints, 3, 3).clone()
    roots = torch.zeros(2, 24, 3)
    roots[..., 1] = 0.9
    roots[..., 2] = torch.arange(24) / 30
    frames = frames_from(rep, rotations, roots)
    boxes = sampler.boxes([GROUND])
    boxes = (boxes[0].expand(2, -1, -1), boxes[1].expand(2, -1, -1), boxes[2].expand(2, -1, -1, -1), boxes[3].expand(2, -1))
    loop = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=4, contact_rate=5.0))
    result = loop.rollout(frames[:, :16], torch.ones(2, 16, dtype=torch.bool), torch.tensor([0.0, 1.0, 1.0, 0.0]).expand(2, 4, 4),
                          boxes, targets=frames[:, 16:20])
    assert "contact_rate" in result["terms"] and torch.isfinite(result["policy_loss"])
    result["policy_loss"].backward()
    assert model.contact_head.weight.grad is not None and model.contact_head.weight.grad.abs().sum() > 0
    off = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=4))
    assert "contact_rate" not in off.rollout(frames[:, :16], torch.ones(2, 16, dtype=torch.bool),
                                             torch.tensor([0.0, 1.0, 1.0, 0.0]).expand(2, 4, 4), boxes,
                                             targets=frames[:, 16:20])["terms"]
