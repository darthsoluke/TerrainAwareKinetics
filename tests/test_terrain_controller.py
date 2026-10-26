"""Controller pieces: scene SDF, tokens, transformer invariances and the differentiable loop."""

import numpy as np
import pytest

from motionbricks.controller.returns import lambda_returns_numpy
from motionbricks.controller.scene_sdf import box_sdf_numpy
from motionbricks.controller.tokens import BODY_PARTS, joint_parts, part_of_joint_name
from motionbricks.data.skeleton_ray_sampler import RayLayout, pack_boxes


def box(center, half, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return {"center_xyz_m": list(center), "half_extent_xyz_m": list(half),
            "rotation_local_to_world": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}


GROUND = box([0, -0.1, 0], [20, 0.1, 20])
NAMES = ["Hips", "Spine1", "Chest", "LeftClavicle", "RightEye", "HeadEnd", "Neck1", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
         "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftShin",
         "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]


def test_body_parts_follow_names():
    parts = joint_parts(NAMES)
    assert parts[NAMES.index("Hips")] == BODY_PARTS.index("root")
    assert parts[NAMES.index("Head")] == BODY_PARTS.index("torso")
    assert parts[NAMES.index("LeftForeArm")] == BODY_PARTS.index("left_arm")
    assert parts[NAMES.index("RightToeBase")] == BODY_PARTS.index("right_leg")
    with pytest.raises(ValueError):
        part_of_joint_name("Tail1")


def test_box_sdf_reference_is_exact_outside_and_negative_inside():
    boxes = pack_boxes([box([0, 0.5, 0], [0.5, 0.5, 0.5], yaw=0.4)], 4)
    points = np.array([[0, 0.5, 0], [0, 1.3, 0], [0, 0.5, 0.0]])
    points[2] = [0, 0.5, 0] + 0.2 * np.array([np.sin(0.4), 0, np.cos(0.4)])   # along the local z axis
    sdf = box_sdf_numpy(points, boxes)
    assert sdf[0] == pytest.approx(-0.5) and sdf[1] == pytest.approx(0.3) and sdf[2] == pytest.approx(-0.3)


def test_lambda_returns_reference():
    costs = np.array([[1.0, 2.0]])
    values = np.array([[0.0, 10.0, 20.0]])
    gamma, lam = 0.5, 0.5
    g1 = 2.0 + gamma * ((1 - lam) * 20 + lam * 20)
    g0 = 1.0 + gamma * ((1 - lam) * 10 + lam * g1)
    assert np.allclose(lambda_returns_numpy(costs, values, gamma, lam), [[g0, g1]])


# ---------------------------------------------------------------- torch
def test_torch_sdf_matches_reference_and_pushes_points_out():
    torch = pytest.importorskip("torch")
    from motionbricks.controller.scene_sdf import box_sdf, penetration_cost
    rng = np.random.default_rng(0)
    boxes_list = [GROUND] + [box(rng.uniform(-1, 1, 3), rng.uniform(.1, .4, 3), rng.uniform(0, 3)) for _ in range(5)]
    packed = pack_boxes(boxes_list, 8)
    tensors = tuple(torch.as_tensor(a, dtype=torch.float64 if a.dtype != bool else torch.bool) for a in packed)
    points = rng.uniform(-1.2, 1.2, (2, 50, 3))
    assert np.allclose(box_sdf(torch.as_tensor(points), tensors).numpy(), box_sdf_numpy(points, packed), atol=1e-9)
    inside = torch.tensor([[[0.0, -0.05, 0.0]]], requires_grad=True)
    penetration_cost(inside, (tensors[0][:1].float(), tensors[1][:1].float(), tensors[2][:1].float(),
                              tensors[3][:1])).sum().backward()
    assert inside.grad[0, 0, 1] < 0          # descending the cost moves the point up, out of the ground


def test_normalized_critic_bootstrap_backward_survives_the_scale_update():
    torch = pytest.importorskip("torch")
    from motionbricks.controller.closed_loop import Critic
    torch.manual_seed(0)
    critic = Critic(4, normalized=True)
    state = torch.randn(2, 4, requires_grad=True)
    policy = critic.frozen(state).clamp(0.0, 50.0).sum()       # bootstrap term of the policy loss
    critic_loss = critic.loss([state], torch.ones(2, 1))        # updates scale in place
    policy.backward()                                          # must not see the in-place update
    critic_loss.backward()
    assert state.grad is not None and critic.net[0].weight.grad is not None


def test_batched_boxes_match_shared_boxes():
    torch = pytest.importorskip("torch")
    from motionbricks.data.skeleton_ray_sampler import cast_rays
    packed = pack_boxes([GROUND, box([0, 1.5, 1], [1, .1, 1])], 4)
    shared = tuple(torch.as_tensor(a) if a.dtype == bool else torch.as_tensor(a, dtype=torch.float32) for a in packed)
    batched = tuple(t[None].expand(2, *t.shape).clone() for t in shared)
    origin = torch.tensor([[[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]]]).expand(2, 2, 3)
    direction = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]).expand(2, 2, 3)
    length = torch.tensor([2.0, 2.0])
    a, b = cast_rays(origin, direction, length, shared), cast_rays(origin, direction, length, batched)
    assert torch.allclose(a["distance"], b["distance"]) and torch.allclose(a["distance"][0], torch.tensor([0.4, 1.0]))


def test_station_rays_see_a_ceiling_ahead():
    torch = pytest.importorskip("torch")
    from motionbricks.controller.tokens import StationProber
    packed = pack_boxes([GROUND, box([0, 1.4, 2.0], [1, .1, .5])], 4)
    boxes = tuple(torch.as_tensor(a) if a.dtype == bool else torch.as_tensor(a, dtype=torch.float32) for a in packed)
    prober = StationProber()
    stations = torch.tensor([[[0.0, 0.5], [0.0, 2.0]]])
    root = torch.tensor([[0.0, 0.9, 0.0]])
    features = prober(stations, root, torch.tensor([[0.0, 1.0]]), boxes)
    assert features.shape == (1, 2, prober.feature_dim)
    up = [name for name, _ in prober.rays].index("up")
    up_distance = features[0, :, up * 6] * 2.5
    assert up_distance[0] == pytest.approx(2.5) and up_distance[1] == pytest.approx(0.4, abs=1e-5)


def _rep():
    pytest.importorskip("torch")
    pytest.importorskip("kimodo")
    from motionbricks.data.bones_seed import build_representation
    return build_representation()


def make_history(rep, frames=16, speed=1.0, batch=2):
    """Native world frames of a neutral pose gliding forward (+Z) at hip height."""
    import torch
    rotations = torch.eye(3).expand(batch, frames, rep.nbjoints, 3, 3).clone()
    roots = torch.zeros(batch, frames, 3)
    roots[..., 1] = 0.95
    roots[..., 2] = torch.arange(frames) * speed / 30
    contacts = torch.zeros(batch, frames, 4)
    canonical, origin = rep(dict(local_joint_rots=rotations, translation=roots, foot_contacts=contacts),
                            to_normalize=False, return_init_heading_info=True,
                            lengths=torch.full((batch,), frames, dtype=torch.long))
    quat = origin["init_heading_quat"].reshape(batch, 4)
    yaw = 2 * torch.atan2(quat[:, 2], quat[:, 0])
    return rep.change_first_heading(canonical, yaw, False, False)


def _model(rep):
    import torch
    from motionbricks.controller.terrain_motion_transformer import TerrainMotionTransformer
    from motionbricks.controller.tokens import PartRayTable, StationProber
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler
    names = list(rep.skeleton.bone_order_names)
    sampler = SkeletonRaySampler(names, box_capacity=8)
    table, prober = PartRayTable(sampler.layout), StationProber()
    torch.manual_seed(0)
    model = TerrainMotionTransformer(rep, names, table.feature_dim, prober.feature_dim)
    return model, sampler, table, prober, names


def _inputs(rep, sampler, table, history, boxes):
    import torch
    from motionbricks.controller.closed_loop import forward_xz, posed_joints
    mask = torch.ones(history.shape[:2], dtype=torch.bool)
    command = torch.tensor([[0.0, 1.0, 1.0, 0.0]]).expand(history.shape[0], 4)
    joints = posed_joints(rep, history[:, -1:])[:, 0]
    part_rays = table.gather(sampler.observe(joints, forward_xz(history), boxes)["ray_features"])
    return mask, command, part_rays


def test_untrained_transformer_holds_the_pose_and_has_fixed_shapes():
    rep = _rep()
    import torch
    from motionbricks.data.pfnn_frame_actor import advance
    from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d
    model, sampler, table, prober, _ = _model(rep)
    boxes = sampler.boxes([GROUND])
    history = make_history(rep)
    mask, command, part_rays = _inputs(rep, sampler, table, history, boxes)
    plan = model.plan(history, mask, command, part_rays)
    stations = prober(plan["path_world_xz"], history[:, -1, :3], torch.stack((history[:, -1, 4], history[:, -1, 3]), -1), boxes)
    out = model.act(history, plan, stations)
    assert plan["path_world_xz"].shape == (2, 8, 2) and out["local_rot6d"].shape == (2, 30, 6)
    assert out["contact_logits"].shape == (2, 4) and torch.allclose(out["root_delta"], torch.zeros(2, 3))
    assert torch.allclose(out["local_rot6d"], matrix_to_cont6d(model.last_local_rotations(history)), atol=1e-6)
    assert advance(rep, history[:, -4:], out).shape == (2, 4, 366)


def test_transformer_is_translation_invariant_and_ignores_padded_frames():
    rep = _rep()
    import torch
    model, sampler, table, prober, _ = _model(rep)
    for head in (model.rotation_head, model.root_head):
        torch.nn.init.normal_(head.weight, std=0.05)
    model.eval()
    history = make_history(rep)
    shifted = history.clone()
    shifted[..., 0] += 5.0
    shifted[..., 2] -= 3.0
    boxes = sampler.boxes([GROUND])
    outputs = []
    for h in (history, shifted):
        mask, command, part_rays = _inputs(rep, sampler, table, h, boxes)
        plan = model.plan(h, mask, command, part_rays)
        stations = prober(plan["path_world_xz"], h[:, -1, :3], torch.stack((h[:, -1, 4], h[:, -1, 3]), -1), boxes)
        outputs.append((plan, model.act(h, plan, stations)))
    (p0, o0), (p1, o1) = outputs
    assert torch.allclose(p1["path_world_xz"] - p0["path_world_xz"], torch.tensor([5.0, -3.0]).expand(2, 8, 2), atol=1e-4)
    assert torch.allclose(o0["local_rot6d"], o1["local_rot6d"], atol=1e-5)
    assert torch.allclose(o0["root_delta"], o1["root_delta"], atol=1e-6)
    # Frames marked as padding must not influence the output.
    mask, command, part_rays = _inputs(rep, sampler, table, history, boxes)
    mask = mask.clone()
    mask[:, :6] = False
    noisy = history.clone()
    noisy[:, :6] += torch.randn_like(noisy[:, :6])
    a = model.plan(history, mask, command, part_rays)["path_world_xz"]
    b = model.plan(noisy, mask, command, part_rays)["path_world_xz"]
    assert torch.allclose(a, b, atol=1e-5)


def test_closed_loop_rollout_is_differentiable_end_to_end():
    rep = _rep()
    import torch
    from motionbricks.controller.closed_loop import ClosedLoop, Critic, RolloutConfig, lambda_returns
    model, sampler, table, prober, names = _model(rep)
    for head in (model.rotation_head, model.root_head):
        torch.nn.init.normal_(head.weight, std=0.01)
    loop = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=3))
    critic = Critic(loop.critic_input_dim())
    boxes = sampler.boxes([GROUND, box([0, 1.2, 1.0], [1, .1, 1])])
    history = make_history(rep)
    mask = torch.ones(history.shape[:2], dtype=torch.bool)
    commands = torch.tensor([0.0, 1.0, 1.0, 0.0]).expand(2, 3, 4)
    targets = make_history(rep, frames=19)[:, 16:]
    result = loop.rollout(history, mask, commands, boxes, critic=critic, targets=targets)
    assert torch.isfinite(result["policy_loss"]) and torch.isfinite(result["critic_loss"])
    assert set(result["terms"]) >= {"penetration", "support", "skating", "command_velocity", "imitation"}
    result["policy_loss"].backward()
    assert model.temporal.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.rotation_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in critic.parameters())          # bootstrap uses a frozen critic
    result["critic_loss"].backward()
    assert critic.net[0].weight.grad is not None
    values = torch.tensor([[0.0, 10.0, 20.0]])
    assert torch.allclose(lambda_returns(torch.tensor([[1.0, 2.0]]), values, 0.5, 0.5),
                          torch.tensor(lambda_returns_numpy([[1.0, 2.0]], values.numpy(), 0.5, 0.5), dtype=torch.float32))


def test_teacher_closed_loop_and_evaluation_steps_run_on_a_batch():
    rep = _rep()
    import torch
    from motionbricks.controller.closed_loop import ClosedLoop, Critic, RolloutConfig
    from motionbricks.controller.training import closed_loop_step, evaluate, teacher_step
    model, sampler, table, prober, names = _model(rep)
    loop = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=4))
    critic = Critic(loop.critic_input_dim())
    frames = make_history(rep, frames=32)
    packed = sampler.boxes([GROUND, box([0, 1.2, 1.0], [1, .1, 1])])
    batch = {"history": frames[:, :16], "history_mask": torch.ones(2, 16, dtype=torch.bool),
             "targets": frames[:, 16:], "commands": torch.tensor([0.0, 1.0, 1.0, 0.0]).expand(2, 16, 4),
             "box_centers": packed[0].expand(2, -1, -1), "box_half": packed[1].expand(2, -1, -1),
             "box_rotation": packed[2].expand(2, -1, -1, -1), "box_mask": packed[3].expand(2, -1),
             "contacts": torch.ones(2, 16, 4), "group": ["flat_control", "crouch"]}
    loss, terms = teacher_step(loop, batch)
    loss.backward()
    assert torch.isfinite(loss) and "path" in terms and "body_fk" in terms
    policy, critic_loss, terms = closed_loop_step(loop, batch, critic, horizon=4)
    assert torch.isfinite(policy) and torch.isfinite(critic_loss) and "imitation" in terms
    model.eval()
    report = evaluate(loop, batch, horizon=4)
    assert report["persistence_body_cm"] >= 0 and report["closed_loop_root_drift_cm_p50"] >= 0
    assert 0 <= report["closed_loop_joint_penetration_over_2cm_frames"] <= 1
    assert 0 <= report["source_joint_penetration_over_2cm_frames"] <= 1
    assert report["source_planted_sole_speed_mps_p50"] >= 0 and set(report["by_group"]) == {"flat_control", "crouch"}
    assert 0 <= report["source_grounded_sole_fraction"] <= 1 and "closed_loop_grounded_sole_speed_mps_p50" in report
    geometric = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=4, skating_mode="geometric"))
    policy, _, terms = closed_loop_step(geometric, batch, critic, horizon=4)
    assert torch.isfinite(policy) and terms["skating"] >= 0
    anchored = ClosedLoop(model, rep, names, sampler, table, prober,
                          RolloutConfig(horizon=4, contact_source="source", support=2500.0, skating=100.0,
                                        support_gap_clip_m=0.1, penetration_depth_clip_m=0.1,
                                        skating_speed_clip_mps=1.0, bootstrap_cap=50.0))
    normalized = Critic(loop.critic_input_dim(), normalized=True)
    policy, critic_loss, terms = closed_loop_step(anchored, batch, normalized, horizon=4)
    assert torch.isfinite(policy) and torch.isfinite(critic_loss) and terms["support"] >= 0
    model.zero_grad(set_to_none=True)
    policy.backward()                                          # the order the trainer uses
    critic_loss.backward()
    assert model.rotation_head.weight.grad is not None
    assert float(normalized.scale) != 1.0                      # running scale moved toward |targets|
    model.train()
    loss, terms = teacher_step(loop, batch, foot_weight=1.0)
    assert torch.isfinite(loss) and terms["planted_foot"] >= 0
