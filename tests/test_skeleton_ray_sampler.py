"""Skeleton rays: analytic box hits, heading frame, fixed shapes, torch == numpy."""

import numpy as np
import pytest

from motionbricks.data.skeleton_ray_sampler import (RayLayout, RayNode, cast_rays_numpy, default_layout,
                                                    features_numpy, pack_boxes)

NAMES = ["Hips", "Chest", "Head", "LeftHand", "RightHand", "LeftForeArm", "RightForeArm", "LeftShin",
         "RightShin", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]


def box(center, half, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # columns: local axes in world
    return {"center_xyz_m": list(center), "half_extent_xyz_m": list(half), "rotation_local_to_world": rotation.tolist()}


def standing(height=1.0):
    joints = np.zeros((1, len(NAMES), 3))
    joints[0, NAMES.index("Hips")] = [0, height, 0]
    joints[0, NAMES.index("Chest")] = [0, height + .3, 0]
    joints[0, NAMES.index("Head")] = [0, height + .6, 0]
    for side, x in (("Left", -.2), ("Right", .2)):
        joints[0, NAMES.index(f"{side}Hand")] = [x, height - .1, 0]
        joints[0, NAMES.index(f"{side}ForeArm")] = [x, height + .1, 0]
        joints[0, NAMES.index(f"{side}Shin")] = [x / 2, height - .5, 0]
        joints[0, NAMES.index(f"{side}Foot")] = [x / 2, .08, 0]
        joints[0, NAMES.index(f"{side}ToeBase")] = [x / 2, .03, .12]
    return joints


def ray(result, layout, label):
    return {key: value[0, layout.labels.index(label)] for key, value in result.items()}


def test_ceiling_above_the_head_is_seen_with_a_downward_normal():
    layout = RayLayout(NAMES)
    boxes = pack_boxes([box([0, 2.0, 0], [2, .1, 2])], 8)          # bottom at 1.9, head at 1.6
    result = cast_rays_numpy(standing(), [[0, 1]], layout, boxes)
    head = ray(result, layout, "Head:up")
    assert head["hit"] and head["distance"] == pytest.approx(0.3)
    assert np.allclose(head["normal"], [0, -1, 0])
    hips_down = ray(result, layout, "Hips:down")
    assert not hips_down["hit"] and hips_down["distance"] == pytest.approx(1.2)


def test_walls_are_reported_in_the_heading_frame():
    layout = RayLayout(NAMES)
    # Facing +x, the character's right is -z; a wall face at z = -0.4.
    boxes = pack_boxes([box([0, 1, -0.5], [2, 1, .1])], 8)
    result = cast_rays_numpy(standing(), [[1, 0]], layout, boxes)
    chest = ray(result, layout, "Chest:right")
    assert chest["hit"] and chest["distance"] == pytest.approx(0.4)
    features = features_numpy(result, [[1, 0]], layout).reshape(layout.rays, 6)
    row = features[layout.labels.index("Chest:right")]
    assert row[0] == pytest.approx(0.4 / 0.6) and np.allclose(row[1:4], [-1, 0, 0]) and row[4] == 1
    assert not ray(result, layout, "Chest:left")["hit"]


def test_node_inside_a_solid_and_midpoint_nodes():
    layout = RayLayout(NAMES, [RayNode(("Hips", "LeftShin"), (("down", .6),)),
                               RayNode(("LeftFoot",), (("down", .5),))])
    joints = standing()
    boxes = pack_boxes([box([0, .05, 0], [1, .1, 1])], 4)            # top at 0.15: foot at 0.08 is inside
    result = cast_rays_numpy(joints, [[0, 1]], layout, boxes)
    thigh, foot = (ray(result, layout, label) for label in ("Hips+LeftShin:down", "LeftFoot:down"))
    expected_origin = (joints[0, NAMES.index("Hips")] + joints[0, NAMES.index("LeftShin")]) / 2
    assert np.allclose(thigh["origin"], expected_origin)
    assert thigh["distance"] == pytest.approx(expected_origin[1] - .15) and not thigh["inside"]
    assert foot["inside"] and foot["distance"] == 0 and np.allclose(foot["normal"], 0)


def test_rotated_boxes_match_brute_force_marching():
    rng = np.random.default_rng(4)
    layout = RayLayout(NAMES)
    boxes_list = [box(rng.uniform([-1.5, 0, -1.5], [1.5, 2.5, 1.5]), rng.uniform(.1, .5, 3), rng.uniform(0, np.pi))
                  for _ in range(12)]
    boxes = pack_boxes(boxes_list, 16)
    joints = standing() + rng.normal(0, .05, standing().shape)
    forward = [[np.sin(.7), np.cos(.7)]]
    result = cast_rays_numpy(joints, forward, layout, boxes)
    centers, half, rotation, mask = boxes
    steps = np.linspace(0, 1, 4001)
    for r in range(layout.rays):
        points = result["origin"][0, r] + steps[:, None] * layout.max_length[r] * result["direction"][0, r]
        local = np.einsum("pki,kij->pkj", points[:, None] - centers, rotation)
        inside = ((np.abs(local) <= half).all(-1) & mask).any(-1)
        if inside[0]:
            assert result["inside"][0, r]
        elif inside.any():
            assert result["hit"][0, r]
            assert result["distance"][0, r] == pytest.approx(steps[inside.argmax()] * layout.max_length[r],
                                                             abs=layout.max_length[r] / 2000)
        else:
            assert not result["hit"][0, r]


def test_layout_shapes_and_validation():
    layout = RayLayout(NAMES)
    assert 40 <= layout.rays <= 64 and layout.feature_dim == layout.rays * 6
    assert len(default_layout()) == 3 + 2 * 8
    with pytest.raises(ValueError):
        RayLayout(NAMES[:-1])
    with pytest.raises(ValueError):
        pack_boxes([box([0, 0, 0], [1, 1, 1])] * 5, 4)


def test_torch_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler
    rng = np.random.default_rng(9)
    boxes_list = [box(rng.uniform([-1.5, 0, -1.5], [1.5, 2.5, 1.5]), rng.uniform(.1, .5, 3), rng.uniform(0, np.pi))
                  for _ in range(20)]
    joints = np.concatenate([standing() + rng.normal(0, .05, standing().shape) for _ in range(3)])
    forward = rng.normal(size=(3, 2))
    sampler = SkeletonRaySampler(NAMES, box_capacity=32)
    reference = cast_rays_numpy(joints, forward, sampler.layout, pack_boxes(boxes_list, 32))
    features = features_numpy(reference, forward, sampler.layout)
    out = sampler.observe(torch.tensor(joints, dtype=torch.float64), torch.tensor(forward, dtype=torch.float64),
                          sampler.boxes(boxes_list, dtype=torch.float64))
    for key in ("distance", "normal", "hit", "inside"):
        assert np.allclose(out[key].numpy(), reference[key], atol=1e-9), key
    assert np.allclose(out["features"].numpy(), features, atol=1e-9)
    assert out["features"].shape == (3, sampler.layout.feature_dim)
