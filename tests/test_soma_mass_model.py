import math

import pytest
import torch

from motionbricks.data.bones_seed import build_representation
from motionbricks.data.soma_mass_model import (MASS_FRACTION_SUM_TOLERANCE, PROVENANCE, SEGMENTS,
                                               mass_fraction_total, segment_center_of_mass_positions,
                                               validate_segments, whole_body_center_of_mass)


@pytest.fixture(scope="module")
def skeleton():
    return build_representation().skeleton


def test_segment_mass_fractions_describe_exactly_one_body():
    assert mass_fraction_total() == pytest.approx(1.0, abs=MASS_FRACTION_SUM_TOLERANCE)


def test_the_model_is_declared_as_not_official_soma_data():
    text = " ".join(str(value) for entry in PROVENANCE.values()
                    for value in (entry.values() if isinstance(entry, dict) else [entry]))
    assert "not measured for SOMA" in text
    assert "Winter" in text
    assert "schematic" in text
    assert "must not be reported as official" in text


@pytest.mark.parametrize("segments", [
    (("a", "Hips", "Head", 0.5, 0.5),),
    (("a", "Hips", "Head", 1.5, 0.5), ("b", "Hips", "Head", -0.5, 0.5)),
    (("a", "Hips", "Head", 1.0, 1.5),),
])
def test_model_rejects_a_table_that_cannot_describe_a_body(segments):
    with pytest.raises(ValueError):
        validate_segments(segments)


def test_model_rejects_a_joint_that_soma30_does_not_have(skeleton):
    with pytest.raises(ValueError, match="no joint named"):
        validate_segments((("a", "Hips", "LeftElbow", 1.0, 0.5),), skeleton)


def test_segment_centres_of_mass_lie_on_their_segments(skeleton):
    pose = skeleton.neutral_joints[None].double()
    points = segment_center_of_mass_positions(pose, skeleton)
    assert points.shape == (1, len(SEGMENTS), 3)
    for index, (_, proximal, distal, _, com) in enumerate(SEGMENTS):
        head = pose[0, skeleton.bone_index[proximal]]
        tail = pose[0, skeleton.bone_index[distal]]
        torch.testing.assert_close(points[0, index], head + com * (tail - head))


def test_center_of_mass_is_neither_the_pelvis_nor_a_joint_mean(skeleton):
    pose = skeleton.neutral_joints[None].double()
    com = whole_body_center_of_mass(pose, skeleton)[0]
    pelvis = pose[0, skeleton.bone_index["Hips"]]
    assert (com - pelvis).norm() > 0.01
    assert (com - pose[0].mean(0)).norm() > 0.05


def test_center_of_mass_is_mass_weighted_not_an_unweighted_joint_average(skeleton):
    pose = skeleton.neutral_joints[None].double().expand(2, -1, -1).contiguous()
    pose[1, skeleton.bone_index["LeftHandMiddleEnd"], 0] += 1.0
    com = whole_body_center_of_mass(pose, skeleton)
    shift = float(com[1, 0] - com[0, 0])
    mass, fraction = [(segment[3], segment[4]) for segment in SEGMENTS
                      if segment[2] == "LeftHandMiddleEnd"][0]
    assert shift == pytest.approx(mass * fraction, abs=1e-9)
    # Averaging all joint positions would move by 1 / 30 m instead, forty times more.
    unweighted = float((pose[1, :, 0] - pose[0, :, 0]).mean())
    assert unweighted == pytest.approx(1.0 / skeleton.nbjoints, abs=1e-9)
    assert abs(shift - unweighted) > 0.02


def test_standard_pose_center_of_mass_is_anatomically_plausible(skeleton):
    pose = skeleton.neutral_joints[None].double()
    com = whole_body_center_of_mass(pose, skeleton)[0]
    lowest = pose[0, :, 1].min()
    head = pose[0, skeleton.bone_index["Head"], 1]
    height_fraction = (com[1] - lowest) / (head - lowest)
    # A standing adult carries the centre of mass near 0.55-0.60 of stature above
    # the sole. The band is wide because this model is schematic and the Head joint
    # is the head centre, not the crown.
    assert 0.55 <= float(height_fraction) <= 0.70


def test_center_of_mass_follows_rigid_translation_and_yaw(skeleton):
    pose = skeleton.neutral_joints[None].double().expand(3, -1, -1).contiguous()
    pose[1] += torch.tensor([3.0, 1.0, -2.0], dtype=pose.dtype)
    cosine, sine = math.cos(0.9), math.sin(0.9)
    yaw = torch.tensor([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=pose.dtype)
    pose[2] = pose[2] @ yaw.T
    com = whole_body_center_of_mass(pose, skeleton)
    torch.testing.assert_close(com[0], com[1] - torch.tensor([3.0, 1.0, -2.0], dtype=pose.dtype))
    torch.testing.assert_close(com[2], com[0] @ yaw.T)
