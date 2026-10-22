import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MOTIONBRICKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOTIONBRICKS))
from motionbricks.data.lafan_resolved_motion import (  # noqa: E402
    decode_geno_bind_bvh,
    decode_lafan_resolved_bvh,
)


def bvh(rows, *, frames=None, frame_time="0.02", offset="0 10 0"):
    if frames is None:
        frames = len(rows)
    body = "\n".join(rows)
    return f"""HIERARCHY
ROOT Hips
{{
 OFFSET 1 2 3
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Knee
 {{
  OFFSET {offset}
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT ToeBase
  {{
   OFFSET 0 0 10
   CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
   End Site
   {{
    OFFSET 0 0 0
   }}
  }}
  End Site
  {{
   OFFSET 0 0 0
  }}
 }}
}}
MOTION
Frames: {frames}
Frame Time: {frame_time}
{body}
""".encode()


def write_bvh(tmp_path, payload):
    path = tmp_path / "motion.bvh"
    path.write_bytes(payload)
    return path


def test_decodes_all_joint_translations_euler_order_and_fk(tmp_path):
    rows = [
        "100 200 300 90 0 0  4 5 6 0 0 90  7 8 9 0 0 0",
        "101 202 303 0 90 0  7 8 9 30 20 10  10 11 12 45 15 5",
    ]
    result = decode_lafan_resolved_bvh(
        write_bvh(tmp_path, bvh(rows)), distance_unit="cm"
    )

    np.testing.assert_allclose(
        result["local_positions"][0], [[1, 2, 3], [.04, .05, .06], [.07, .08, .09]]
    )
    assert result["source_skeleton"]["names"] == ["Hips", "Knee", "ToeBase"]
    assert result["source_skeleton"]["parents"] == [-1, 0, 1]
    np.testing.assert_allclose(result["source_skeleton"]["header_offsets"][1], [0, .1, 0])
    assert result["source_fps"] == pytest.approx(50)

    angles = np.array([
        [90, 0, 0], [0, 0, 90], [0, 0, 0],
        [0, 90, 0], [30, 20, 10], [45, 15, 5],
    ])
    expected_local = Rotation.from_euler("ZYX", angles, degrees=True).as_matrix().reshape(
        2, 3, 3, 3
    )
    np.testing.assert_allclose(result["local_rot_mats"], expected_local, atol=1e-12)
    assert not np.allclose(result["local_rot_mats"][0, 0], np.eye(3))

    # Independent recursive FK reference for two translated descendants.
    expected_positions = np.empty_like(result["posed_joints"])
    expected_global_rots = np.empty_like(result["global_rot_mats"])
    parents = [-1, 0, 1]
    for frame in range(2):
        for joint, parent in enumerate(parents):
            if parent == -1:
                expected_positions[frame, joint] = result["local_positions"][frame, joint]
                expected_global_rots[frame, joint] = expected_local[frame, joint]
            else:
                expected_positions[frame, joint] = (
                    expected_positions[frame, parent]
                    + expected_global_rots[frame, parent]
                    @ result["local_positions"][frame, joint]
                )
                expected_global_rots[frame, joint] = (
                    expected_global_rots[frame, parent] @ expected_local[frame, joint]
                )
    np.testing.assert_allclose(result["posed_joints"], expected_positions, atol=1e-12)
    np.testing.assert_allclose(result["global_rot_mats"], expected_global_rots, atol=1e-12)
    assert result["posed_joints"][0, 1, 0] == pytest.approx(.95)
    assert not np.allclose(result["posed_joints"][0, 2], [0, 0, 0])


def test_geno_bind_preserves_nonidentity_rotation(tmp_path):
    payload = bvh(["0 0 0 0 0 0  0 1 0 15 25 35  0 0 1 0 0 0"])
    result = decode_geno_bind_bvh(write_bvh(tmp_path, payload), distance_unit="cm")
    expected = Rotation.from_euler("ZYX", [15, 25, 35], degrees=True).as_matrix()
    np.testing.assert_allclose(result["local_rot_mats"][0, 1], expected, atol=1e-12)
    assert not np.allclose(result["local_rot_mats"][0, 1], np.eye(3))


@pytest.mark.parametrize(
    "payload, message",
    [
        (bvh(["0 0 0 0 0 0 1 2 3 4 5"]), "shape"),
        (bvh(["0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11", "0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11"], frames=3), "shape"),
        (bvh(["0 0 nan 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12"]), "non-finite"),
    ],
)
def test_rejects_invalid_motion_rows(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        decode_lafan_resolved_bvh(write_bvh(tmp_path, payload), distance_unit="cm")


@pytest.mark.parametrize("unit", [None, "", "centimeters", "in"])
def test_requires_explicit_supported_distance_unit(tmp_path, unit):
    row = "0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0"
    with pytest.raises(ValueError, match="distance_unit"):
        decode_lafan_resolved_bvh(write_bvh(tmp_path, bvh([row])), distance_unit=unit)


def test_rejects_nonfinite_offsets_and_bad_channel_order(tmp_path):
    row = "0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0"
    bad_offset = bvh([row], offset="0 nan 0")
    with pytest.raises(ValueError, match="OFFSET"):
        decode_lafan_resolved_bvh(write_bvh(tmp_path, bad_offset), distance_unit="cm")

    bad_channels = bvh([row]).replace(
        b"Zrotation Yrotation Xrotation", b"Xrotation Yrotation Zrotation"
    )
    with pytest.raises(ValueError, match="channel order"):
        decode_lafan_resolved_bvh(write_bvh(tmp_path, bad_channels), distance_unit="cm")
