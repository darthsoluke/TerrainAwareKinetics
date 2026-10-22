"""Schematic SOMA30 segment mass model for offline centre-of-mass diagnostics.

The official SOMA skeleton and skin assets ship no segment masses, no inertia
tensors and no centre-of-mass definitions; nothing in this module is official SOMA
data. Each value is either a cited literature number or an explicitly flagged
schematic assumption, and the model exists only so that an approximate whole-body
centre of mass can be built for offline force diagnostics.

Approximation assumptions, all unverified against SOMA:

* The 14 segments below are the classical anthropometric split. SOMA30's clavicle
  (Shoulder), spine (Spine1, Spine2), neck (Neck1, Neck2), jaw and eye joints carry
  no separate mass; they are folded into the trunk or the head+neck segment.
* The foot segment runs from the ankle joint to the ball of the foot, because
  SOMA30 has no toe-tip joint. It is therefore several centimetres shorter than the
  anthropometric foot. The foot is 1.45% of body mass, so the effect on the total
  centre of mass is small, but it is not zero.
* Segment mass is lumped at a fixed fraction along a straight proximal-to-distal
  line. Real segment centres of mass are not on that line.
* The pelvis is not the centre of mass and joint positions are never averaged.
  Only this segment model produces a centre of mass.
"""

import torch

MASS_FRACTION_SUM_TOLERANCE = 1e-9

#: name, proximal joint, distal joint, mass / total body mass, COM fraction from the proximal joint
SEGMENTS = (
    ("trunk", "Hips", "Chest", 0.497, 0.500),
    ("head_neck", "Chest", "Head", 0.081, 0.500),
    ("left_upper_arm", "LeftArm", "LeftForeArm", 0.028, 0.436),
    ("left_forearm", "LeftForeArm", "LeftHand", 0.016, 0.430),
    ("left_hand", "LeftHand", "LeftHandMiddleEnd", 0.006, 0.506),
    ("right_upper_arm", "RightArm", "RightForeArm", 0.028, 0.436),
    ("right_forearm", "RightForeArm", "RightHand", 0.016, 0.430),
    ("right_hand", "RightHand", "RightHandMiddleEnd", 0.006, 0.506),
    ("left_thigh", "LeftLeg", "LeftShin", 0.100, 0.433),
    ("left_shank", "LeftShin", "LeftFoot", 0.0465, 0.433),
    ("left_foot", "LeftFoot", "LeftToeBase", 0.0145, 0.500),
    ("right_thigh", "RightLeg", "RightShin", 0.100, 0.433),
    ("right_shank", "RightShin", "RightFoot", 0.0465, 0.433),
    ("right_foot", "RightFoot", "RightToeBase", 0.0145, 0.500),
)

PROVENANCE = {
    "mass_fractions": {
        "source": "Winter, D. A. (2009), Biomechanics and Motor Control of Human Movement, 4th ed., "
                  "Table 4.1 segment mass fractions (male, Dempster-based). The head+neck, trunk, "
                  "upper-arm, forearm, hand, thigh, shank and foot entries are the tabulated values.",
        "internal_check": "The 14 fractions sum to exactly 1.000, which a unit test asserts.",
        "status": "literature value, not measured for SOMA and not shipped with the official SOMA assets",
    },
    "com_fractions": {
        "source": "Upper arm 0.436, forearm 0.430, hand 0.506, thigh 0.433, shank 0.433 and foot 0.500 are "
                  "Winter's tabulated values. The trunk and head+neck fractions are schematic 0.500 "
                  "choices made here and are NOT quoted from a table.",
        "status": "partly literature, partly unverified schematic assumption",
    },
    "joint_mapping": {
        "source": "Chosen here by reading the SOMA30 joint names and parent chain; the official assets "
                  "ship no segment definitions to compare against.",
        "status": "unverified approximation",
    },
    "overall": "Approximate and schematic. It is adequate for checking whether a required ground force has "
               "the right order of magnitude and the right sign pattern. It must not be reported as official "
               "SOMA data, and it must not be used to claim a measured load distribution.",
}


def mass_fraction_total(segments=SEGMENTS):
    """Total body mass fraction covered by a segment table."""
    return sum(segment[3] for segment in segments)


def validate_segments(segments, skeleton=None):
    """Reject a segment table that cannot describe a whole body."""
    if not segments:
        raise ValueError("At least one segment is required")
    total = mass_fraction_total(segments)
    if abs(total - 1.0) > MASS_FRACTION_SUM_TOLERANCE:
        raise ValueError("Segment mass fractions must sum to 1, got " + repr(total))
    for name, proximal, distal, mass, com in segments:
        if mass <= 0 or not 0.0 <= com <= 1.0:
            raise ValueError(name + ": a positive mass and a COM fraction in [0, 1] are required")
        if skeleton is not None:
            for joint in (proximal, distal):
                if joint not in skeleton.bone_index:
                    raise ValueError(name + ": SOMA30 has no joint named " + joint)
    return segments


def segment_center_of_mass_positions(posed_joints, skeleton, segments=SEGMENTS):
    """Straight-line segment centres of mass, shaped [..., frames, segments, xyz]."""
    validate_segments(segments, skeleton)
    if posed_joints.ndim < 2 or posed_joints.shape[-1] != 3:
        raise ValueError("Posed joints must be [..., frames, joints, xyz]")
    points = []
    for _, proximal, distal, _, com in segments:
        head = posed_joints[..., skeleton.bone_index[proximal], :]
        tail = posed_joints[..., skeleton.bone_index[distal], :]
        points.append(head + com * (tail - head))
    return torch.stack(points, dim=-2)


def whole_body_center_of_mass(posed_joints, skeleton, segments=SEGMENTS):
    """Mass-weighted centre of mass of the schematic body, shaped [..., frames, xyz].

    Joint positions are never averaged: each segment contributes its own lumped point
    mass. The pelvis is not treated as the centre of mass.
    """
    validate_segments(segments, skeleton)
    points = segment_center_of_mass_positions(posed_joints, skeleton, segments)
    weights = torch.tensor([segment[3] for segment in segments],
                           dtype=posed_joints.dtype, device=posed_joints.device)
    return (points * weights[..., None]).sum(dim=-2) / weights.sum()
