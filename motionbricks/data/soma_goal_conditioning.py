"""Sparse MB pose goals and a train-only reference-keyframe primitive.

MB 3/6.1 supplies recent context plus one or two style keyframes to Root, Pose
and decoder. The velocity lookup below is a deliberately minimal reference
primitive, not the unpublished UE blendspace implementation or motion playback.
"""

import torch

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


def pose_conditions(local, representation, mode, endpoint=False, goal_local=None):
    """Dense normalized conditions: four observed frames and optionally one goal.

    Source endpoints are legal TRAINING conditions only. Inference must pass a
    separately supplied goal; changing unobserved source body cannot change it.
    """
    if local.ndim != 3 or local.shape[1] < 8 or local.shape[1] % 4:
        raise ValueError("Whole-token segment with four history and future frames required")
    feature = extract_feature_from_motion_rep(local, representation, mode)
    mask = torch.zeros(local.shape[:2], device=local.device, dtype=torch.bool)
    mask[:, :4] = True
    values = torch.zeros_like(feature)
    values[:, :4] = feature[:, :4]
    if endpoint and goal_local is not None:
        raise ValueError("Training source endpoint and external goal are mutually exclusive")
    if endpoint:
        values[:, -1] = feature[:, -1].detach()
        mask[:, -1] = True
    elif goal_local is not None:
        if goal_local.shape != (len(local), 4, local.shape[-1]):
            raise ValueError("External goal requires four normalized local feature frames")
        values[:, -1] = extract_feature_from_motion_rep(goal_local, representation, mode)[:, -1].detach()
        mask[:, -1] = True
    return values, mask


def library_records(records):
    selected = [r for r in records if r["split"] == "train" and r["frames_60hz"] >= 129]
    if not selected:
        raise ValueError("Reference keyframes require complete train clips")
    return selected


class ReferencePoseLibrary:
    """Select a single raw reference pose by commanded heading-local XZ velocity.

    No source future, skeleton fitting, phase matching, pose blending or IK.
    Canonicalization/placement use the official MB representation's yaw math.
    """

    def __init__(self, payload, representation, device):
        if payload.get("schema") != "motionbricks.train_pose_library.v1":
            raise ValueError("Unknown reference library")
        provenance = payload["provenance"]
        if not provenance or any(p["split"] != "train" for p in provenance):
            raise ValueError("Reference library contains non-training data")
        self.features = payload["features"].to(device)
        self.velocities = payload["local_velocity_xz_mps"].to(device)
        if (self.features.ndim != 2 or self.velocities.shape != (len(self.features), 2)
                or len(provenance) != len(self.features)
                or not torch.isfinite(self.features).all() or not torch.isfinite(self.velocities).all()):
            raise ValueError("Invalid reference-keyframe features")
        self.representation, self.provenance = representation, provenance

    @torch.no_grad()
    def select(self, velocity_xz, heading_rad, endpoint_root):
        velocity = torch.as_tensor(velocity_xz, device=self.features.device,
                                   dtype=self.features.dtype).reshape(-1, 2)
        heading = torch.as_tensor(heading_rad, device=self.features.device,
                                  dtype=self.features.dtype).reshape(-1)
        if endpoint_root.shape != (len(velocity), 4, 5) or len(heading) != len(velocity):
            raise ValueError("Reference goals must match the command batch")
        # Convert the command with the same native yaw operation as body poses,
        # rather than guessing the representation's heading sign convention.
        origin = self.features[:1].expand(len(velocity), -1).clone()
        origin[:, :5] = 0
        origin[:, 3], origin[:, 4] = heading.cos(), heading.sin()
        probe = origin[:, None].expand(-1, 2, -1).clone()
        probe[:, 1, [0, 2]] = velocity
        canonical = self.representation.change_first_heading(probe, torch.zeros_like(heading), False, False)
        query = canonical[:, 1, [0, 2]]
        ids = (query[:, None] - self.velocities[None]).square().sum(-1).argmin(-1)
        target = self.representation.change_first_heading(
            self.features[ids, None], heading, False, False).expand(-1, 4, -1).clone()
        target[..., [0, 2]] = endpoint_root[..., [0, 2]]
        mask = torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
        mask[:, -1] = True
        return target, mask, ids
