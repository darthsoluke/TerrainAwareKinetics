"""Thin native SOMA30 state and official SOMA77 rig adapter for v3."""

from pathlib import Path

import torch

from motionbricks.data.soma_lab import pack_rig_frames
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats

from .surface import OfficialSurfaceSampler


REPRESENTATION_NAME = "somaskel30_dual_root_global_joints"
RIG_FORMAT = "soma77-rig-v1"
FRAME_BYTES = (3 + 77 * 16) * 4


def _posed_transforms(transforms):
    return transforms


class NativeSOMAAdapter:
    """No legacy network construction, pose correction or source-root replacement.

    Root deltas use world XYZ metres per 30 Hz step. Rotations use native column
    6D in world axes. The redundant heading head is auxiliary: physical heading
    is derived by the native representation from the authoritative FK body.
    """

    @torch.inference_mode()
    def __init__(self, bundle_path, device="cpu"):
        self.bundle_path = Path(bundle_path).resolve(strict=True)
        # Map the immutable archive; access only stats, skeleton metadata and seed.
        bundle = torch.load(self.bundle_path, map_location="cpu", weights_only=False, mmap=True)
        stats = Stats()
        stats.register_from_tensors(bundle["stats_mean"].clone(), bundle["stats_std"].clone())
        self.representation = GlobalRootGlobalJoints(
            30, SOMASkeleton30(), REPRESENTATION_NAME, stats=stats).to(device)
        self.initial_history = bundle["initial_history"].clone().to(device)
        self.seed_provenance = bundle["seed_provenance"]
        self.skinner = move_skinner(load_official_skinner(self.representation.skeleton), device)
        self.skinner.skeleton_skin.to(device)
        self._official_lbs = self.skinner.lbs
        # Stop at the official skin() rig boundary; LBS is measured separately.
        self.skinner.lbs = _posed_transforms
        self.surface_sampler = OfficialSurfaceSampler(self.skinner, sample_count=512).to(device)

    def skin_asset(self):
        skin = self.skinner
        return dict(format=RIG_FORMAT, rig_count=77, vertex_count=18056,
                    influences_per_vertex=skin.lbs_indices.shape[-1],
                    bind_vertices=skin.bind_vertices.cpu().flatten().tolist(),
                    inverse_bind=skin.bind_rig_transform_inv.cpu().flatten().tolist(),
                    indices=skin.lbs_indices.cpu().flatten().tolist(),
                    weights=skin.lbs_weights.cpu().flatten().tolist(),
                    faces=skin.faces.cpu().flatten().tolist())

    def _frame_from_pose(self, pose):
        joints, rotations = pose["posed_joints"], pose["global_joint_rots"]
        batch, frames = joints.shape[:2]
        rig = self.skinner.skin(rotations.reshape(-1, 30, 3, 3),
                               joints.reshape(-1, 30, 3), rot_is_global=True)
        rig = rig.reshape(batch, frames, 77, 4, 4)
        return dict(root=joints[..., self.representation.skeleton.root_idx, :],
                    rig=rig, joints=joints, body_points=self.surface_sampler(rig),
                    global_joint_rots=rotations, local_joint_rots=pose["local_joint_rots"])

    def source_frame(self, world366):
        """Decode unnormalized native [B,T,366], preserving its world root."""
        pose = self.representation.inverse(world366, False, return_all=False,
                                           joint_positions_from="global_rot_data")
        return self._frame_from_pose(pose)

    def reencode_history(self, history):
        """Derive all redundant channels using existing native representation APIs."""
        rep = self.representation
        pose = rep.inverse(history, False, return_all=False, joint_positions_from="global_rot_data")
        roots = pose["posed_joints"][..., rep.skeleton.root_idx, :]
        lengths = torch.full((history.shape[0],), 4, device=history.device, dtype=torch.long)
        canonical, origin = rep(dict(posed_joints=pose["posed_joints"],
            global_joint_rots=pose["global_joint_rots"], local_joint_rots=pose["local_joint_rots"],
            translation=roots, foot_contacts=history[..., rep.indices["foot_contacts"]]),
            to_normalize=False, return_init_heading_info=True, lengths=lengths)
        quat = origin["init_heading_quat"].reshape(history.shape[0], 4)
        heading = 2 * torch.atan2(quat[:, 2], quat[:, 0])
        world = rep.change_first_heading(canonical, heading, False, False)
        world[..., rep.indices["global_root_pos_2d"]] += roots[:, :1, [0, 2]]
        local = rep.dual_rep.global_to_local(world, False, False, lengths=lengths)
        dual = rep.dual_rep.cat_global_and_local(world, local)
        return dict(history=world, local_history=local, dual_history=dual)

    def step(self, history, actor_output):
        rep = self.representation
        proposal = history[:, -1:].clone()
        proposal[..., rep.indices["global_root_pos"]] += actor_output["root_delta"][:, None]
        proposal[..., rep.indices["global_rot_data"]] = actor_output["global_joint_rot6d"].reshape(-1, 1, 180)
        # Native four foot channels are not measured contact observations.
        proposal[..., rep.indices["foot_contacts"]] = 0
        state = self.reencode_history(torch.cat((history[:, 1:], proposal), dim=1))
        frame = self.source_frame(state["history"][:, -1:])
        return dict(**frame, **state, heading_delta=actor_output["heading_delta"])

    def fullskin(self, rig):
        """All 18056 vertices, every official influence, using official LBS."""
        vertices = self._official_lbs(rig.reshape(-1, 77, 4, 4))
        return vertices.reshape(*rig.shape[:-3], 18056, 3)

    @staticmethod
    def pack_frame(frame):
        """One B=1 stream, little-endian root3 + row-major 77x4x4."""
        return pack_rig_frames(frame["root"][0].detach().cpu().numpy(),
                               frame["rig"][0].detach().cpu().numpy())

    @staticmethod
    def pack_actor_frame(frame, input_valid):
        """Transfer validity and one actor frame together at publication."""
        packed = torch.cat((frame["root"][0].reshape(-1), frame["rig"][0].reshape(-1),
                            input_valid.to(frame["root"].dtype))).detach().cpu().numpy()
        if not bool(packed[-1]):
            raise RuntimeError("Unknown scene or invalid actor input: no capacity frame published")
        return pack_rig_frames(packed[:3][None], packed[3:-1].reshape(1, 77, 4, 4))
