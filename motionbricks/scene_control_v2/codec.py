"""Native SOMA codec adapter with explicit-head VQ and selected official skinning."""

import copy

import torch
from torch import Tensor, nn

from .quantizer import ExplicitHeadQuantizer, FrozenCodebookIdentity, tensor_state_sha256


def native_window_transform(init_heading_info: dict, vertical_origin_m: Tensor) -> Tensor:
    """Restore native initial-heading/XZ canonicalization plus the recorded Y origin."""
    from motionbricks.motionlib.core.utils.rotations import quaternion_to_matrix
    heading = init_heading_info["init_heading_quat"].detach()
    xz = init_heading_info["root_pos_init_xz"].detach()
    batch = heading.shape[0]
    if heading.shape != (batch, 4) or xz.shape != (batch, 2) or vertical_origin_m.shape != (batch,):
        raise ValueError("Expected batched native heading [B,4], XZ [B,2], and vertical origin [B]")
    transform = torch.eye(4, dtype=heading.dtype, device=heading.device).expand(batch, 4, 4).clone()
    transform[:, :3, :3] = quaternion_to_matrix(heading)
    transform[:, [0, 2], 3] = xz.to(heading)
    transform[:, 1, 3] = vertical_origin_m.detach().to(heading)
    return transform


class SceneControlCodec(nn.Module):
    """Reuse native feature/encoder/decoder APIs without its combined-index quantizer.

    The supplied native codec is a construction/warmstart donor. Only its encoder,
    decoder and representation are retained by reference. Supply an independently
    constructed/loaded donor, not a live actor shared with another task.
    """

    def __init__(self, native_codec, quantizer: ExplicitHeadQuantizer, contact_channels: int):
        super().__init__()
        self.encoder = native_codec.encoder
        self.decoder = native_codec.decoder
        self.quantizer = quantizer
        self.motion_rep = native_codec.motion_rep
        self.encoder_feature_mode = native_codec.encoder_input_feature_mode
        self.target_feature_mode = native_codec.decoder_target_cond_feature_mode
        self.frames_per_token = 2 ** native_codec._down_t
        self.contact_head = nn.Linear(len(self.motion_rep.indices["all"]), contact_channels)
        self.register_buffer("vocabulary_frozen", torch.tensor(False))

    def _features(self, motion: Tensor, mode: str) -> Tensor:
        from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
        return extract_feature_from_motion_rep(motion, self.motion_rep, mode)

    def _encode_latents(self, motion: Tensor) -> Tensor:
        return self.encoder(self._features(motion, self.encoder_feature_mode).transpose(1, 2).contiguous())

    def _decode_latents(self, latents: Tensor, *, target_cond, has_target_cond, external_cond) -> dict:
        from motionbricks.helper.data_training_util import convert_sparse_cond_to_dense_cond_if_needed
        if target_cond is not None:
            target_cond, has_target_cond = convert_sparse_cond_to_dense_cond_if_needed(
                target_cond, has_target_cond, latents.shape[-1] * self.frames_per_token)
            target_cond = self._features(target_cond, self.target_feature_mode)
        decoded = self.decoder(latents, external_cond, target_cond, has_target_cond).transpose(1, 2).contiguous()
        return {"recon_state": decoded, "contact_logits": self.contact_head(decoded)}

    def forward(self, motion: Tensor, *, target_cond, has_target_cond, external_cond) -> dict:
        if motion.shape[1] % self.frames_per_token:
            raise ValueError("Codec frame length must be divisible by frames_per_token")
        quantized = self.quantizer(self._encode_latents(motion))
        output = self._decode_latents(quantized.latents, target_cond=target_cond,
                                      has_target_cond=has_target_cond, external_cond=external_cond)
        output.update(indices=quantized.indices, l_commit=quantized.commitment, perplexity=quantized.perplexity)
        return output

    @torch.no_grad()
    def encode(self, motion: Tensor) -> Tensor:
        if not bool(self.vocabulary_frozen):
            raise RuntimeError("Freeze the vocabulary before generating cached labels")
        if motion.shape[1] % self.frames_per_token:
            raise ValueError("Codec frame length must be divisible by frames_per_token")
        return self.quantizer.encode(self._encode_latents(motion))

    def decode(self, indices: Tensor, *, target_cond, has_target_cond, external_cond) -> dict:
        latents = self.quantizer.dequantize(indices).transpose(1, 2).contiguous()
        return self._decode_latents(latents, target_cond=target_cond,
                                    has_target_cond=has_target_cond, external_cond=external_cond)

    def freeze_vocabulary(self, representation_id: str, *, normalization_sha256: str,
                          parent: FrozenCodebookIdentity | None = None) -> FrozenCodebookIdentity:
        identity = FrozenCodebookIdentity(
            quantizer_config=self.quantizer.config, codebook_sha256=self.quantizer.codebook_sha256(),
            encoder_sha256=tensor_state_sha256(self.encoder.state_dict()), representation_id=representation_id,
            normalization_sha256=normalization_sha256, parent_id=None if parent is None else parent.identity_id)
        self.quantizer.freeze()
        self.vocabulary_frozen.fill_(True)
        self.encoder.eval().requires_grad_(False)
        if parent is not None and identity.identity_id == parent.identity_id:
            return parent
        return identity

    def train(self, mode: bool = True):
        super().train(mode)
        if bool(self.vocabulary_frozen):
            self.encoder.eval()
            self.quantizer.eval()
        return self


class SomaGeometryAdapter(nn.Module):
    """Metric FK plus a fixed vertex subset of the official differentiable SOMA LBS.

    sample_vertex_indices is a versioned training sample, including contact points.
    Full-skin and swept/substep acceptance stay outside the training adapter.
    """

    def __init__(self, motion_rep, skinner, sample_vertex_indices: Tensor):
        super().__init__()
        self.motion_rep = motion_rep
        self.skeleton_input = skinner.skeleton_input
        self.skeleton_skin = skinner.skeleton_skin
        indices = torch.as_tensor(sample_vertex_indices, dtype=torch.long, device=skinner.bind_vertices.device)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("A nonempty explicit skin vertex sample is required")
        self.register_buffer("sample_vertex_indices", indices.clone())
        self.register_buffer("bind_vertices", skinner.bind_vertices.index_select(0, indices).clone())
        self.register_buffer("lbs_indices", skinner.lbs_indices.index_select(0, indices).clone())
        self.register_buffer("lbs_weights", skinner.lbs_weights.index_select(0, indices).clone())
        self.register_buffer("bind_rig_transform_inv", skinner.bind_rig_transform_inv.clone())
        # Keep the official LBS function, with selected tensor inputs in model buffers.
        self._selected_skinner = copy.copy(skinner)

    def forward(self, recon_state: Tensor, *, world_from_window: Tensor) -> dict[str, Tensor]:
        from kimodo.skeleton.kinematics import batch_rigid_transform
        geometry = self.motion_rep.inverse(recon_state, is_normalized=True, return_quat=False,
                                           return_all=False, joint_positions_from="global_rot_data")
        joints = geometry["posed_joints"]
        local = geometry["local_joint_rots"]
        batch, frames = local.shape[:2]
        local77 = self.skeleton_input.to_SOMASkeleton77(local.reshape(-1, 30, 3, 3))
        neutral = self.skeleton_skin.neutral_joints.to(local).expand(batch * frames, -1, -1)
        posed, global_rots = batch_rigid_transform(
            local77, neutral, self.skeleton_skin.joint_parents.cpu(), self.skeleton_skin.root_idx)
        root = joints[:, :, self.skeleton_input.root_idx].reshape(-1, 1, 3)
        posed = posed + root
        transforms = torch.eye(4, dtype=local.dtype, device=local.device).expand(
            batch * frames, self.skeleton_skin.nbjoints, 4, 4).clone()
        transforms[..., :3, :3] = global_rots
        transforms[..., :3, 3] = posed
        for name in ("bind_vertices", "lbs_indices", "lbs_weights", "bind_rig_transform_inv"):
            setattr(self._selected_skinner, name, getattr(self, name))
        samples = self._selected_skinner.lbs(transforms).reshape(batch, frames, -1, 3)
        if world_from_window.shape != (batch, 4, 4):
            raise ValueError("world_from_window must be an explicit fixed [B,4,4] transform")
        restore = world_from_window.detach().to(joints)
        rotation, translation = restore[:, :3, :3], restore[:, :3, 3]
        world_joints = torch.einsum("bij,btkj->btki", rotation, joints) + translation[:, None, None]
        world_samples = torch.einsum("bij,btsj->btsi", rotation, samples) + translation[:, None, None]
        world_local = local.clone()
        world_local[:, :, self.skeleton_input.root_idx] = rotation[:, None] @ local[:, :, self.skeleton_input.root_idx]
        return {"joint_positions_m": world_joints, "body_samples_m": world_samples,
                "local_rot_mats": world_local}
