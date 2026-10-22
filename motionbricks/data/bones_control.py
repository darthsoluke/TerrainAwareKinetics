"""SOMA adapters for the official MB Root/Pose networks and token decoder."""

import hashlib
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.bones_models import BonesRootModel
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motion_backbone.models.pose_model import MotionModel as PoseModel


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_tokenizer(run, representation, expected_sha256=None):
    run = Path(run)
    checkpoint = run / "checkpoints/final.ckpt"
    if expected_sha256 and file_sha256(checkpoint) != expected_sha256:
        raise ValueError("Frozen tokenizer checkpoint differs from the protocol")
    config = OmegaConf.load(run / "upstream_config_overrides.yaml")
    network = instantiate(config.model.pose_vqvae_network,
                          motion_rep=representation.dual_rep.local_motion_rep)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    network.load_state_dict({name.removeprefix("pose_net."): value for name, value in state.items()
                             if name.startswith("pose_net.")}, strict=True)
    network.requires_grad_(False).eval()
    return network, config.model.pose_vqvae_network


class BonesControlRootModel(BonesRootModel):
    def _sample_the_conditions(self, feature, num_frames):
        values, mask = super()._sample_the_conditions(feature, num_frames)
        if self.args.get("body_goal_policy") == "single_endpoint":
            # Restore one native in-betweening goal, not four dense target poses.
            if feature.shape[-1] > 5:
                values[:, -1] = feature[:, -1]
                mask[:, -1] = True
            else:
                height = 1 if feature.shape[-1] == 5 else 3
                values[:, -1, height] = feature[:, -1, height]
        if feature.shape[-1] == 4:
            history = self.backbone_net.get_num_frames_per_token()
            # The last observed velocity cannot use the first unavailable future frame.
            values[:, history - 1, :3] = values[:, history - 2, :3]
        return values, mask


class BonesPoseModel(PoseModel):
    """Train masked future tokens with observed body history and freeze the tokenizer."""

    def _load_vqvae_models(self):
        self._vqvae_model_loaded = True
        if not self.backbone_net.initted.item():
            self.backbone_net.init_embedding_from_codebooks(
                self._supporting_networks["pose_net"].get_codebook(), None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pose_net = self._supporting_networks["pose_net"]
        # Upstream assigns False to the method name rather than calling it.
        self.pose_net.__dict__.pop("requires_grad_", None)
        self.pose_net.requires_grad_(False).eval()
        self.frozen_tokenizer_digest = state_digest(self.pose_net)

    def train(self, mode=True):
        super().train(mode)
        self.pose_net.eval()
        return self

    def _sample_the_local_pose_conditions(self, local_motions, num_frames, text_embedding=None):
        from motionbricks.data.soma_goal_conditioning import pose_conditions
        values, mask = pose_conditions(local_motions, self.local_motion_rep,
                                      self.args["local_pose_feature"],
                                      endpoint=self.args.get("body_goal_policy") == "single_endpoint")
        return values, mask, None, None

    def _get_token_masks(self, batch):
        target = batch["groundtruth_pose_tokens"]
        future = torch.ones_like(target, dtype=torch.bool)
        empty = torch.zeros_like(future)
        masked = torch.ones_like(future)
        fraction = masked.flatten(1).float().mean(-1, keepdim=True)
        # Even the first source token can contain future information via noncausal convolutions.
        return future, masked, empty, empty, fraction, torch.zeros_like(fraction)

    def on_train_epoch_start(self):
        self.pose_net.eval()

    def on_train_end(self):
        if state_digest(self.pose_net) != self.frozen_tokenizer_digest:
            raise RuntimeError("Tokenizer parameters or EMA changed during Pose training")

    def predict_local(self, local_motion, global_root, goal_local=None):
        frames = local_motion.shape[1]
        tokens = frames // self.backbone_net.get_num_frames_per_token()
        heads = self.backbone_net.get_num_heads()[0]
        ids = torch.full((len(local_motion), tokens, heads), self.backbone_net.POSE_MASK_ID,
                         device=local_motion.device, dtype=torch.long)
        from motionbricks.data.soma_goal_conditioning import pose_conditions
        values, mask = pose_conditions(local_motion, self.local_motion_rep,
                                      self.args["local_pose_feature"], goal_local=goal_local)
        root_cond = extract_feature_from_motion_rep(global_root, self.global_motion_rep,
                                                    self.args["cond_root_feature"])
        lengths = torch.full((len(local_motion), 1), tokens, device=local_motion.device, dtype=torch.long)
        output = self.backbone_net(ids, root_cond, values, mask, lengths, None, None)
        predicted = output["pose_logits"].argmax(-1)
        # Decode the same supplied conditions, not hidden source-body features.
        decoder_values = torch.zeros_like(local_motion)
        feature_indices = extract_feature_from_motion_rep(None, self.local_motion_rep,
                                                          self.args["local_pose_feature"], fetch_feat_idx=True)
        decoder_values[..., feature_indices] = values
        decoded = decode_tokens(self.pose_net, decoder_values, mask, predicted,
                                self.motion_rep.dual_rep.global_to_local(
                                    global_root, True, True,
                                    lengths=lengths.flatten() * self.backbone_net.get_num_frames_per_token()))
        return output, decoded, predicted

    def validation_step(self, batch, batch_idx):
        frames = self.args["max_tokens"] * self.backbone_net.get_num_frames_per_token()
        motion = batch["motion"][:, :frames + 1]
        local = self.motion_rep.dual_rep.global_to_local(
            motion, True, True, lengths=batch["motion_len"].clamp(max=frames + 1))[:, :frames]
        target = self.pose_net.encode_into_idx(local, fetch_overall_indices=False)
        output, decoded, ids = self.predict_local(local, motion[:, :frames])
        loss = torch.nn.functional.cross_entropy(output["pose_logits"].reshape(-1, output["pose_logits"].shape[-1]),
                                                 target.reshape(-1).long())
        predicted_pose = self.local_motion_rep.inverse(decoded, True, return_all=False)["posed_joints"]
        source_pose = self.local_motion_rep.inverse(local, True, return_all=False)["posed_joints"]
        error = (predicted_pose[:, 4:] - source_pose[:, 4:]).norm(dim=-1).mean()
        for name, value in {"loss": loss, "pose_token_accuracy": (ids == target).float().mean(),
                            "source_root_pose_mpjpe_m": error}.items():
            self.log("val/" + name, value, batch_size=len(motion), sync_dist=True)


def decode_tokens(tokenizer, pose_values, pose_mask, tokens, local_root, token_mask=None):
    """Use the checkpoint's actual feature contract, including heading rate."""
    external = extract_feature_from_motion_rep(local_root, tokenizer.motion_rep,
                                               tokenizer.decoder_external_cond_feature_mode)
    condition = extract_feature_from_motion_rep(pose_values, tokenizer.motion_rep,
                                                tokenizer.decoder_target_cond_feature_mode)
    return tokenizer.forward_decoder(tokens, condition, pose_mask, external,
                                      use_overall_indices=False, token_mask=token_mask)["recon_state"]


class SomaMotionInference(motion_inference):
    """Official MB root and token prediction, adapted to the frozen SOMA decoder."""

    def _sample_tokens_with_highest_prob(self, pose_tokens, pose_tokens_prob,
                                         pred_num_tokens, step, num_pose_inference_steps):
        # Preserve the released cosine confidence schedule, with a real batch axis.
        batch, count, heads = pose_tokens.shape
        active = torch.arange(count, device=pose_tokens.device)[None, :, None] < pred_num_tokens.reshape(batch, 1, 1)
        scores = torch.where(active, pose_tokens_prob, torch.inf).flatten(1)
        fraction = torch.cos(torch.tensor(step / num_pose_inference_steps * torch.pi * 0.5,
                                          device=pose_tokens.device))
        number = (fraction * pred_num_tokens.reshape(batch, 1) * heads).long().clamp(min=1)
        rank_mask = torch.arange(count * heads, device=pose_tokens.device)[None] < number
        ordered = scores.argsort(-1)
        selected = torch.zeros_like(rank_mask).scatter(1, ordered, rank_mask)
        return pose_tokens.masked_fill(selected.reshape_as(pose_tokens), self._pose_model.backbone_net.POSE_MASK_ID)

    def _predict_root_trajectories(self, batch, config):
        if config.get("debug_ground_truth_root_trajectories") is not None:
            raise ValueError("Control mode cannot substitute a source future trajectory")
        native = getattr(self, "use_native_root_stage", True)
        output = self._root_model.backbone_net(
            batch["global_root_values"], batch["has_global_root_values"],
            batch["local_root_values"], batch["has_local_root_values"],
            batch["local_poses"], batch["has_local_poses"], batch["num_tokens"],
            text_embeddings=batch["text_embeddings"], has_text_embeddings=batch["has_text_embeddings"],
            allowed_pred_num_tokens=batch["allowed_pred_num_tokens"], config=config)
        lengths = output["pred_num_tokens"].reshape(-1, 1)
        global_root = output["pred_global_root_values"]
        if not native:
            global_root = global_root.clone()
            global_root[:, :4] = batch["global_root_values"][:, :4]
        local_root = self.motion_rep.dual_rep.global_to_local(
            global_root, True, True, lengths=lengths.flatten() * 4)
        if native:
            # Released MB differentiation/final-goal handling, with an explicit
            # batch axis: upstream repeat([1, 4]) fails for B>1 vector lengths.
            dimensions = len(self.local_motion_rep.indices["root"])
            end = (lengths.long() * self._pose_model.backbone_net.get_num_frames_per_token() - 1)
            indices = end[:, :, None].expand(-1, -1, dimensions)
            estimated = local_root.gather(1, indices)
            provided = batch["has_local_root_values"][:, -1:, None].float()
            final = provided * batch["local_root_values"][:, -1:] + (1 - provided) * estimated
            local_root = local_root.scatter(1, indices, final)
        return lengths, global_root, local_root

    def _decode_motions_from_predicted_root_and_pose_tokens(self, batch, config, info):
        if config.get("final_root_pred_mode", "from_pose_module") != "from_pose_module":
            raise ValueError("Use the conditioned decoder's root once; no second root replacement")
        local = batch["pred_local_root_values"]
        full = local.new_zeros((*local.shape[:2], len(self.local_motion_rep.indices["all"])))
        indices = extract_feature_from_motion_rep(None, self.local_motion_rep,
                                                  self.INTERNAL_POSE_FEATURE_MODE, fetch_feat_idx=True)
        full[..., indices] = batch["pred_pose_cond"]
        token_mask = torch.arange(batch["pred_pose_tokens"].shape[1], device=local.device)[None] < batch["pred_num_tokens"].reshape(-1, 1)
        if info.get("capture_decoder_inputs", False):
            # Training reuses precisely the deployed Root/Pose inputs, detached.
            info["decoder_inputs"] = tuple(value.detach().clone() for value in
                (full, batch["pred_has_pose_cond"], batch["pred_pose_tokens"], local, token_mask))
        decoded = decode_tokens(self._vqvae_pose_model, full, batch["pred_has_pose_cond"],
                                batch["pred_pose_tokens"], local, token_mask=token_mask)
        if info.get("capture_decoder_output", False):
            info["decoder_output"] = decoded
        lengths = batch["pred_num_tokens"].flatten() * 4
        return self.local_motion_rep.unnormalize(decoded), self.motion_rep.dual_rep.local_to_global(
            decoded, True, False, lengths=lengths)
