"""Adjacent-window history learning; reuse MB token prediction and CE targets."""

import numpy as np
import torch

from motionbricks.data.bones_control import BonesPoseModel
from motionbricks.data.bones_rollout import control_inputs, load_control_models
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep, sample_motion_segments_from_motion_clips


def transport_target_to_history(source, generated_history, representation):
    """Apply one common planar rigid transform, never move the ground or snap a foot."""
    angle = torch.atan2(generated_history[:, 0, 4], generated_history[:, 0, 3])
    transformed = representation.change_first_heading(source, angle, True, False)
    transformed[..., representation.indices["global_root_pos_2d"]] += generated_history[:, :1, [0, 2]]
    return representation.normalize(transformed)


class BonesHistoryPoseModel(BonesPoseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history_generator_holder = []
        self.history_counts = {"teacher": 0, "generated": 0}

    def history_generator(self, device):
        if not self._history_generator_holder:
            devices = [device.index or 0] if device.type == "cuda" else []
            # Loading frozen networks must not change training's paired sampling RNG.
            with torch.random.fork_rng(devices=devices):
                generator, _ = load_control_models(self.args["history_root_run"],
                                                   self.args["history_generator_pose_run"],
                                                   self.motion_rep, device=str(device))
            self._history_generator_holder.append(generator)
        return self._history_generator_holder[0]

    def training_step(self, batch, batch_idx, **kwargs):
        device = batch["motion"].device
        count = len(batch["motion"])
        requested = int(np.random.choice(np.arange(self.args["min_tokens"], self.args["max_tokens"] + 1)))
        enough = int((batch["motion_len"].sort().values[count // 2] - 5) // 4)
        tokens = min(requested, enough)
        if tokens < self.args["min_tokens"]:
            raise ValueError("Too few adjacent frames for the authorized history protocol")
        frames = tokens * 4
        info = {}
        adjacent = sample_motion_segments_from_motion_clips(batch["motion"], batch["motion_len"],
                                                            frames + 4, 1, info=info,
                                                            motion_rep=self.global_motion_rep)
        angle = torch.rand(count, device=device) * (2 * torch.pi)
        adjacent = self.global_motion_rep.change_first_heading(adjacent, angle, True, True)
        current = adjacent[:, 4:].clone()
        raw = self.global_motion_rep.unnormalize(adjacent)
        draws = torch.rand(count, device=device)
        progress = min(1.0, float(self.global_step) / self.args["history_ramp_steps"])
        probability = self.args["history_max_probability"] * progress
        use_generated = draws < probability if self.args["history_policy"] == "generated" else draws < 0
        observed = self.global_motion_rep.unnormalize(current[:, :4])
        if use_generated.any():
            generator = self.history_generator(device)
            with torch.no_grad():
                previous_local = self.motion_rep.dual_rep.global_to_local(
                    raw, False, False, lengths=torch.full((count,), frames + 5, device=device))
                goal = raw[:, frames - 4:frames, :5]
                velocity = previous_local[:, frames - 4:frames, 1:3].mean(1)
                predicted, _ = generator.predict(*control_inputs(raw[:, :4], goal, velocity, self.motion_rep),
                                                   tokens, config={"pose_token_sampling_use_argmax": True})
                generated = predicted[:, 4:8].detach()
                transported = transport_target_to_history(current, generated, self.global_motion_rep)
                current = torch.where(use_generated[:, None, None], transported, current)
                observed = torch.where(use_generated[:, None, None], generated, observed)
        current_raw = self.global_motion_rep.unnormalize(current)
        origin = observed[:, :1, [0, 2]].clone()
        current_raw[..., self.global_motion_rep.indices["global_root_pos_2d"]] -= origin
        observed = observed.clone()
        observed[..., self.global_motion_rep.indices["global_root_pos_2d"]] -= origin
        current = self.global_motion_rep.normalize(current_raw)
        local = self.motion_rep.dual_rep.global_to_local(current, True, True,
                                                        lengths=torch.full((count,), frames + 1, device=device))[:, :frames]
        with torch.no_grad():
            target = self.pose_net.encode_into_idx(local, fetch_overall_indices=False)
        history_local = self.motion_rep.dual_rep.global_to_local(observed, False, True,
                                                                lengths=torch.full((count,), 4, device=device))
        conditional_local = local.clone()
        conditional_local[:, :4] = history_local
        pose_values, pose_mask, _, _ = self._sample_the_local_pose_conditions(conditional_local, frames)
        conditional_global = current[:, :frames].clone()
        conditional_global[:, :4] = self.global_motion_rep.normalize(observed)
        root = extract_feature_from_motion_rep(conditional_global, self.global_motion_rep,
                                               self.args["cond_root_feature"])
        focused, masked, incorrect, correct, fraction, wrong_fraction = self._get_token_masks(
            {"groundtruth_pose_tokens": target})
        model_batch = {"groundtruth_pose_tokens": target, "focused_token_mask": focused}
        inputs = torch.full_like(target, self.backbone_net.POSE_MASK_ID)
        output = self.backbone_net(inputs, root, pose_values, pose_mask,
                                   torch.full((count, 1), tokens, device=device), None, None)
        losses = self.loss(model_batch, output)
        self.history_counts["generated"] += int(use_generated.sum())
        self.history_counts["teacher"] += int((~use_generated).sum())
        self.log("loss/train_loss", losses["loss"], on_step=True, batch_size=count)
        self.log("history/generated_fraction", use_generated.float().mean(), on_step=True, batch_size=count)
        return losses["loss"]
