"""Human training/evaluation policies around the unchanged upstream MB networks."""

import torch
import torch.nn.functional as F

from motionbricks.data.bones_seed import official_sole_offsets, official_sole_surfaces, sole_positions
from motionbricks.data.bones_evaluation import reconstruction_metrics
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.models.root_model import MotionModel as RootModel
from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel, masked_motion_mean
from motionbricks.vqvae.neural_modules.vqvae import VQVAE


def apply_root_velocity_condition(prediction, external, representation):
    """Use the same normalized planar command for training and token decoding."""
    indices = list(representation.indices["local_root_rot_vel"]) + list(representation.indices["local_root_vel"])
    if representation.root_mode != "local" or len(indices) != 3:
        raise ValueError("Root authority requires the planar local-root representation")
    if external is None or external.shape != (*prediction.shape[:2], len(indices)):
        raise ValueError("Dense root velocity condition must match every decoded frame")
    result = prediction.clone()
    result[..., indices] = external
    return result


class ConditionedBonesVQVAE(VQVAE):
    """Official MB layers with a serialized root-output training policy."""

    def __init__(self, root_condition_policy="provided_velocity", **kwargs):
        super().__init__(**kwargs)
        if root_condition_policy not in ("provided_velocity", "learned_refinement") or self.decoder_external_cond_feature_mode != "root_without_hip_height":
            raise ValueError("This adapter requires provided planar velocity and heading-rate conditions")
        self.root_condition_policy = root_condition_policy

    def _apply_condition(self, output, external_cond):
        output["raw_recon_state"] = output["recon_state"]
        if self.root_condition_policy == "provided_velocity":
            output["recon_state"] = apply_root_velocity_condition(output["recon_state"], external_cond, self.motion_rep)
        return output

    def forward(self, x, target_cond, has_target_cond=None, external_cond=None):
        return self._apply_condition(super().forward(x, target_cond, has_target_cond, external_cond), external_cond)

    def forward_decoder(self, x, target_cond, has_target_cond=None, external_cond=None,
                        use_overall_indices=True, token_mask=None):
        return self._apply_condition(super().forward_decoder(x, target_cond, has_target_cond, external_cond,
                                                             use_overall_indices, token_mask), external_cond)


def contact_masked_reconstruction(predicted, target, confidence, contact_indices, frame_mask=None):
    if confidence.shape != (*predicted.shape[:2], len(contact_indices)):
        raise ValueError("Saved contact confidence must match the sampled motion window")
    weights = torch.ones_like(predicted)
    weights[..., contact_indices] = confidence.to(predicted.dtype)
    if frame_mask is not None:
        if frame_mask.dtype != torch.bool or frame_mask.shape != predicted.shape[:2]:
            raise ValueError("Reconstruction frame mask must match the motion")
        weights = weights * frame_mask.detach()[..., None]
    return (F.smooth_l1_loss(predicted, target, reduction="none") * weights).sum() / weights.sum().clamp(min=1)


def stance_anchor_distances(soles, support):
    """Distance to each predicted contiguous stance's centroid, never an inference foot lock."""
    if soles.ndim != 4 or soles.shape[-1] != 3 or support.shape != soles.shape[:-1]:
        raise ValueError("Support mask must match the sole positions")
    batch, frames, channels = support.shape
    support = support.detach().bool()
    starts = support & ~F.pad(support[:, :-1], (0, 0, 1, 0), value=False)
    groups = starts.long().cumsum(1)
    base = torch.arange(batch * channels, device=soles.device).reshape(batch, 1, channels) * (frames + 1)
    indices = (groups + base).reshape(-1)
    weights = support.reshape(-1).to(soles.dtype)
    counts = soles.new_zeros(batch * channels * (frames + 1)).scatter_add(0, indices, weights)
    sums = soles.new_zeros(len(counts), 3).scatter_add(
        0, indices[:, None].expand(-1, 3), soles.reshape(-1, 3) * weights[:, None])
    centers = sums / counts.clamp(min=1)[:, None]
    return (soles - centers[indices].reshape_as(soles)).norm(dim=-1)


def contact_geometry_losses(soles, source_contacts, confidence, fps, penetration_tolerance_m=0.005):
    if soles.ndim != 4 or soles.shape[-1] != 3 or soles.shape[1] < 2:
        raise ValueError("Soles must be [batch, frames>=2, channels, xyz]")
    if source_contacts.shape != soles.shape[:-1] or confidence.shape != source_contacts.shape:
        raise ValueError("Source contacts and confidence must match the sole frames")
    if fps <= 0 or penetration_tolerance_m < 0:
        raise ValueError("Positive FPS and nonnegative penetration tolerance are required")
    support = (source_contacts.detach() > 0.5) & confidence.detach().bool()
    stance = support[:, 1:] & support[:, :-1]
    speed = (soles[:, 1:] - soles[:, :-1]).norm(dim=-1) * fps
    anchor = stance_anchor_distances(soles, support)
    return {
        "l_trusted_skate_mps": (speed * stance).sum() / stance.sum().clamp(min=1),
        "l_sole_penetration_m": F.relu(-soles[..., 1] - penetration_tolerance_m).mean(),
        "l_stance_anchor_m": (anchor * support).sum() / support.sum().clamp(min=1),
        "l_support_height_m": (F.relu(soles[..., 1].abs() - penetration_tolerance_m) * support).sum()
                              / support.sum().clamp(min=1),
    }


SOLE_REGION_SUPERVISION_KEYS = ("sole_region_coeff_contact_height_m", "sole_region_coeff_penetration_m",
                                "sole_region_coeff_slip_mps")
#: Loss names the sole-region supervision replaces. Leaving any of them on would
#: penalise the same geometry twice, so a nonzero value is rejected rather than ignored.
REPLACED_GEOMETRY_COEFFICIENTS = ("trusted_skate_coeff", "penetration_coeff", "stance_anchor_coeff",
                                  "support_height_coeff")


class BonesTokenizer(MotionVQVAEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        offsets, _ = official_sole_offsets(self.motion_rep.skeleton)
        self.register_buffer("sole_offsets", offsets)
        self.sole_region_skinner = None
        self._sole_region_device = None
        self.sole_region_version = None
        if self.args.get("sole_region_supervision", False):
            from motionbricks.data.sole_region_contact import SOLE_LABEL_THRESHOLDS, sole_region_indices
            from motionbricks.data.soma_skin_reference import load_official_skinner
            skeleton = self.motion_rep.skeleton
            surfaces, _, _, mesh_sha256 = official_sole_surfaces(skeleton)
            self.sole_region_version = self.args.get("sole_region_labels_version")
            if not self.sole_region_version:
                raise ValueError("sole_region_supervision needs sole_region_labels_version to record what was used")
            enabled = [name for name in REPLACED_GEOMETRY_COEFFICIENTS if self.args.get(name, 0.0)]
            if enabled:
                raise ValueError("sole_region_supervision replaces " + ", ".join(enabled)
                                 + "; set those coefficients to zero so the geometry is not penalised twice")
            self.sole_region_contract = {"labels_version": self.sole_region_version,
                                         "mesh_sha256": mesh_sha256,
                                         "thresholds": dict(SOLE_LABEL_THRESHOLDS)}
            self.sole_region_skinner = load_official_skinner(skeleton)
            for side in ("left", "right"):
                self.register_buffer(f"sole_region_indices_{side}",
                                     torch.as_tensor(sole_region_indices(surfaces, side.capitalize())))

    def _sole_region_skinner_on(self, device, dtype):
        if self._sole_region_device != device or self._sole_region_dtype != dtype:
            from motionbricks.data.soma_skin_reference import move_skinner
            move_skinner(self.sole_region_skinner, device, dtype)
            self._sole_region_device, self._sole_region_dtype = device, dtype
        return self.sole_region_skinner

    def sole_region_losses(self, batch, pred):
        """Differentiable sole-region supervision, per side, from the batch's own labels."""
        from motionbricks.data.sole_geometry_losses import (region_geometry_from_motion, sole_geometry_supervision,
                                                          stance_displacement_loss)
        skeleton = self.motion_rep.skeleton
        local = pred["local_joint_rots"]
        root = pred["posed_joints"][..., skeleton.root_idx, :]
        batch_size, frames = local.shape[:2]
        skinner = self._sole_region_skinner_on(local.device, local.dtype)
        tolerance = self.args.get("sole_region_penetration_tolerance_m", 0.005)
        totals = {"l_contact_height_m": 0.0, "l_penetration_m": 0.0, "l_measured_slip_mps": 0.0}
        counts = {"contact_frames_supervised": 0, "measured_stable_intervals_supervised": 0,
                  "measured_vertices_supervised": 0}
        indices = [getattr(self, f"sole_region_indices_{side}") for side in ("left", "right")]
        all_height, all_horizontal = region_geometry_from_motion(
            local.reshape(batch_size * frames, *local.shape[2:]), root.reshape(batch_size * frames, 3),
            skinner, torch.cat(indices))
        offset = 0
        for side in ("left", "right"):
            width = len(getattr(self, f"sole_region_indices_{side}"))
            height = all_height[:, offset:offset + width].reshape(batch_size, frames, width)
            horizontal = all_horizontal[:, offset:offset + width].reshape(batch_size, frames, width, 2)
            offset += width
            labels = {}
            for key in ("frame_state", "interval_state", "region_measured", "vertex_measured"):
                name = f"sole_{key}_{side}"
                if name not in batch:
                    raise ValueError(f"Sole-region supervision is on but the batch has no {name}")
                values = batch[name].to(local.device)
                # The trainer already gathered these labels onto the sampled window with
                # the same chosen_ids and frame_indices as the motion, so the lengths must
                # match exactly. Slicing here would silently misalign them.
                length = frames if key == "frame_state" else frames - 1
                if values.shape[1] != length:
                    raise ValueError(f"{name} covers {values.shape[1]} frames but the sampled window has {length}")
                labels[key] = values
            frame_mask = batch.get("motion_frame_mask")
            if frame_mask is not None:
                eligible = frame_mask[:, :-1] & frame_mask[:, 1:]
                labels["region_measured"] = labels["region_measured"] & eligible
                labels["vertex_measured"] = labels["vertex_measured"] & eligible[..., None]
            result = sole_geometry_supervision(height, horizontal,
                                               labels, self.motion_rep.fps,
                                               penetration_tolerance_m=tolerance, frame_mask=frame_mask)
            if self.args.get("sole_region_coeff_stance_displacement_m", 0.0):
                displacement, pairs = stance_displacement_loss(
                    horizontal, labels["interval_state"], labels["region_measured"], labels["vertex_measured"])
                totals["l_stance_displacement_m"] = totals.get("l_stance_displacement_m", 0.0) + displacement
                counts["stance_displacement_pairs"] = counts.get("stance_displacement_pairs", 0) + pairs
            for name in ("l_contact_height_m", "l_penetration_m", "l_measured_slip_mps"):
                totals[name] = totals[name] + result[name]
            for name in ("contact_frames_supervised", "measured_stable_intervals_supervised", "measured_vertices_supervised"):
                counts[name] = counts[name] + result[name]
        return totals, counts

    def loss(self, batch, pose_net_output, root_net_output):
        observed = None
        if "executed_history" in batch:
            if self.pose_net.motion_rep.name != "local":
                raise ValueError("Executed-history supervision requires the native local decoder")
            batch = dict(batch)
            batch["local_motions"] = batch["local_motions"].detach()
            batch["global_motions"] = batch["global_motions"].detach()
            history = batch["executed_history"].detach()
            if history.shape != (*batch["global_motions"].shape[:1], 4, batch["global_motions"].shape[-1]):
                raise ValueError("Executed history must be four raw global frames")
            observed = self.global_motion_rep.inverse(history, False, return_all=False)
            if any(self.args.get(key, 0.) for key in (*REPLACED_GEOMETRY_COEFFICIENTS, "skate_contact_loss_coeff")):
                raise ValueError("Executed-history training requires source-masked sole-region losses, not legacy geometry")
            frame_mask = torch.ones(batch["local_motions"].shape[:2], dtype=torch.bool, device=history.device)
            frame_mask[:, :4] = False
            interval_mask = frame_mask[:, 1:].clone()  # Includes actual history[3] -> prediction[4].
            batch.update(motion_frame_mask=frame_mask, motion_interval_mask=interval_mask,
                         executed_history_positions=observed["posed_joints"])
        losses = super().loss(batch, pose_net_output, root_net_output)
        predicted = pose_net_output["recon_state"]
        target = batch["local_motions"]
        pred = self.local_motion_rep.inverse(predicted, True, return_all=True)
        gt = (self.local_motion_rep.inverse(target, True, return_all=True) if observed is None else
              self.global_motion_rep.inverse(batch["global_motions"].detach(), True, return_all=True))
        frame_mask = batch.get("motion_frame_mask")
        skeleton = self.motion_rep.skeleton
        pred_sole = sole_positions(pred["posed_joints"], pred["global_joint_rots"], skeleton, self.sole_offsets)
        gt_sole = sole_positions(gt["posed_joints"], gt["global_joint_rots"], skeleton, self.sole_offsets)
        confident = batch["contact_confidence"].bool()
        reconstruction = contact_masked_reconstruction(
            predicted, target, confident, self.local_motion_rep.indices["foot_contacts"], frame_mask)
        losses["loss"] += reconstruction - losses["l_recons_pose"]
        losses["l_recons_pose"] = reconstruction
        if self.args.get("sole_region_supervision", False):
            assert self.sole_region_version, "Sole-region supervision must carry a labels version"
            region, region_counts = self.sole_region_losses(batch, pred)
            losses.update(region)
            losses.update({f"sole_region_{name}": float(value) for name, value in region_counts.items()})
            # Only numbers may enter the loss dict: Lightning logs every entry. The
            # version and mesh digests live on the model and in the run contract.
            for name, coeff_key in zip(("l_contact_height_m", "l_penetration_m", "l_measured_slip_mps"),
                                       SOLE_REGION_SUPERVISION_KEYS):
                coefficient = float(self.args.get(coeff_key, 0.0))
                losses[f"sole_region_coeff_{name}"] = coefficient
                losses[f"sole_region_weighted_{name}"] = coefficient * region[name]
                losses["loss"] = losses["loss"] + coefficient * region[name]
            if "l_stance_displacement_m" in region:
                coefficient = float(self.args["sole_region_coeff_stance_displacement_m"])
                losses["sole_region_weighted_l_stance_displacement_m"] = coefficient * region["l_stance_displacement_m"]
                losses["loss"] = losses["loss"] + losses["sole_region_weighted_l_stance_displacement_m"]
        # Saved source labels, not predicted contacts, select support intervals.
        geometry = contact_geometry_losses(pred_sole, gt["foot_contacts"], confident, self.motion_rep.fps)
        losses.update(geometry)
        losses["loss"] += self.args.get("trusted_skate_coeff", 0.0) * geometry["l_trusted_skate_mps"]
        losses["loss"] += self.args.get("penetration_coeff", 0.0) * geometry["l_sole_penetration_m"]
        losses["loss"] += self.args.get("stance_anchor_coeff", 0.0) * geometry["l_stance_anchor_m"]
        losses["loss"] += self.args.get("support_height_coeff", 0.0) * geometry["l_support_height_m"]
        # The engine consumes FK rotations, not the separately reconstructed XYZ features.
        losses["l_fk_position_m"] = masked_motion_mean((pred["posed_joints"] - gt["posed_joints"]).norm(dim=-1), frame_mask)
        losses["l_sole_position_m"] = masked_motion_mean((pred_sole - gt_sole).norm(dim=-1), frame_mask)
        losses["loss"] += self.args.get("fk_position_coeff", 0.0) * losses["l_fk_position_m"]
        losses["loss"] += self.args.get("sole_position_coeff", 0.0) * losses["l_sole_position_m"]
        if observed is not None:
            from motionbricks.data.soma_continuation import continuation_losses
            terms = continuation_losses(pred, gt, observed, self.motion_rep.fps,
                                        future=int(self.args.get("rollout_commit_frames", 8)))
            losses.update(terms)
            coefficient = float(self.args.get("rollout_rotation_increment_coeff", 0.0))
            if coefficient < 0:
                raise ValueError("Rotation increment coefficient must be nonnegative")
            losses["weighted_rotation_increment"] = coefficient * terms["l_rotation_increment_chordal"]
            losses["loss"] += losses["weighted_rotation_increment"]
        consistency_keys = ("kinematic_position_consistency_coeff", "kinematic_velocity_consistency_coeff")
        if any(self.args.get(key, 0.0) for key in consistency_keys):
            from motionbricks.data.kinematic_consistency import kinematic_consistency_losses
            # SOMA's current GlobalJoints representation retains heading, so its
            # velocity channels and reconstructed FK positions use the same axes.
            if self.local_motion_rep.removed_heading:
                raise ValueError("Kinematic consistency requires world-axis GlobalJoints velocities")
            velocity = self.local_motion_rep.unnormalize(predicted)[
                ..., self.local_motion_rep.indices["local_vel"]].reshape_as(pred["posed_joints"])
            consistent = kinematic_consistency_losses(
                pred["posed_joints"], pred["all_posed_joints"]["ric_data"], velocity, self.motion_rep.fps)
            losses.update(consistent)
            for name, key in zip(consistent, consistency_keys):
                coefficient = float(self.args.get(key, 0.0))
                if coefficient < 0:
                    raise ValueError("Consistency coefficients must be nonnegative")
                losses[f"weighted_{name}"] = coefficient * consistent[name]
                losses["loss"] += losses[f"weighted_{name}"]
        return losses

    def validation_step(self, batch, batch_idx):
        frames = self.args["max_tokens"] * self.get_num_frames_per_code()
        global_motion = batch["motion"][:, :frames + 1]
        lengths = torch.full((len(global_motion),), len(global_motion[0]), device=global_motion.device)
        local = self.motion_rep.dual_rep.global_to_local(global_motion, True, True, lengths=lengths)[:, :frames]
        external = extract_feature_from_motion_rep(local, self.local_motion_rep,
                                                   self.pose_net.decoder_external_cond_feature_mode)
        context_mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
        context_mask[:, :4] = True
        output = self.pose_net(local, target_cond=local * context_mask[..., None],
                               has_target_cond=context_mask, external_cond=external)
        confidence = batch["contact_confidence"][:, :frames]
        reconstruction = contact_masked_reconstruction(
            output["recon_state"], local, confidence, self.local_motion_rep.indices["foot_contacts"])
        metrics, _, _ = reconstruction_metrics(output["recon_state"], local,
                                              self.local_motion_rep, self.sole_offsets, confidence, skip_frames=4)
        self.log("val/loss", reconstruction, sync_dist=True, batch_size=batch["batch_size"])
        for name, value in metrics.items():
            if value is not None:
                self.log(f"val/{name}", value, sync_dist=True, batch_size=batch["batch_size"])
        if self.args.get("support_repair", False):
            # A pose-only improvement must not win checkpoint selection while stance remains poor.
            score = metrics["mpjpe_m"] + 0.5 * (metrics["trusted_skate_mps"] or 0)
            score += metrics["trusted_support_clearance_abs_p95_m"] or 0
            self.log("val/support_repair_score", score, sync_dist=True, batch_size=batch["batch_size"])


class BonesRootModel(RootModel):
    def _sample_the_conditions(self, feature, num_frames):
        history = self.backbone_net.get_num_frames_per_token()
        values = torch.cat([feature[:, :history], feature[:, -history:]], 1).clone()
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if feature.shape[-1] > 5:
            # All future body-pose values are unavailable to the root-only backend.
            values[:, history:] = 0
            mask[:, history:] = False
        else:
            height_index = 1 if feature.shape[-1] == 5 else 3
            values[:, history:, height_index] = feature[:, history - 1:history, height_index]
        return values, mask

    def validation_step(self, batch, batch_idx):
        frames = self.args["max_tokens"] * self.backbone_net.get_num_frames_per_token()
        global_motion = batch["motion"][:, :frames + 1]
        lengths = torch.full((len(global_motion),), len(global_motion[0]), device=global_motion.device)
        local = self.motion_rep.dual_rep.global_to_local(global_motion, True, True, lengths=lengths)[:, :frames]
        global_motion = global_motion[:, :frames]
        roots_global = extract_feature_from_motion_rep(global_motion, self.global_motion_rep, "root")
        roots_local = extract_feature_from_motion_rep(local, self.local_motion_rep, "root")
        poses = extract_feature_from_motion_rep(local, self.local_motion_rep, self.args["local_pose_feature"])
        global_values, global_mask = self._sample_the_conditions(roots_global, frames)
        local_values, local_mask = self._sample_the_conditions(roots_local, frames)
        pose_values, pose_mask = self._sample_the_conditions(poses, frames)
        tokens = torch.full((len(global_motion), 1), self.args["max_tokens"], device=local.device, dtype=torch.long)
        output = self.backbone_net(global_values, global_mask, local_values, local_mask,
                                   pose_values, pose_mask, tokens)
        prediction = output["pred_global_root_values"][:, :frames]
        loss = F.smooth_l1_loss(prediction, roots_global)
        predicted_units = self.global_motion_rep.stats.unnormalize(prediction, self.global_motion_rep.indices["root"])
        target_units = self.global_motion_rep.stats.unnormalize(roots_global, self.global_motion_rep.indices["root"])
        root_error = (predicted_units[..., [0, 2]] - target_units[..., [0, 2]]).norm(dim=-1).mean()
        for name, value in {"loss": loss, "root_xz_error_m": root_error}.items():
            self.log(f"val/{name}", value, sync_dist=True, batch_size=batch["batch_size"])
