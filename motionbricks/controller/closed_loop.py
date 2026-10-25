"""Differentiable closed-loop training for the frame model (P2 with analytic gradients).

The kinematic controller is its own dynamics: the next state is the model output passed
through official FK (pfnn_frame_actor.advance), and the scene costs are analytic signed
distances (scene_sdf). So, as in SuperMimic's imagination step but without a learned world
model, a short rollout of H frames on the model's own outputs is differentiated end to end
(backpropagation through time), and a critic bootstraps the cost beyond H.

Per step costs (all per sample, [B]):

* imitation: only where a source future exists and the command equals the source command
  (mask); root displacement and local rotations of the source frame;
* command: planar velocity and facing against the left/right stick command;
* penetration: FK joints (as spheres) entering any solid;
* support: soles of feet the model says are in contact, kept on the nearest surface;
* skating: horizontal sole speed, weighted either by predicted contact ("contact", v1/v2)
  or by how close the sole is to a surface in this and the previous frame ("geometric":
  the model cannot avoid it by predicting no contact);
* smoothness: joint acceleration.

The policy minimises sum_t gamma^t c_t + gamma^H V(s_H), with gradients through every
generated frame and through V(s_H). The critic is trained on detached TD(lambda) targets.
Nothing here starts training; a bounded plan and the preflight gates come first.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from motionbricks.data.pfnn_frame_actor import FPS, advance
from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix

from .returns import lambda_returns, lambda_returns_numpy  # noqa: F401  (re-exported)
from .scene_sdf import box_sdf, penetration_cost
from .terrain_motion_transformer import body_command


@dataclass(frozen=True)
class RolloutConfig:
    horizon: int = 16
    gamma: float = 0.97
    lam: float = 0.95
    joint_radius_m: float = 0.03
    foot_radius_m: float = 0.0
    imitation: float = 1.0
    command_velocity: float = 0.5
    command_heading: float = 0.2
    penetration: float = 100.0
    support: float = 20.0
    skating: float = 1.0
    skating_mode: str = "contact"            # "contact" or "geometric"
    skating_height_m: float = 0.03           # geometric: sole within this of a surface counts as planted
    skating_softness_m: float = 0.01
    contact_source: str = "predicted"        # "predicted" (model's contact) or "source" (source labels where
                                             # a source frame is a valid target; predicted elsewhere)
    smoothness: float = 0.01
    bootstrap: bool = True
    # Robustness (v5). Defaults keep v1-v4 behaviour. Caps bound every per-step cost when the
    # rollout has drifted far from the data, so one bad sample cannot dominate the batch.
    support_gap_clip_m: float = float("inf")
    penetration_depth_clip_m: float = float("inf")
    skating_speed_clip_mps: float = float("inf")
    bootstrap_cap: float = float("inf")       # upper bound on the bootstrapped cost-to-go


def forward_xz(history):
    """World XZ forward of the last frame; same convention as VolumetricEgoSampler."""
    return torch.stack((history[:, -1, 4], history[:, -1, 3]), -1)


def posed_joints(rep, frames):
    return rep.inverse(frames, False, return_all=False, joint_positions_from="global_rot_data")["posed_joints"]


def default_sole_offsets(rep):
    """[4, 3] sole offsets from the foot joints (LeftFoot, LeftToe, RightFoot, RightToe), bind pose."""
    from motionbricks.data.bones_seed import official_sole_offsets
    offsets, _ = official_sole_offsets(rep.skeleton)
    return torch.as_tensor(np.asarray(offsets), dtype=torch.float32).reshape(4, 3)


def joint_radii(joint_names, config):
    """Sphere radius per joint for the penetration cost; feet and toes may touch surfaces."""
    feet = ("Foot", "Toe", "Ankle")
    return torch.tensor([config.foot_radius_m if any(f in n for f in feet) else config.joint_radius_m
                         for n in joint_names])


class Critic(nn.Module):
    """Cost-to-go of a state: last normalized frame, heading-frame command, body rays.

    With normalized=True the network predicts value / scale, where scale is a running mean
    of |target| updated from each batch of TD targets; the loss is computed in that unit,
    so large or growing costs do not blow up the critic's own gradients.
    """

    def __init__(self, input_dim, width=256, normalized=False, momentum=0.99):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 1))
        self.normalized, self.momentum = normalized, momentum
        self.register_buffer("scale", torch.ones(()))

    def forward(self, state):
        out = self.net(state)[:, 0]
        return out * self.scale if self.normalized else out

    def frozen(self, state):
        """Value with detached weights: gradients reach the state (the policy), not the critic."""
        out = state
        for layer in self.net:
            out = F.linear(out, layer.weight.detach(), layer.bias.detach()) if isinstance(layer, nn.Linear) else layer(out)
        out = out[:, 0]
        return out * self.scale.detach() if self.normalized else out

    def loss(self, states, targets):
        """Regression loss on detached states [list of B x D] against targets [B, H]."""
        if self.normalized:
            with torch.no_grad():
                observed = targets.abs().mean().clamp_min(1e-3)
                self.scale.mul_(self.momentum).add_((1 - self.momentum) * observed)
            predicted = torch.stack([self.net(s.detach())[:, 0] for s in states], 1)
            return F.mse_loss(predicted, targets / self.scale)
        predicted = torch.stack([self(s.detach()) for s in states], 1)
        return F.mse_loss(predicted, targets)


class ClosedLoop:
    """Wires model, FK and scene queries for one rollout; owns no trainable weights."""

    def __init__(self, model, rep, joint_names, ray_sampler, part_table, station_prober, config=None,
                 sole_offsets=None):
        self.model, self.rep = model, rep
        self.sampler, self.table, self.prober = ray_sampler, part_table, station_prober
        self.config = config or RolloutConfig()
        self.foot_index = torch.as_tensor(rep.skeleton.foot_joint_idx).long().cpu()
        self.radii = joint_radii(joint_names, self.config)
        self.sole_offsets = default_sole_offsets(rep) if sole_offsets is None else torch.as_tensor(sole_offsets)

    def critic_input_dim(self):
        return self.model.config.frame_dim + 4 + len(self.table.table) * self.table.feature_dim

    def observe(self, history, command, boxes):
        """Body rays and critic state from the last emitted frame (no future information)."""
        joints = posed_joints(self.rep, history[:, -1:])[:, 0]
        rays = self.sampler.observe(joints, forward_xz(history), boxes)["ray_features"]
        part_rays = self.table.gather(rays)
        state = torch.cat((self.model.normalized_history(history)[:, -1], body_command(history, command),
                           part_rays.flatten(1)), -1)
        return joints, part_rays, state

    def step(self, history, history_mask, command, boxes, volume=None):
        """One emitted frame: plan, query the proposed path, act, append via official FK."""
        _, part_rays, state = self.observe(history, command, boxes)
        plan = self.model.plan(history, history_mask, command, part_rays, volume)
        stations = self.prober(plan["path_world_xz"], history[:, -1, :3], forward_xz(history), boxes)
        pred = self.model.act(history, plan, stations)
        appended = advance(self.rep, history[:, -4:], pred)[:, -1:]
        new_history = torch.cat((history[:, 1:], appended), 1)
        new_mask = torch.cat((history_mask[:, 1:], history_mask.new_ones(history_mask.shape[0], 1)), 1)
        return pred, new_history, new_mask, state

    def _contact_index(self, device):
        return torch.as_tensor(self.rep.indices["foot_contacts"], device=device).long().flatten()

    def _planted_weight(self, contact, soles, soles_before, boxes):
        config = self.config
        if config.skating_mode == "contact":
            return contact
        if config.skating_mode != "geometric":
            raise ValueError(f"Unknown skating mode {config.skating_mode!r}")
        near = [torch.sigmoid((config.skating_height_m - box_sdf(points, boxes)) / config.skating_softness_m)
                for points in (soles, soles_before)]
        return near[0] * near[1]

    def soles(self, joints):
        return joints[:, self.foot_index.to(joints.device)] + self.sole_offsets.to(joints)

    def step_costs(self, pred, history_before, history_after, command, boxes, joints_before, joints_two_back,
                   target=None, target_previous=None, imitation_mask=None):
        """Weighted per-sample costs [B] and their unweighted parts."""
        config = self.config
        joints = posed_joints(self.rep, history_after[:, -1:])[:, 0]
        command_body = body_command(history_before, command)
        c, s = history_before[:, -1, 3], history_before[:, -1, 4]
        delta = pred["root_delta"]
        velocity_body = torch.stack((c * delta[:, 0] - s * delta[:, 2], s * delta[:, 0] + c * delta[:, 2]), -1) * FPS
        facing = history_after[:, -1, 3:5]
        target_facing = command[:, 2:4]
        contact = pred["contact_logits"].sigmoid()
        if config.contact_source == "source" and target is not None:
            # Source heel/toe contacts anchor support and skating to what the actor did; the
            # model cannot lower these costs by lifting its feet or predicting no contact.
            labels = target.index_select(-1, self._contact_index(target.device))
            valid = imitation_mask[:, None]
            contact = valid * labels + (1 - valid) * contact
        elif config.contact_source not in ("predicted", "source"):
            raise ValueError(f"Unknown contact source {config.contact_source!r}")
        soles, soles_before = self.soles(joints), self.soles(joints_before)
        parts = {
            "command_velocity": (velocity_body - command_body[:, :2]).square().sum(-1),
            "command_heading": 1 - (facing * target_facing).sum(-1),
            "penetration": penetration_cost(joints, boxes, self.radii.to(joints),
                                            max_depth=None if config.penetration_depth_clip_m == float("inf")
                                            else config.penetration_depth_clip_m),
            "support": (contact * box_sdf(soles, boxes).clamp(-config.support_gap_clip_m,
                                                               config.support_gap_clip_m).square()).mean(-1),
            "skating": (self._planted_weight(contact, soles, soles_before, boxes)
                        * ((soles - soles_before)[..., [0, 2]] * FPS).square().sum(-1)
                        .clamp_max(config.skating_speed_clip_mps ** 2)).mean(-1),
            # Joint acceleration in units of 10 m/s^2.
            "smoothness": ((joints - 2 * joints_before + joints_two_back) * FPS * FPS / 10).square().sum(-1).mean(-1),
        }
        if target is not None:
            decoded = self.rep.inverse(target[:, None], False, return_all=False,
                                       joint_positions_from="global_rot_data")
            target_delta = target[:, :3] - target_previous[:, :3]
            rotation = (cont6d_to_matrix(pred["local_rot6d"]) - decoded["local_joint_rots"][:, 0]).square()
            imitation = (F.smooth_l1_loss(delta / .05, target_delta / .05, reduction="none").sum(-1)
                         + rotation.flatten(1).mean(-1))
            parts["imitation"] = imitation * imitation_mask
        weights = {name: getattr(config, name) for name in parts}
        total = sum(weights[name] * value for name, value in parts.items())
        return total, parts, joints

    def rollout(self, history, history_mask, commands, boxes, critic=None, targets=None, imitation_mask=None,
                volume=None):
        """Differentiable H-step rollout. commands [B, H, 4]; targets [B, H, 366] (source frames).

        The last four history frames must be real (advance rebuilds features from them).
        imitation_mask [B, H] selects steps whose source frame is a valid target (same
        command as the source, scene unchanged); it defaults to all ones when targets exist.
        Use separate optimisers: policy_loss for the model, critic_loss for the critic.
        """
        config = self.config
        if targets is not None and imitation_mask is None:
            imitation_mask = torch.ones(targets.shape[:2], device=targets.device, dtype=torch.bool)
        horizon = commands.shape[1]
        joints_before = posed_joints(self.rep, history[:, -1:])[:, 0]
        joints_two_back = posed_joints(self.rep, history[:, -2:-1])[:, 0]
        costs, states, logs = [], [], {}
        target_previous = history[:, -1]
        for t in range(horizon):
            command = commands[:, t]
            pred, new_history, history_mask, state = self.step(history, history_mask, command, boxes, volume)
            total, parts, joints = self.step_costs(
                pred, history, new_history, command, boxes, joints_before, joints_two_back,
                None if targets is None else targets[:, t], target_previous,
                None if imitation_mask is None else imitation_mask[:, t].to(history.dtype))
            if targets is not None:
                target_previous = targets[:, t]
            costs.append(total)
            states.append(state)
            for name, value in parts.items():
                logs.setdefault(name, []).append(value.detach().mean())
            joints_two_back, joints_before, history = joints_before, joints, new_history
        _, _, final_state = self.observe(history, commands[:, -1], boxes)
        states.append(final_state)
        costs = torch.stack(costs, 1)                                                  # [B, H]
        discount = config.gamma ** torch.arange(horizon, device=costs.device, dtype=costs.dtype)
        objective = (costs * discount).sum(1)
        result = {"costs": costs.detach(), "history": history, "history_mask": history_mask,
                  "terms": {name: torch.stack(v).mean() for name, v in logs.items()}}
        if critic is not None:
            if config.bootstrap:
                value = critic.frozen(final_state).clamp(0.0, config.bootstrap_cap)
                objective = objective + config.gamma ** horizon * value
            with torch.no_grad():
                values = torch.stack([critic(s.detach()) for s in states], 1)          # [B, H+1]
                targets_td = lambda_returns(costs.detach(), values, config.gamma, config.lam)
            result["critic_loss"] = critic.loss(states[:-1], targets_td)
        result["policy_loss"] = objective.mean()
        return result
