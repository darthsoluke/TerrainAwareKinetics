"""Thin kinematic adapter around the released MotionBricks inference pipeline."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from motionbricks.exp_setup.experiment import test as load_models
from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motion_backbone.demo.clips import clip_holder_G1
from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
from motionbricks.motion_backbone.inference.motion_inference import motion_inference


SCHEMA = "motionbricks.kinematic.v1"
COORDINATES = "RH_Y_UP_Z_FORWARD_X_LEFT_METERS"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class KinematicBackend(full_navigation_agent):
    """Reuse official constraints/planning, but never construct a MuJoCo converter.

    Each plan continues from the last four generated features, not from physics.
    Only the startup context and target constraints come from the published cache.
    """

    def __init__(self, result_dir="out", device="cuda"):
        torch.nn.Module.__init__(self)
        result_dir = Path(result_dir).resolve()
        args = argparse.Namespace(result_dir=str(result_dir), data_root=None, EXP="default",
                                  return_model_configs=True, return_dataloader=False)
        models, configs = load_models(args)
        # FK composes many matrices; avoid TF32 drift in the transport pose.
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(1)
        self.checkpoints = {}
        for name in ("root", "pose"):
            path = Path(configs[name].ckpt_path)
            models[name].load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
            self.checkpoints[name] = {"filename": path.name, "sha256": file_sha256(path)}
        vqvae = Path(configs["pose"].model.args.vqvae_model_ckpt_path)
        self.checkpoints["vqvae"] = {"filename": vqvae.name, "sha256": file_sha256(vqvae)}
        self._device = device
        self._inferencer = motion_inference(models, models["pose"].args, device=device).eval()
        self._motion_rep = self._inferencer.motion_rep
        self._fps = self._motion_rep.fps
        self._clip_holder = clip_holder_G1(ckpt_path=str(result_dir / "G1-clip.ckpt"), converter=None).to(device)
        # Cache qpos is unused; remove it so accidental simulator fallback fails immediately.
        del self._clip_holder.mujoco_qpos
        self._target_root_realignment = True
        self._source_root_realignment = True
        self._speed_scale_min = self._speed_scale_max = 1.0
        self.FORCE_CANONICALIZATION = False
        self.SKIP_ENDING_TARGET_COND = False
        self.manifest = self._make_manifest()
        self.model_id = hashlib.sha256(json.dumps(self.checkpoints, sort_keys=True).encode()).hexdigest()
        self.reset()

    def reset(self):
        self.context = self._clip_holder.motion_feature[0, :4][None].clone()
        self.sequence = 0
        self.plan_contexts = {}

    def _make_manifest(self):
        skeleton = self._motion_rep.skeleton
        names = skeleton.bone_order_names
        parents = skeleton.joint_parents.cpu().tolist()
        positions = skeleton.neutral_joints.cpu().tolist()
        if skeleton.name != "g1skel34" or len(names) != 34:
            raise ValueError("No verified retarget mapping for this checkpoint skeleton")
        for model in (self._inferencer._root_model, self._inferencer._pose_model, self._inferencer._vqvae_pose_model):
            other = model.motion_rep.skeleton
            if names != other.bone_order_names or not torch.equal(skeleton.joint_parents, other.joint_parents.to(skeleton.joint_parents.device)):
                raise ValueError("Root/pose checkpoints disagree on source hierarchy")
            if not torch.allclose(skeleton.neutral_joints, other.neutral_joints.to(skeleton.neutral_joints.device)):
                raise ValueError("Root/pose checkpoints disagree on source reference positions")
        joints = [{"name": n, "parent_index": p, "neutral_position": v} for n, p, v in zip(names, parents, positions)]
        identity = {"coordinates": COORDINATES, "joints": joints}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        chains = [{"name": "Spine", "start": "waist_yaw_skel", "end": "waist_pitch_skel"}]
        for side in ("left", "right"):
            chains.extend([
                {"name": side.title() + "Arm", "start": side + "_shoulder_yaw_skel", "end": side + "_wrist_roll_skel"},
                {"name": side.title() + "Hand", "start": side + "_hand_roll_skel", "end": side + "_hand_roll_skel"},
                {"name": side.title() + "Leg", "start": side + "_hip_yaw_skel", "end": side + "_ankle_pitch_skel"},
                {"name": side.title() + "Foot", "start": side + "_ankle_roll_skel", "end": side + "_ankle_roll_skel"},
                {"name": side.title() + "Toe", "start": side + "_toe_base", "end": side + "_toe_base"},
            ])
        return {"schema_version": 2, "name": skeleton.name, "source_id": "motionbricks:" + skeleton.name + ":" + digest,
                "coordinates": COORDINATES, "root_bone": names[0], "motionbricks_checkpoint_verified": True,
                "provenance": {"repository": "https://github.com/NVlabs/GR00T-WholeBodyControl", "commit": commit},
                "checkpoints": self.checkpoints, "fps": self._fps, "joints": joints,
                "ground_anchor_bones": [names[i] for i in skeleton.foot_joint_idx], "retarget_chains": chains,
                "supported_modes": list(self._clip_holder.CLIPS), "retarget_profile_version": 3,
                "locomotion_capabilities": {
                    "supported": ["idle", "start", "stop", "slow_walk", "walk", "fast_walk", "turn_in_place", "strafe_left", "strafe_right", "backward_walk", "crouch_idle", "crouch_walk"],
                    "unsupported": ["run", "sprint", "jump", "fall", "land", "stairs", "slopes"],
                    "mode_selection": "backend using official G1 clips and directional remapping",
                    "crouch_mode": "walk_stealth",
                },
                "limitations": ["Published G1 weights, NOT the unreleased human27 model", "No source head, neck, fingers or clavicle",
                                "Navigation-only, one loopback client; no collision or foot IK",
                                "One outstanding plan request; supports replanning at a cached playback frame"]}

    @torch.inference_mode()
    def plan(self, mode="walk", velocity_ue_mps=(0.7, 0.0, 0.0), facing_ue=(1.0, 0.0, 0.0),
             context_sequence=None, context_frame=None, locomotion=None):
        if mode not in self._clip_holder.CLIPS:
            raise ValueError("Unknown motion mode: " + mode)
        velocity = torch.as_tensor(velocity_ue_mps, dtype=torch.float32, device=self._device)
        facing = torch.as_tensor(facing_ue, dtype=torch.float32, device=self._device)
        if velocity.shape != (3,) or facing.shape != (3,) or not torch.isfinite(velocity).all() or not torch.isfinite(facing).all():
            raise ValueError("Expected finite UE XYZ directions")
        if abs(velocity[2]) > 1.e-5 or abs(facing[2]) > 1.e-5 or facing[:2].norm() < 1.e-5 or velocity.norm() > 3.0:
            raise ValueError("This navigation adapter requires planar velocity <= 3 m/s and a nonzero planar facing")
        context = self.context
        if context_sequence is not None or context_frame is not None:
            if type(context_sequence) is not int or type(context_frame) is not int or context_sequence not in self.plan_contexts:
                raise ValueError("Unknown playback context sequence")
            history = self.plan_contexts[context_sequence]
            if context_frame < 0 or context_frame + 3 >= history.shape[1]:
                raise ValueError("Playback context frame out of range")
            context = history[:, context_frame:context_frame + 4]
        speed = velocity.norm()
        direction = velocity / speed.clamp_min(1.e-5)
        facing = facing / facing.norm()
        # Official navigation planner takes Z-up X-forward Y-left directions;
        # UE is Z-up X-forward Y-right. Its internal conversion to motion space is retained.
        movement = direction * direction.new_tensor([1.0, -1.0, 0.0])
        facing = facing * facing.new_tensor([1.0, -1.0, 0.0])
        stance = "standing"
        if locomotion is not None:
            if not isinstance(locomotion, dict) or set(locomotion) != {"gait", "stance"}:
                raise ValueError("Expected locomotion gait and stance")
            gait, stance = locomotion["gait"], locomotion["stance"]
            if gait not in ("slow_walk", "walk", "fast_walk") or stance not in ("standing", "crouched"):
                raise ValueError("Unsupported gait or stance; no substitute animation is available")
            speed = speed.clamp_max({"slow_walk": .35, "walk": .7, "fast_walk": 1.0}[gait])
            if stance == "crouched":
                mode = "walk_stealth"
            elif speed < 1.e-5:
                mode = "idle"
            elif gait == "slow_walk":
                mode = "slow_walk"
            # Use the official directional clip remapper instead of a UE blendspace.
            mode_tensor = torch.tensor([list(self._clip_holder.CLIPS).index(mode)], device=self._device)
            remapped = self._clip_holder.blendspace_modes_remap_from_velocity(
                mode_tensor, movement[None, [1, 0]], torch.atan2(facing[1], facing[0]).reshape(1))
            mode = list(self._clip_holder.CLIPS)[int(remapped.item())]
            if mode in ("walk_left", "walk_right"):
                speed = speed.clamp_max(self._clip_holder.CLIPS[mode]["avg_root_vel"] / 2.0)
            elif stance == "crouched":
                speed = speed.clamp_max(self._clip_holder.CLIPS[mode]["avg_root_vel"] / 2.0)
        else:
            mode = "idle" if speed < 1.e-5 else mode
        inputs = {"mode": torch.tensor([list(self._clip_holder.CLIPS).index(mode)], device=self._device),
                  "movement_direction": movement[None], "facing_direction": facing[None],
                  "target_vel": speed.reshape(1), "random_seed": torch.tensor([1234 + self.sequence], device=self._device),
                  "context_motion_features": context}
        mask = self._clip_holder.CLIPS[mode]["allowed_pred_num_tokens"]
        allowed = torch.tensor(mask, device=self._device, dtype=torch.bool)
        # Use a longer permitted neural horizon; short idle chunks cannot cover a busy render GPU.
        min_tokens = self._inferencer._root_model.backbone_net._args["min_tokens"]
        new_frames = (torch.arange(len(mask), device=self._device) + min_tokens - 1) * self.NUM_FRAMES_PER_TOKEN
        long_horizon = allowed & (new_frames >= 36)
        if not long_horizon.any():
            long_horizon[torch.nonzero(allowed)[-1, 0]] = True
        inputs["allowed_pred_num_tokens"] = long_horizon[None]
        started = time.perf_counter()
        boundary = self._motion_rep.inverse(context[:, -1:], is_normalized=False, joint_positions_from="global_rot_data", return_quat=False)
        boundary_positions = boundary["posed_joints"][0]
        boundary_rotations = torch.nn.functional.normalize(matrix_to_quaternion(boundary["global_joint_rots"][0]), dim=-1)[..., [1, 2, 3, 0]]
        inputs["context_global_joint_positions"], inputs["context_global_joint_rotations"] = self._process_input_to_joint_transforms(inputs)
        context_speed = ((inputs["context_global_joint_positions"][0, -1, 0, [0, 2]] -
                          inputs["context_global_joint_positions"][0, -2, 0, [0, 2]]) * self._fps).norm().item()
        current_heading = torch.atan2(inputs["context_global_joint_rotations"][0, -1, 0, 0, 2],
                                     inputs["context_global_joint_rotations"][0, -1, 0, 2, 2])
        heading_error = abs(((torch.atan2(facing[1], facing[0]) - current_heading + torch.pi) % (2 * torch.pi) - torch.pi).item())
        if stance == "crouched" and speed < 1.e-5:
            # The official spring planner interprets zero target speed as the clip default.
            # Its explicit target API holds the root while the crouch template conditions the pose.
            root = boundary_positions[0, 0]
            inputs["specific_target_positions"] = root[[2, 0, 1]].reshape(1, 1, 3).expand(1, 4, 3).clone()
            inputs["specific_target_headings"] = torch.atan2(facing[1], facing[0]).reshape(1, 1).expand(1, 4).clone()
            inputs["has_specific_target"] = torch.ones(1, 1, device=self._device)
        (inputs["target_root_position"], inputs["target_root_positions"], inputs["target_root_headings"],
         inputs["target_root_heading"], inputs["start_root_positions"], inputs["start_root_headings"]) = self._generate_spring_model_position_and_heading(inputs)
        self._override_target_transforms(inputs)
        (inputs["target_global_joint_positions"], inputs["target_global_joint_rotations"],
         inputs["target_global_root_positions"]) = self._generate_target_joint_transforms(inputs)
        # Keep the official checkpoint's fast network precision, but use full precision for FK.
        torch.set_float32_matmul_precision("high")
        try:
            features, frame_count = self._predict_inbetween_motion_features(inputs)
        finally:
            torch.set_float32_matmul_precision("highest")
        count = int(frame_count.item())
        if count <= 4 or count > features.shape[1] or not torch.isfinite(features[:, :count]).all():
            raise RuntimeError("Invalid generated feature sequence")
        generated = features[:, :count]
        self.context = generated[:, -4:].clone()
        # FK from predicted rotations preserves bone lengths; do not use redundant predicted positions for rendering.
        transforms = self._motion_rep.inverse(generated, is_normalized=False, joint_positions_from="global_rot_data", return_quat=False)
        positions = transforms["posed_joints"][0]
        rotations = torch.nn.functional.normalize(matrix_to_quaternion(transforms["global_joint_rots"][0]), dim=-1)[..., [1, 2, 3, 0]]
        if not torch.isfinite(positions).all() or not torch.isfinite(rotations).all():
            raise RuntimeError("Invalid generated transforms")
        self.sequence += 1
        self.plan_contexts[self.sequence] = torch.cat((context, generated[:, 4:]), dim=1).clone()
        while len(self.plan_contexts) > 4:
            oldest = next(sequence for sequence in self.plan_contexts if sequence != context_sequence)
            del self.plan_contexts[oldest]
        # Include one previous-context boundary frame for continuous interpolated playback.
        positions = torch.cat((boundary_positions, positions[4:]), dim=0)
        rotations = torch.cat((boundary_rotations, rotations[4:]), dim=0)
        frame_data = torch.cat((positions, rotations), dim=-1).cpu().tolist()
        forward_dot = torch.dot(direction * direction.new_tensor([1.0, -1.0, 0.0]), facing).item()
        if stance == "crouched":
            state = "crouch_idle" if speed < 1.e-5 else "crouch_walk"
        elif speed < 1.e-5:
            state = "turn_in_place" if heading_error > 0.15 else "stop" if context_speed > 0.1 else "idle"
        elif mode in ("walk_left", "walk_right"):
            state = "strafe_left" if mode == "walk_left" else "strafe_right"
        elif forward_dot < -0.707:
            state = "backward_walk"
        else:
            state = "start" if context_speed < 0.1 else "slow_walk" if mode == "slow_walk" else "fast_walk" if locomotion and locomotion["gait"] == "fast_walk" else "walk"
        return {"schema": SCHEMA, "source_id": self.manifest["source_id"], "model_id": self.model_id,
                "coordinates": COORDINATES, "sequence": self.sequence, "fps": self._fps,
                "mode": mode, "neural_inference": True, "physics": False,
                "locomotion_state": state, "resolved_speed_mps": speed.item(),
                "inference_ms": (time.perf_counter() - started) * 1000.0, "frames": frame_data}
