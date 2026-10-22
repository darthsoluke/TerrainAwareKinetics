"""SOMA kinematic adapter for the shipped MotionBricks smart-locomotion methods.

The upstream spring, proxy-pose placement, and inference packing are invoked
directly. The adapter replaces only the G1/MuJoCo clip loading and playback.
Reference clips provide input keyframes; played outputs remain neural predictions.
"""

import math

import torch

from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent


class NativeReferenceClips(torch.nn.Module):
    def __init__(self, motion_rep, clips):
        super().__init__()
        if list(clips)[0:1] != ["idle"] or len(clips) < 2:
            raise ValueError("Clip bank must begin with idle and include a moving style")
        self.CLIPS = {}
        prepared = {}
        for name, raw in clips.items():
            if (raw.ndim != 2 or raw.shape[1] != len(motion_rep.indices["all"])
                    or len(raw) <= 4 or not torch.isfinite(raw).all()):
                raise ValueError(f"Invalid reference clip: {name}")
            raw = motion_rep.change_first_heading(raw[None], 0.0, False, False)
            transforms = motion_rep.inverse(raw, False, return_quat=False, return_all=True)
            positions, rotations = transforms["posed_joints"][0], transforms["global_joint_rots"][0]
            root = positions[:, motion_rep.skeleton.root_idx]
            projected = root * root.new_tensor([1, 0, 1])
            heading = torch.atan2(rotations[:, 0, 0, 2], rotations[:, 0, 2, 2])
            speed = float(torch.diff(root[:, [0, 2]], dim=0).norm(dim=-1).mean() * motion_rep.fps)
            self.CLIPS[name] = {"avg_root_vel": speed, "frames": len(root)}
            prepared[name] = (projected, positions - projected[:, None], rotations, heading)
        count, frames = len(prepared), max(len(value[0]) for value in prepared.values())
        first = next(iter(prepared.values()))
        for field, index in (("global_root_positions", 0), ("global_joint_positions", 1),
                             ("global_joint_rotations", 2), ("global_headings", 3)):
            shape = (count, frames, *first[index].shape[1:])
            buffer = first[index].new_zeros(shape)
            for row, value in enumerate(prepared.values()):
                buffer[row, :len(value[index])] = value[index]
            self.register_buffer(field, buffer)
        self.register_buffer("num_frames_per_clip", torch.tensor(
            [len(value[0]) for value in prepared.values()], device=first[0].device, dtype=torch.long))


class _InferenceInputs:
    def __init__(self, infer, keyframes_per_end, fixed_tokens, context_keyframes, *, pose_iterations=1):
        if type(pose_iterations) is not int or not 1 <= pose_iterations <= 16:
            raise ValueError('Native pose iterations must be an integer in [1, 16]')
        self.infer = infer
        self._root_model = infer._root_model
        self.local_motion_rep = infer.local_motion_rep
        self.keyframes_per_end = keyframes_per_end
        self.context_keyframes = context_keyframes
        self.fixed_tokens = fixed_tokens
        self.pose_iterations = pose_iterations
        self.last = None

    def predict(self, global_root_values, has_global_root_values, local_root_values,
                has_local_root_values, local_poses, has_local_poses, num_tokens, **kwargs):
        kwargs['config'] = dict(kwargs['config'], num_inference_step=self.pose_iterations)
        masks = [has_global_root_values.clone(), has_local_root_values.clone(), has_local_poses.clone()]
        if self.context_keyframes == 1:
            # Last context velocity needs a future frame: use the last measurable
            # interval for local root, and the newest pose/position for the others.
            for mask, index in zip(masks, (3, 2, 3)):
                mask[:, :4] = False
                mask[:, index] = True
        if self.keyframes_per_end == 1:
            for mask in masks:
                mask[:, 4:7] = False
        if self.fixed_tokens is not None:
            minimum, maximum = self.infer._args["min_tokens"], self.infer._args["max_tokens"]
            allowed = global_root_values.new_zeros((1, maximum - minimum + 1))
            allowed[0, self.fixed_tokens - minimum] = 1
            kwargs["allowed_pred_num_tokens"] = allowed
        self.last = {
            "global_root_values": global_root_values.detach().clone(),
            "has_global_root_values": masks[0].detach().clone(),
            "local_root_values": local_root_values.detach().clone(),
            "has_local_root_values": masks[1].detach().clone(),
            "local_poses": local_poses.detach().clone(),
            "has_local_poses": masks[2].detach().clone(),
            "num_tokens": num_tokens.detach().clone(),
            "allowed_pred_num_tokens": kwargs["allowed_pred_num_tokens"].detach().clone(),
            "config": dict(kwargs["config"]),
        }
        return self.infer.predict(global_root_values, masks[0], local_root_values, masks[1],
                                  local_poses, masks[2], num_tokens, **kwargs)


class NativeSmartLocomotion(torch.nn.Module):
    NUM_FRAMES_PER_TOKEN = 4
    _generate_spring_model_position_and_heading = full_navigation_agent._generate_spring_model_position_and_heading
    _generate_target_joint_transforms = full_navigation_agent._generate_target_joint_transforms
    _predict_inbetween_motion_features = full_navigation_agent._predict_inbetween_motion_features

    def __init__(self, infer, clip_bank, *, keyframes_per_end=4, context_keyframes=None, fixed_tokens=None,
                 source_root_realignment=True, skip_ending_target_cond=False, min_pred_tokens=None):
        super().__init__()
        if keyframes_per_end not in (1, 4):
            raise ValueError("Use one or four keyframes per end")
        context_keyframes = keyframes_per_end if context_keyframes is None else context_keyframes
        if context_keyframes not in (1, 4):
            raise ValueError("Use one or four context keyframes")
        minimum, maximum = infer._args["min_tokens"], infer._args["max_tokens"]
        if fixed_tokens is not None and not minimum <= fixed_tokens <= maximum:
            raise ValueError("Fixed duration is outside the native token range")
        if min_pred_tokens is not None and not minimum <= min_pred_tokens <= min(maximum, 11):
            raise ValueError("Minimum live duration is outside the native token range")
        if infer.motion_rep.skeleton.root_idx != 0 or infer.motion_rep.fps != 30:
            raise ValueError("The shipped primitive requires root index 0 and native 30 Hz")
        self.base_infer = infer
        self._inferencer = _InferenceInputs(infer, keyframes_per_end, fixed_tokens, context_keyframes)
        self._clip_holder = clip_bank
        self._motion_rep = infer.motion_rep
        self._device = torch.device(infer.device)
        self._fps = infer.motion_rep.fps
        self._speed_scale_min = self._speed_scale_max = 1.0
        self._target_root_realignment = True
        self._source_root_realignment = source_root_realignment
        self.FORCE_CANONICALIZATION = False
        self.SKIP_ENDING_TARGET_COND = bool(skip_ending_target_cond)
        self.keyframes_per_end = keyframes_per_end
        self.context_keyframes = context_keyframes
        self.min_pred_tokens = minimum if min_pred_tokens is None else min_pred_tokens

    @property
    def last_constraints(self):
        return self._inferencer.last

    @torch.inference_mode()
    def predict(self, history, velocity_xz, heading, *, style, random_seed):
        if (history.ndim != 3 or history.shape != (1, 4, len(self._motion_rep.indices["all"]))
                or not torch.isfinite(history).all()):
            raise ValueError("Expected four finite played history frames")
        velocity = history.new_tensor(velocity_xz)
        if velocity.shape != (2,) or not torch.isfinite(velocity).all() or not math.isfinite(heading):
            raise ValueError("Command must contain finite XZ velocity and heading")
        if style not in self._clip_holder.CLIPS or isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("Invalid reference style or sampling seed")
        transforms = self._motion_rep.inverse(history, False, return_quat=False, return_all=True)
        speed = velocity.norm()
        movement = torch.zeros_like(velocity) if speed == 0 else velocity / speed
        minimum, maximum = self.base_infer._args["min_tokens"], self.base_infer._args["max_tokens"]
        # The shipped G1 helper multiplies target_vel by two. Cancel that unit
        # adapter so Eq. 6's input is the requested SOMA velocity in m/s.
        inputs = {
            "context_global_joint_positions": transforms["posed_joints"].clone(),
            "context_global_joint_rotations": transforms["global_joint_rots"].clone(),
            "mode": torch.tensor([list(self._clip_holder.CLIPS).index(style)], device=history.device),
            "movement_direction": movement[[1, 0]][None],
            "facing_direction": history.new_tensor([[math.cos(heading), math.sin(heading)]]),
            "target_vel": speed.reshape(1, 1) / 2,
            "random_seed": torch.tensor([random_seed], device=history.device),
            "allowed_pred_num_tokens": history.new_tensor([[int(self.min_pred_tokens <= value <= 11)
                                                           for value in range(minimum, maximum + 1)]]),
        }
        (inputs["target_root_position"], inputs["target_root_positions"],
         inputs["target_root_headings"], inputs["target_root_heading"],
         inputs["start_root_positions"], inputs["start_root_headings"]) = self._generate_spring_model_position_and_heading(inputs)
        (inputs["target_global_joint_positions"], inputs["target_global_joint_rotations"],
         inputs["target_global_root_positions"]) = self._generate_target_joint_transforms(inputs)
        prediction, frame_count = self._predict_inbetween_motion_features(inputs)
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite native smart-locomotion prediction")
        return prediction, frame_count // self.NUM_FRAMES_PER_TOKEN
