"""Eight-frame committed playback of unmodified MB smart-locomotion predictions."""

import time
from types import SimpleNamespace

import numpy as np
import torch

from motionbricks.data.native_mb_locomotion import NativeReferenceClips, NativeSmartLocomotion


def native_locomotion_contract(bundle):
    """Bind sparse navigation inputs to the actual upstream checkpoint curriculum."""
    from motionbricks.motion_backbone.models.root_model import MotionModel as RootModel
    from motionbricks.motion_backbone.models.pose_model import MotionModel as PoseModel

    coverage = {}
    for stage, model_type in (("root", RootModel), ("pose", PoseModel)):
        args = bundle[stage + "_config"]["model"]["args"]
        step = bundle["metadata"][stage + "_step"]
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("Live inference requires a completed checkpoint update count")
        probe = SimpleNamespace(trainer=SimpleNamespace(global_step=step), args=args)
        coverage[stage] = {}
        for end in ("start", "end"):
            # Call the real training scheduler; do not infer readiness from configured maxima.
            probabilities = model_type._construct_keyframe_prob(
                probe, args["max_num_" + end + "_keyframes"], args["no_" + end + "_keyframe_prob"])
            coverage[stage][end] = [int(value) for value in np.flatnonzero(probabilities) if value > 0]
            if 1 not in coverage[stage][end]:
                raise ValueError("Checkpoint curriculum does not cover single-keyframe navigation")
    return {"schema": "motionbricks.soma_locomotion_conditions.v1",
            "context_keyframes": 1, "target_keyframes": 1,
            "skip_ending_target_cond": False, "trained_keyframe_counts": coverage,
            "basis": "MB paper sparse navigation; actual Root/Pose training curriculum",
            "output_root": "native decoder refinement, no replacement"}


class NativeSomaLiveBackend:
    PUBLISH_FRAMES = 8
    DROP_END_FRAMES = 4

    def __init__(self, inference, bundle, *, reuse_motion_buffer=False):
        self.history = bundle["initial_history"].to(inference.device).clone()
        clips = {key: value.to(inference.device) for key, value in bundle["reference_clips"].items()}
        self.condition_contract = native_locomotion_contract(bundle)
        self.reuse_motion_buffer = reuse_motion_buffer
        self._future_buffer = None
        self._last_published_history = None
        self._last_command = None
        self.last_generation = None
        if reuse_motion_buffer:
            self.condition_contract.update(buffer_policy="command change or buffer low; paper Algorithm 1",
                                           drop_end_frames=self.DROP_END_FRAMES, min_live_tokens=4)
        self.controller = NativeSmartLocomotion(
            inference, NativeReferenceClips(inference.global_motion_rep, clips),
            keyframes_per_end=self.condition_contract["target_keyframes"],
            context_keyframes=self.condition_contract["context_keyframes"],
            skip_ending_target_cond=self.condition_contract["skip_ending_target_cond"],
            min_pred_tokens=4 if reuse_motion_buffer else None)
        self.last_style = None

    def _buffer_context_offset(self):
        previous = self._last_published_history
        if previous is None or previous.shape != self.history.shape:
            return None
        # A constant planar collision offset changes coordinates, not the cached motion.
        offset = self.history[0, -1, :3] - previous[0, -1, :3]
        offset = offset.clone()
        offset[1] = 0
        if (not torch.allclose(self.history[..., 3:], previous[..., 3:], atol=1e-5, rtol=0)
                or not torch.allclose(self.history[..., :3], previous[..., :3] + offset, atol=1e-4, rtol=0)):
            return None
        return offset

    @torch.inference_mode()
    def predict_plan(self, velocity_xz, heading):
        velocity = np.asarray(velocity_xz, dtype=np.float64)
        if (velocity.shape != (1, 2) or len(heading) != 1
                or not np.isfinite(velocity).all() or not np.isfinite(heading).all()):
            raise ValueError("Expected one finite planar command")
        speed = float(np.linalg.norm(velocity))
        style = "idle" if speed < 1e-6 else "run" if speed >= 1.8 else "walk"
        command = (tuple(velocity[0]), float(heading[0]), style)
        buffered = getattr(self, "reuse_motion_buffer", False)
        reason = "initial"
        offset = None
        if buffered and self._last_command is not None:
            offset = self._buffer_context_offset()
            if command != self._last_command:
                reason = "command_changed"
            elif offset is None:
                reason = "context_changed"
            elif self._future_buffer.shape[1] < self.PUBLISH_FRAMES:
                reason = "buffer_low"
            else:
                future = self._future_buffer[:, :self.PUBLISH_FRAMES].clone()
                future[..., :3] += offset
                self._future_buffer = self._future_buffer[:, self.PUBLISH_FRAMES:].clone()
                self._future_buffer[..., :3] += offset
                self._last_published_history = future[:, -4:].clone()
                self.last_style = style
                self.last_generation = {"neural_generation": False, "reason": "buffer_sample",
                                        "remaining_frames": self._future_buffer.shape[1]}
                return {"planned_motion": future, "generation_seconds": 0.0}
        if self.history.is_cuda:
            torch.cuda.synchronize(self.history.device)
        started = time.perf_counter()
        prediction, count = self.controller.predict(self.history, velocity[0].tolist(), float(heading[0]),
                                                     style=style, random_seed=2333)
        frames = int(count.item()) * 4 if isinstance(count, torch.Tensor) else int(count) * 4
        if frames < 12 or prediction.shape[1] < frames:
            raise ValueError("Native prediction cannot provide an eight-frame future after four context frames")
        # Input history owns the preceding clock. Do not replay decoded context.
        if buffered:
            if frames < 4 + self.PUBLISH_FRAMES + self.DROP_END_FRAMES:
                raise ValueError("Native prediction cannot satisfy context/publish/drop frame contract")
            self._future_buffer = prediction[:, 4:frames - self.DROP_END_FRAMES].clone()
            future = self._future_buffer[:, :self.PUBLISH_FRAMES].clone()
            self._future_buffer = self._future_buffer[:, self.PUBLISH_FRAMES:].clone()
            self._last_published_history = future[:, -4:].clone()
            self._last_command = command
        else:
            future = prediction[:, 4:12].clone()
        if self.history.is_cuda:
            torch.cuda.synchronize(self.history.device)
        self.last_style = style
        self.last_generation = {"neural_generation": True, "reason": reason,
                                "remaining_frames": self._future_buffer.shape[1] if buffered else 0,
                                "predicted_frames": frames}
        return {"planned_motion": future, "generation_seconds": time.perf_counter() - started}
