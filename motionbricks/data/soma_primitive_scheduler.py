"""Ablate native primitive consumption without changing any generated frame.

MotionBricks Algorithm 1 replans on changed constraints or a low buffer. The
paper's 3--9 frame recommendation uses 30 fps, unlike this SOMA pilot's 60 fps.
Longer consumption is an experiment, not a reproduction of that default.
"""

import torch


class NativePrimitiveScheduler:
    def __init__(self, backend, replan_frames):
        if (replan_frames < backend.commit_frames
                or replan_frames % backend.commit_frames
                or replan_frames > backend.horizon_frames - 4):
            raise ValueError("Replan interval must fit whole published blocks in a native primitive")
        self.backend = backend
        self.replan_frames = replan_frames
        self._motion = None
        self._command = None
        self._metadata = None
        self._cursor = 0
        self.model_calls = 0

    @torch.inference_mode()
    def step(self, velocity_xz, heading_rad):
        history = self.backend.history
        velocity = torch.as_tensor(velocity_xz, dtype=history.dtype, device=history.device).reshape(-1, 2)
        heading = torch.as_tensor(heading_rad, dtype=history.dtype, device=history.device).reshape(-1, 1)
        if len(velocity) != len(history) or len(heading) != len(history):
            raise ValueError("Commands must match the history batch")
        command = torch.cat((velocity, heading), -1)
        if not torch.isfinite(command).all():
            raise ValueError("Commands must be finite")
        reason = None
        if self._motion is None:
            reason = "initial"
        elif not torch.equal(command, self._command):
            reason = "command_changed"
        elif self._motion.shape[1] - self._cursor < self.backend.commit_frames:
            reason = "buffer_low"
        elif self._cursor >= self.replan_frames:
            reason = "interval_limit"
        elapsed = 0.0
        if reason is not None:
            result = self.backend.predict_plan(velocity, heading.flatten())
            self._motion = result.pop("planned_motion").detach().clone()
            if self._motion.shape[1] < self.backend.commit_frames:
                raise ValueError("Prediction is shorter than a published block")
            self._metadata = result
            elapsed = result["generation_seconds"]
            self._command = command.clone()
            self._cursor = 0
            self.model_calls += 1
        offset = self._cursor
        committed = self._motion[:, offset:offset + self.backend.commit_frames].clone()
        self._cursor += self.backend.commit_frames
        self.backend.history = committed[:, -4:].clone()
        start = self.backend.published_frames
        self.backend.published_frames += self.backend.commit_frames
        return {**self._metadata, "motion": committed, "start_frame": start,
                "timestamps_s": torch.arange(start, start + self.backend.commit_frames,
                                               device=committed.device) / self.backend.representation.fps,
                "generation_seconds": elapsed, "replanned": reason is not None,
                "replan_reason": reason, "primitive_frame_offset": offset}
