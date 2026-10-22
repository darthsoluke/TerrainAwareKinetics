"""Bounded online authored stair interaction through MB's original Smart Object.

Commands select/offset INPUT keyframes. Every published motion frame is an
unchanged neural prediction. No playback clips, root projection, IK or blending.
"""

import time

import numpy as np
import torch

from motionbricks.data.native_mb_live import native_locomotion_contract
from motionbricks.data.native_mb_smart_object import NativeSmartObject


class NativeStairLiveBackend:
    PUBLISH_FRAMES = 8

    def __init__(self, inference, bundle, course, *, primitive=None, exit_locomotion=None,
                 use_authored_frames=False):
        self.course_id = course["id"]
        self.history = torch.tensor(course["initial_history"], device=inference.device, dtype=torch.float32)[None]
        self.seed = self.history.clone()
        self.goals = [torch.tensor(g["features"], device=inference.device, dtype=torch.float32)[None] for g in course["goals"]]
        if (not self.goals or self.history.shape != (1, 4, 366)
                or any(g.shape != self.history.shape or not torch.isfinite(g).all() for g in self.goals)
                or not torch.isfinite(self.history).all()):
            raise ValueError("Invalid sparse stair conditions")
        self.forward = np.asarray(course["forward_xz"], dtype=np.float64)
        if self.forward.shape != (2,) or not np.isfinite(self.forward).all() or abs(np.linalg.norm(self.forward) - 1) > 1e-5:
            raise ValueError("Invalid course direction")
        self.right = np.array([-self.forward[1], self.forward[0]])
        self.condition_contract = native_locomotion_contract(bundle) if primitive is None else {}
        self.primitive = primitive or NativeSmartObject(inference)
        self.exit_locomotion = exit_locomotion
        self.frame_origins = course.get('input_frame_origins_y_m') if use_authored_frames else None
        if use_authored_frames and (self.frame_origins is None
                or len(self.frame_origins) != len(self.goals) + 1 or not np.isfinite(self.frame_origins).all()):
            raise ValueError('Authored input frames require one explicit origin per seed/landing')
        self.cursor = -1
        self.active_goal_index = -1
        self.primitive_state = "hold"
        self.lateral_offset = 0.
        self._command = None
        self._future = self.history[:, :0].clone()
        self._last_history = None
        self._pending_cursor = None
        self._hold_goal = None
        self._landing_goal = self.seed.clone()
        self._landing_offset = 0.
        self.last_generation = {}

    @torch.inference_mode()
    def predict_plan(self, velocity_xz, heading):
        velocity = np.asarray(velocity_xz, dtype=np.float64)
        if velocity.shape != (1, 2) or not np.isfinite(velocity).all() or not np.isfinite(heading).all():
            raise ValueError("Invalid stair command")
        along, lateral = float(velocity[0] @ self.forward), float(velocity[0] @ self.right)
        direction = 1 if along > .1 else -1 if along < -.1 else 0
        strafe = 1 if lateral > .1 else -1 if lateral < -.1 else 0
        command = (direction, strafe)
        context_changed = self._last_history is not None and not torch.allclose(self.history, self._last_history, atol=1e-4, rtol=0)
        changed = command != self._command or context_changed
        if context_changed:
            raise ValueError("Authored stair context diverged from its immutable raw prediction")
        # MB Smart Object tau=0: finish the current landing phase before a new
        # intent takes effect. Replanning a half-airborne stop is not a hold skill.
        if (changed or strafe) and self._pending_cursor is None:
            self._future = self.history[:, :0].clone()
            self._hold_goal = None
        elif changed:
            self._hold_goal = None
        self._command = command
        if strafe:
            self.lateral_offset = float(np.clip(self.lateral_offset + lateral * self.PUBLISH_FRAMES / 30., -.25, .25))
        pieces, produced, plans = [], 0, []
        started = time.perf_counter()
        while produced < self.PUBLISH_FRAMES:
            if not self._future.shape[1]:
                context = torch.cat([self.history] + pieces, dim=1)[:, -4:].clone()
                if self.exit_locomotion is not None and self.cursor == len(self.goals) - 1 and direction >= 0:
                    # This bounded course ends here. Idle is a locomotion intent,
                    # not an indefinitely repeated moving interaction endpoint.
                    facing = float(np.arctan2(self.forward[0], self.forward[1]))
                    result = self.exit_locomotion.predict(context, [[0., 0.]], [facing],
                        target_offset_xz=self.right * self.lateral_offset)
                    self._future = result['planned_motion'].clone()
                    if self._future.shape != (1, self.PUBLISH_FRAMES, context.shape[-1]):
                        raise ValueError('Exit locomotion must publish eight native frames')
                    self.primitive_state = 'exit_idle'
                    plans.append({'frames': 4 + self.PUBLISH_FRAMES, 'goal_index': self.cursor,
                                  'state': self.primitive_state,
                                  'neural_generation': result['neural_generation'],
                                  'platform_y_m': self.exit_locomotion.platform_y_m})
                    continue
                target_index = min(len(self.goals) - 1, self.cursor + 1) if direction > 0 else max(-1, self.cursor - 1)
                moving = direction != 0 and target_index != self.cursor
                if moving:
                    goal = (self.seed if target_index < 0 else self.goals[target_index]).clone()
                    # Backtracking requests a stationary endpoint, not time-reversed playback.
                    if direction < 0:
                        goal = goal[:, -1:].repeat(1, 4, 1)
                    offset = goal.new_tensor(self.right * self.lateral_offset)
                    goal[..., 0] += offset[0]
                    goal[..., 2] += offset[1]
                    self._pending_cursor = target_index
                    self._landing_goal = goal.clone()
                    self._landing_offset = self.lateral_offset
                    self.active_goal_index = target_index
                    self.primitive_state = "advance" if direction > 0 else "backtrack"
                else:
                    if self._hold_goal is None:
                        # Repeated INPUT poses request zero terminal velocity. Outputs are never repeated.
                        self._hold_goal = self._landing_goal[:, -1:].repeat(1, 4, 1)
                        shift = self._hold_goal.new_tensor(self.right * (self.lateral_offset - self._landing_offset))
                        self._hold_goal[..., 0] += shift[0]
                        self._hold_goal[..., 2] += shift[1]
                    goal = self._hold_goal
                    self.primitive_state = "strafe" if strafe else "complete" if direction else "hold"
                    self._pending_cursor = None
                origin = self.frame_origins[self.cursor + 1] if self.frame_origins is not None else 0.
                frame_kwargs = {'frame_origin_y_m': origin} if self.frame_origins is not None else {}
                prediction, frames = self.primitive.predict(context, goal, **frame_kwargs)
                if frames < 8 or frames > prediction.shape[1]:
                    raise ValueError("Smart Object must generate a finite future beyond its four context frames")
                self._future = prediction[:, 4:frames].clone()
                plans.append({"frames": frames, "goal_index": self.active_goal_index,
                              'input_frame_origin_y_m': origin,
                              "state": self.primitive_state, "goal_root_m": goal[0, -1, :3].tolist()})
            count = min(self.PUBLISH_FRAMES - produced, self._future.shape[1])
            pieces.append(self._future[:, :count].clone())
            self._future = self._future[:, count:].clone()
            produced += count
            if not self._future.shape[1] and self._pending_cursor is not None:
                self.cursor = self._pending_cursor
                self._pending_cursor = None
                self._hold_goal = None
        future = torch.cat(pieces, dim=1)
        if not torch.isfinite(future).all():
            raise ValueError("Nonfinite neural stair output")
        self._last_history = future[:, -4:].clone()
        self.last_generation = {"neural_generation": any(p.get('neural_generation', True) for p in plans), "plans": plans,
                                "remaining_frames": self._future.shape[1], "course": self.course_id,
                                "cursor": self.cursor, "state": self.primitive_state,
                                "requested_direction": direction, "phase_switch": "native_smart_object_tau_0",
                                "lateral_input_offset_m": self.lateral_offset}
        return {"planned_motion": future, "generation_seconds": time.perf_counter() - started if plans else 0.}
