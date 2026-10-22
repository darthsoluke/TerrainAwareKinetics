"""Causal planar-intent governor with immutable, simulation-frame state."""

from dataclasses import dataclass, replace
import math
import time

import numpy as np


DEFAULTS = dict(enabled=False, strength=.5, minimum_speed=.6, recovery_acceleration=1.)
LIMITS = dict(strength=[0., 1.], minimum_speed=[.3, 1.5], recovery_acceleration=[.25, 3.])


def validate_governor(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError('Incomplete terrain speed controls')
    if type(value['enabled']) is not bool:
        raise ValueError('Terrain speed enable must be boolean')
    result = dict(value)
    for key, (low, high) in LIMITS.items():
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or not low <= item <= high:
            raise ValueError(f'Invalid terrain speed setting: {key}')
        result[key] = float(item)
    return result


@dataclass(frozen=True)
class GovernorState:
    frame: int = 0
    target_speed: float | None = None
    risk_limited: bool = False
    hold_frames: int = 0
    hold_speed: float = 12.
    path: tuple = ()
    path_frame: int = 0

    def bind_plan(self, path, source_frame):
        return replace(self, path=tuple(tuple(float(v) for v in point) for point in path),
                       path_frame=source_frame)


class TerrainSpeedMap:
    def __init__(self, scene):
        self.height = np.asarray(scene['heightfield_y_m'])
        self.origin = np.asarray(scene['min_point_world_xyz_m'], dtype=np.float64)
        self.dx = float(scene['dx_m'])
        self.height.setflags(write=False)

    def heights(self, xz):
        indices = np.floor((xz-self.origin[[0, 2]])/self.dx+.5).astype(np.int64)
        x, z = indices[..., 0], indices[..., 1]
        valid = (x >= 0) & (z >= 0) & (x < self.height.shape[1]) & (z < self.height.shape[0])
        height = self.height[np.clip(z, 0, self.height.shape[0]-1), np.clip(x, 0, self.height.shape[1]-1)]
        valid &= np.isfinite(height)
        return np.where(valid, height, 0.), valid

    @staticmethod
    def _previous_curve(state, position, direction, stations):
        if len(state.path) < 2 or state.frame-state.path_frame > 28:
            return None
        path = np.asarray(state.path, dtype=np.float64)
        delta = np.diff(path, axis=0)
        length2 = np.sum(delta*delta, axis=1)
        projection = np.clip(np.sum((position-path[:-1])*delta, axis=1)/np.maximum(length2, 1e-12), 0., 1.)
        closest = path[:-1]+projection[:, None]*delta
        index = int(np.argmin(np.sum((closest-position)**2, axis=1)))
        tangent = delta[index]/max(float(np.linalg.norm(delta[index])), 1e-8)
        if np.linalg.norm(closest[index]-position) > .75 or float(tangent@direction) < .25:
            return None
        if index==len(delta)-1 and projection[index]>=1. and float((path[-1]-position)@direction)<=0.:
            return None
        tail = np.concatenate((position[None], path[index+1:]), axis=0)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(tail, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 1e-6]
        tail, arc = tail[keep], arc[keep]
        if len(tail) < 2:
            return None
        end_direction = (tail[-1]-tail[-2])/(arc[-1]-arc[-2])
        result = np.stack([np.interp(stations, arc, tail[:, axis]) for axis in (0, 1)], axis=-1)
        result += np.maximum(stations-arc[-1], 0.)[:, None]*end_direction
        return result

    def govern(self, root, requested, state, config, frames):
        started = time.perf_counter()
        root = np.asarray(root, dtype=np.float64)
        requested = np.asarray(requested, dtype=np.float64).reshape(2)
        desired = float(np.linalg.norm(requested))
        velocity = (root[-1, [0, 2]]-root[-2, [0, 2]])*30.
        actual = float(np.linalg.norm(velocity))
        dt = frames/30.
        info = dict(enabled=config['enabled'], requested_speed_mps=desired, actual_history_speed_mps=actual,
            source_frame=state.frame, publish_frames=frames, extra_root_passes=0)
        if not config['enabled'] or desired < 1e-8:
            new = replace(state, frame=state.frame+frames, target_speed=None if desired < 1e-8 else desired,
                          risk_limited=False, hold_frames=0, hold_speed=12.)
            info.update(target_speed_mps=desired, mode='off' if not config['enabled'] else 'stopped', risk=0.,
                        sample_count=0, risk_point_world_xyz_m=None, overhead_ms=(time.perf_counter()-started)*1000)
            return requested.tolist(), new, info
        strength = config['strength']
        minimum = min(desired, config['minimum_speed'])
        reaction = .4+.2*strength+dt
        decel, margin = 2., .25+.2*strength
        planning_speed = max(actual, desired)
        needed = planning_speed*reaction+max(planning_speed*planning_speed-minimum*minimum, 0.)/(2*decel)+margin
        extent = min(16., max(2., needed+1., desired*.7))
        stations = np.linspace(0., extent, 257)
        position = root[-1, [0, 2]]
        direction = requested/desired
        routes = [position[None]+stations[:, None]*direction]
        names = ['requested']
        if actual > .05:
            routes.append(position[None]+stations[:, None]*velocity/actual)
            names.append('actual_velocity')
        curve = self._previous_curve(state, position, direction, stations)
        if curve is not None:
            routes.append(curve)
            names.append('session_predicted_path')
        centers = np.stack(routes)
        tangents = np.gradient(centers, axis=1)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=-1, keepdims=True), 1e-8)
        right = np.stack((tangents[..., 1], -tangents[..., 0]), axis=-1)
        lateral = np.linspace(-.35, .35, 5)
        points = centers[:, None]+right[:, None]*lateral[None, :, None, None]
        height, valid = self.heights(points)
        spacing = extent/256
        lag = max(1, round(.18/spacing))
        previous = np.maximum(np.arange(257)-lag, 0)
        earlier = np.maximum(np.arange(257)-2*lag, 0)
        span = np.maximum(stations-stations[previous], spacing)
        change = np.max(np.abs(height-height[..., previous]), axis=1)
        slope = change/span
        rough = np.max(np.abs(height-2*height[..., previous]+height[..., earlier]), axis=1)
        cross = np.ptp(height, axis=1)
        risk = np.maximum.reduce((np.clip((change-.025)/.15, 0., 1.),
            np.clip((slope-.20)/.8, 0., 1.), np.clip((rough-.025)/.15, 0., 1.),
            np.clip((cross-.035)/.18, 0., 1.)))
        known = valid.all(axis=1) & valid[..., previous].all(axis=1)
        risk = np.where(known, risk, 1.)
        entry = np.maximum(minimum, desired*(1-(.35+.30*strength)*risk))
        entry = np.where(known, entry, 0.)
        free_distance = np.maximum(stations-planning_speed*reaction-margin, 0.)
        caps = np.where(risk > 1e-6, np.sqrt(entry**2+2*decel*free_distance), desired)
        route_index, station_index = np.unravel_index(int(caps.argmin()), caps.shape)
        cap = min(desired, float(caps[route_index, station_index]))
        scan_limited = needed+1. > 16.
        visibility_cap = -decel*reaction+math.sqrt((decel*reaction)**2+minimum**2+2*decel*max(extent-margin, 0.))
        if scan_limited:
            cap = min(cap, visibility_cap)
        distance = float(stations[station_index])
        entry_speed = float(entry[route_index, station_index])
        hazard = float(risk[route_index, station_index])
        hold_frames = max(0, state.hold_frames-frames)
        hold_speed = state.hold_speed
        if hazard > .05 and distance <= .7+actual*dt:
            hold_frames, hold_speed = max(0, 12-frames), min(desired, entry_speed)
        if state.hold_frames > 0:
            cap = min(cap, state.hold_speed)
        previous_target = desired if state.target_speed is None else min(desired, state.target_speed)
        if state.target_speed is None or (actual < .1 and not state.risk_limited):
            target = cap
        elif cap < previous_target:
            target = max(cap, previous_target-decel*dt)
        elif state.risk_limited:
            target = min(cap, previous_target+config['recovery_acceleration']*dt)
        else:
            target = cap
        target = min(desired, max(0., target))
        boundary = not bool(known[route_index, station_index])
        point = None if boundary or hazard <= 1e-6 else [float(centers[route_index, station_index, 0]),
            float(height[route_index, 2, station_index]), float(centers[route_index, station_index, 1])]
        braking = actual*reaction+max(actual*actual-entry_speed*entry_speed, 0.)/(2*decel)+margin
        forecast_braking = planning_speed*reaction+max(planning_speed*planning_speed-entry_speed*entry_speed, 0.)/(2*decel)+margin
        mode = ('boundary' if boundary else 'horizon_limited' if scan_limited and cap<=visibility_cap else
                'braking' if target < previous_target-1e-5 else
                'recovering' if target > previous_target+1e-5 else 'limited' if target < desired-1e-4 else 'clear')
        new = replace(state, frame=state.frame+frames, target_speed=target,
                      risk_limited=target < desired-1e-4,
                      hold_frames=hold_frames, hold_speed=hold_speed)
        info.update(target_speed_mps=target, cap_speed_mps=cap, entry_speed_mps=entry_speed, mode=mode,
            risk=hazard, risk_distance_m=distance if hazard > 0 else None, required_braking_distance_m=braking,
            braking_margin_m=distance-braking if hazard > 0 else None, reaction_seconds=reaction,
            requested_scan_braking_distance_m=forecast_braking, scan_required_extent_m=needed+1.,
            scan_extent_m=extent, scan_limited=scan_limited, visibility_cap_mps=visibility_cap,
            planning_speed_mps=planning_speed, known_fraction=float(valid.mean()),
            sample_count=int(valid.size), route=names[route_index], risk_point_world_xyz_m=point,
            path_source_frame=state.path_frame if curve is not None else None,
            overhead_ms=(time.perf_counter()-started)*1000)
        return (direction*target).tolist(), new, info
