"""World-space scene queries and causal, fixed-budget full-body observations."""

from dataclasses import dataclass

import torch

from .joint_sdf_observation import HeightfieldPrismSDF


class AnalyticPlaneField:
    """An explicitly authored static plane, optionally bounded by a known map."""

    def __init__(self, height, bounds_xz=None):
        self.height = float(height)
        self.bounds = bounds_xz
        if not torch.isfinite(torch.tensor(self.height)):
            raise ValueError('Plane height must be finite')

    def query(self, points):
        if points.shape[-1] != 3 or not torch.isfinite(points).all():
            raise ValueError('Finite world XYZ points required')
        distance = points[..., 1] - self.height
        normal = torch.zeros_like(points)
        normal[..., 1] = 1
        valid = torch.ones_like(distance, dtype=torch.bool)
        if self.bounds is not None:
            xmin, xmax, zmin, zmax = self.bounds
            edge = torch.stack((points[..., 0]-xmin, xmax-points[..., 0],
                                points[..., 2]-zmin, zmax-points[..., 2]), -1).amin(-1)
            valid = (edge >= 0) & (distance.abs() <= edge + 1e-6)
        return (torch.where(valid, distance, 0),
                torch.where(valid[..., None], normal, 0), valid)


class AnalyticBoxField:
    """Exact single solid box in otherwise known empty space; geometry tests only."""

    def __init__(self, center, half_extent):
        self.center = torch.as_tensor(center, dtype=torch.float64)
        self.half = torch.as_tensor(half_extent, dtype=torch.float64)
        if (self.center.shape != (3,) or self.half.shape != (3,)
                or not torch.isfinite(self.center).all() or not torch.isfinite(self.half).all()
                or not (self.half > 0).all()):
            raise ValueError('Finite center and positive XYZ half extents required')

    def query(self, points):
        if points.shape[-1] != 3 or not torch.isfinite(points).all():
            raise ValueError('Finite world XYZ points required')
        delta = points-self.center.to(points)
        q = delta.abs()-self.half.to(points)
        outside = q.clamp_min(0)
        length = outside.norm(dim=-1)
        distance = length+q.amax(-1).clamp_max(0)
        sign = torch.where(delta >= 0, 1., -1.)
        exterior_normal = outside*sign/length.clamp_min(1e-12)[..., None]
        axis = q.argmax(-1, keepdim=True)
        interior_normal = torch.zeros_like(points).scatter(-1, axis, sign.gather(-1, axis))
        normal = torch.where((length > 0)[..., None], exterior_normal, interior_normal)
        return distance, normal, torch.ones_like(distance, dtype=torch.bool)


def field_from_heightfield(scene):
    """Select a geometry algorithm, never a motion model or a flat-motion gate."""
    height = torch.as_tensor(scene['heightfield_y_m']).detach().cpu()
    known = torch.isfinite(height)
    if 'heightfield_valid' in scene:
        known &= torch.as_tensor(scene['heightfield_valid']).cpu().bool()
    if height.ndim != 2 or not known.any():
        raise ValueError('Scene has no known heightfield observations')
    if known.all() and torch.equal(height, height[:1, :1].expand_as(height)):
        origin = torch.as_tensor(scene['min_point_world_xyz_m']).cpu()
        spacing = float(scene['dx_m'])
        if spacing <= 0:
            raise ValueError('Positive scene spacing required')
        bounds = (float(origin[0])-spacing/2, float(origin[0])+(height.shape[1]-.5)*spacing,
                  float(origin[2])-spacing/2, float(origin[2])+(height.shape[0]-.5)*spacing)
        return AnalyticPlaneField(float(height[0, 0]), bounds)
    return HeightfieldPrismSDF(scene)


@dataclass(frozen=True)
class VolumeSampling:
    grid_size: int = 8
    body_halfwidth_m: float = .8
    corridor_halfwidth_m: float = .7
    vertical_below_m: float = 2.0
    vertical_above_m: float = 2.0
    min_lookahead_m: float = 1.2
    lookahead_margin_m: float = .6
    direction_epsilon_mps: float = .05

    def __post_init__(self):
        if self.grid_size < 4 or self.grid_size % 2:
            raise ValueError('An even grid size >= 4 is required')
        for name, value in vars(self).items():
            if name != 'grid_size' and (not torch.isfinite(torch.tensor(value)) or value <= 0):
                raise ValueError('Sampling dimensions must be positive and finite')


class VolumetricEgoSampler:
    """Three separate 3D lattices: body, requested direction, actual motion."""

    feature_dim = 15

    def __init__(self, config=None):
        self.config = config or VolumeSampling()
        self._grids = {}

    def points(self, history_root, velocity_xz, horizon_seconds):
        if history_root.ndim != 3 or history_root.shape[1:] != (4, 5):
            raise ValueError('Exactly four factual Root frames required')
        if velocity_xz.shape != (len(history_root), 2):
            raise ValueError('World VX/VZ command shape mismatch')
        if (not torch.isfinite(history_root).all() or not torch.isfinite(velocity_xz).all()
                or not 0 < horizon_seconds <= 3):
            raise ValueError('Finite observations and bounded horizon required')
        cfg = self.config
        key = (history_root.device, history_root.dtype)
        if key not in self._grids:
            axis = torch.linspace(-1, 1, cfg.grid_size, device=history_root.device, dtype=history_root.dtype)
            self._grids[key] = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        grid = self._grids[key]
        root = history_root[:, -1]
        yaw = torch.atan2(root[:, 4], root[:, 3])
        face = torch.stack((yaw.sin(), yaw.cos()), -1)
        actual_velocity = (history_root[:, 3, [0, 2]]-history_root[:, 2, [0, 2]])*30

        def direction(v):
            speed = v.norm(dim=-1, keepdim=True)
            return torch.where(speed >= cfg.direction_epsilon_mps,
                               v/speed.clamp_min(cfg.direction_epsilon_mps), face)

        forward = torch.stack((face, direction(velocity_xz), direction(actual_velocity)), 1)
        right = torch.stack((forward[..., 1], -forward[..., 0]), -1)
        speeds = torch.stack((velocity_xz.norm(dim=-1), actual_velocity.norm(dim=-1)), 1)
        reach = (speeds*horizon_seconds+cfg.lookahead_margin_m).clamp_min(cfg.min_lookahead_m)
        longitudinal = torch.cat((grid[None, None, :, 2].expand(len(root), 1, -1)*cfg.body_halfwidth_m,
                                  (grid[None, None, :, 2]+1)*.5*reach[..., None]), 1)
        width = root.new_tensor([cfg.body_halfwidth_m, cfg.corridor_halfwidth_m, cfg.corridor_halfwidth_m])
        lateral = grid[None, None, :, 0]*width[None, :, None]
        xz = right[:, :, None]*lateral[..., None]+forward[:, :, None]*longitudinal[..., None]
        y = (grid[:, 1]+1)*.5*(cfg.vertical_below_m+cfg.vertical_above_m)-cfg.vertical_below_m
        offset = torch.stack((xz[..., 0], y[None, None].expand_as(xz[..., 0]), xz[..., 1]), -1)
        return root[:, None, None, :3]+offset

    def observe(self, history_root, velocity_xz, field, horizon_seconds):
        # Observation values are causal inputs, not a differentiable collision corrector.
        with torch.no_grad():
            history_root, velocity_xz = history_root.detach(), velocity_xz.detach()
            points = self.points(history_root, velocity_xz, horizon_seconds)
            return self.observe_points(history_root, points, field)

    @torch.no_grad()
    def observe_points(self, history_root, points, field, *, samples=None):
        """Encode the same physical features at explicitly selected query points."""
        with torch.no_grad():
            distance, normal, valid = field.query(points) if samples is None else samples
            if not valid.flatten(1).any(-1).all():
                raise ValueError('All scene observations are unknown; no flat fallback is allowed')
            root = history_root[:, -1]
            relative = points-root[:, None, None, :3]
            yaw = torch.atan2(root[:, 4], root[:, 3])
            c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
            local = torch.stack((c*relative[..., 0]-s*relative[..., 2], relative[..., 1],
                                 s*relative[..., 0]+c*relative[..., 2]), -1)
            heading = torch.stack((yaw.cos(), yaw.sin()), -1)[:, None, None].expand(*points.shape[:-1], 2)
            region = torch.eye(3, device=points.device, dtype=points.dtype)[None, :, None].expand(*points.shape[:-1], 3)
            features = torch.cat((torch.asinh(relative), torch.asinh(local), torch.asinh(distance)[..., None],
                                  normal, heading, region), -1)
            return dict(points=points, distance=distance, normal=normal, valid=valid,
                        features=torch.where(valid[..., None], features, 0))
