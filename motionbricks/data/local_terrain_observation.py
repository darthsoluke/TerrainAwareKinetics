"""Causal local heightfield observations for kinematic terrain control.

Coordinates are Y-up, with XZ horizontal. Heightfields are indexed [Z, X],
and min_point_world_xyz_m is the world-space center of cell [0, 0]. This is
height-only geometry: it cannot represent overhangs or ceilings. Box-cell height
discontinuities imply step risers, but this observation does not sample their
vertical faces or normals explicitly.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class LocalTerrainObservationConfig:
    """Metric grids: near +/-0.4 m; intent forward 0.2..1.4 m and +/-0.6 m by default."""

    patch_height: int = 9
    patch_width: int = 9
    near_spacing_m: float = 0.10
    forward_spacing_m: float = 0.15
    forward_start_m: float = 0.20
    forward_lateral_spacing_m: float | None = None
    scale_forward_with_intent: bool = False

    def __post_init__(self):
        if not isinstance(self.scale_forward_with_intent, bool):
            raise ValueError('scale_forward_with_intent must be boolean')
        for name in ("patch_height", "patch_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if not math.isfinite(self.near_spacing_m) or self.near_spacing_m <= 0:
            raise ValueError("near_spacing_m must be finite and positive")
        if not math.isfinite(self.forward_spacing_m) or self.forward_spacing_m <= 0:
            raise ValueError("forward_spacing_m must be finite and positive")
        if not math.isfinite(self.forward_start_m) or self.forward_start_m < 0:
            raise ValueError("forward_start_m must be finite and nonnegative")
        if self.forward_lateral_spacing_m is not None and (
                not math.isfinite(self.forward_lateral_spacing_m)
                or self.forward_lateral_spacing_m <= 0):
            raise ValueError("forward_lateral_spacing_m must be finite and positive")


@dataclass(frozen=True)
class LocalTerrainObservation:
    """Two fixed-size local XYZ patches, explicit validity, and optional world XZ.

    `points` is [B, 2, H, W, 3] in meters relative to the current root origin,
    with axes X/Y/Z; `valid` is [B, 2, H, W]. Invalid Y values are zero sentinels;
    query XZ remains available and the full point must be masked by `valid`.
    `world_query_xz` is for diagnostics only.
    Probe fields reserve a future local 3D body/probe query with known/collision
    data; they are None here and do not imply heightfield overhang support.
    """

    points: torch.Tensor
    valid: torch.Tensor
    world_query_xz: torch.Tensor
    query_envelope_m: torch.Tensor | None = None
    body_probe_positions_local_xyz_m: torch.Tensor | None = None
    body_probe_valid: torch.Tensor | None = None
    body_probe_clearance_m: torch.Tensor | None = None

    def as_model_inputs(self):
        """Return the stable model-facing dictionary contract."""
        result = {"points": self.points, "valid": self.valid}
        if self.query_envelope_m is not None: result['query_envelope_m']=self.query_envelope_m
        return result


def _batch_tensor(
    value: torch.Tensor | Sequence[float], *, device: torch.device,
    dtype: torch.dtype, trailing_shape: tuple[int, ...], name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == len(trailing_shape):
        result = result.unsqueeze(0)
    if result.ndim != len(trailing_shape) + 1 or tuple(result.shape[1:]) != trailing_shape:
        raise ValueError(f"{name} must have shape {trailing_shape} or [B, {', '.join(map(str, trailing_shape))}]")
    return result


def extract_local_terrain_observation(
    heightfield_y_m: torch.Tensor | Sequence[Sequence[float]],
    min_point_world_xyz_m: torch.Tensor | Sequence[float],
    dx_m: float | torch.Tensor | Sequence[float],
    root_position_world_xyz_m: torch.Tensor | Sequence[float],
    root_heading_yaw_rad: float | torch.Tensor | Sequence[float],
    control_direction_local_xz: torch.Tensor | Sequence[float],
    *,
    heightfield_valid: torch.Tensor | None = None,
    config: LocalTerrainObservationConfig | None = None,
    query_controls: torch.Tensor | None = None,
    query_path_world_xz: torch.Tensor | None = None,
    query_path_valid: torch.Tensor | None = None,
) -> LocalTerrainObservation:
    """Sample a local terrain patch using only static map and current controls.

    Args:
        heightfield_y_m: [Z, X] or [B, Z, X] world-Y cell heights; NaN/Inf means unknown.
        min_point_world_xyz_m: [3] or [B, 3], center of the first heightfield cell.
        dx_m: positive square-cell spacing in meters.
        root_position_world_xyz_m: [3] or [B, 3] current root position in world XYZ.
        root_heading_yaw_rad: scalar or [B], yaw around +Y; zero faces world +Z.
        control_direction_local_xz: [2] or [B, 2], command in root-local XZ (e.g. WASD).
            A zero vector selects heading-forward for the directional patch.
        heightfield_valid: optional boolean [Z, X] or [B, Z, X] known-cell mask.
        config: metric grid dimensions and spacing.

    World-to-root yaw is the inverse of R_y(yaw): local X=cos*yaw*worldX -
    sin*yaw*worldZ and local Z=sin*yaw*worldX + cos*yaw*worldZ. Sampling uses
    nearest cells (no interpolation); any selected unknown/out-of-map cell is
    invalid, without clamping. Invalid returned heights are zero and must always
    be paired with `valid` by consumers.
    """
    config = config or LocalTerrainObservationConfig()
    heightfield = torch.as_tensor(heightfield_y_m)
    if not heightfield.is_floating_point():
        heightfield = heightfield.to(torch.float32)
    if heightfield.ndim == 2:
        heightfield = heightfield.unsqueeze(0)
    if heightfield.ndim != 3 or min(heightfield.shape[-2:]) <= 0:
        raise ValueError("heightfield_y_m must have shape [Z, X] or [B, Z, X]")
    device, dtype = heightfield.device, heightfield.dtype
    heightfield = heightfield.contiguous()
    map_batch, map_z, map_x = heightfield.shape
    origin = _batch_tensor(min_point_world_xyz_m, device=device, dtype=dtype,
                           trailing_shape=(3,), name="min_point_world_xyz_m")
    root = _batch_tensor(root_position_world_xyz_m, device=device, dtype=dtype,
                         trailing_shape=(3,), name="root_position_world_xyz_m")
    command = _batch_tensor(control_direction_local_xz, device=device, dtype=dtype,
                            trailing_shape=(2,), name="control_direction_local_xz")
    yaw = torch.as_tensor(root_heading_yaw_rad, device=device, dtype=dtype).reshape(-1)
    spacing = torch.as_tensor(dx_m, device=device, dtype=dtype).reshape(-1)
    batch = max(map_batch, origin.shape[0], root.shape[0], command.shape[0], yaw.numel(), spacing.numel())
    if any(size not in (1, batch) for size in (map_batch, origin.shape[0], root.shape[0], command.shape[0], yaw.numel(), spacing.numel())):
        raise ValueError("Map, pose, heading, and command batch sizes must match or be one")
    heightfield = heightfield.expand(batch, -1, -1)
    origin = origin.expand(batch, -1)
    root = root.expand(batch, -1)
    command = command.expand(batch, -1)
    yaw = yaw.expand(batch)
    if not torch.isfinite(origin).all() or not torch.isfinite(root).all() or not torch.isfinite(yaw).all() or not torch.isfinite(command).all():
        raise ValueError("Pose, origin, heading, and command must be finite")
    if not torch.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("dx_m must be finite, positive, and scalar or batch-sized")
    spacing = spacing.expand(batch)

    if heightfield_valid is None:
        known = torch.isfinite(heightfield)
    else:
        known = torch.as_tensor(heightfield_valid, device=device)
        if known.ndim == 2:
            known = known.unsqueeze(0)
        if known.shape not in ((1, map_z, map_x), (batch, map_z, map_x)) or known.dtype != torch.bool:
            raise ValueError("heightfield_valid must be boolean with shape [Z, X] or [B, Z, X]")
        known = known.expand(batch, -1, -1) & torch.isfinite(heightfield)

    near_x_axis = (torch.arange(config.patch_width, device=device, dtype=dtype)
                   - (config.patch_width - 1) / 2) * config.near_spacing_m
    near_z_axis = (torch.arange(config.patch_height, device=device, dtype=dtype)
                   - (config.patch_height - 1) / 2) * config.near_spacing_m
    near_x, near_z = torch.meshgrid(near_x_axis, near_z_axis, indexing="xy")
    near_local = torch.stack((near_x, near_z), dim=-1)

    lateral = (torch.arange(config.patch_width, device=device, dtype=dtype)
               - (config.patch_width - 1) / 2) * (
                   config.forward_lateral_spacing_m or config.forward_spacing_m)
    forward = config.forward_start_m + torch.arange(config.patch_height, device=device, dtype=dtype) * config.forward_spacing_m
    forward_grid, side_grid = torch.meshgrid(forward, lateral, indexing="ij")
    command_norm = torch.linalg.vector_norm(command, dim=-1)
    if config.scale_forward_with_intent:
        # The online command is a planar endpoint displacement, equal to desired
        # speed times the prediction horizon. Near support resolution stays fixed.
        near_radius = (config.patch_height - 1) / 2 * config.near_spacing_m
        maximum = config.forward_start_m + (config.patch_height - 1) * config.forward_spacing_m
        extent = (command_norm + near_radius).clamp(min=2 * near_radius, max=maximum)
        if query_controls is not None:
            controls = torch.as_tensor(query_controls,device=device,dtype=dtype)
            if controls.shape not in ((batch,2),(batch,4)) or not torch.isfinite(controls).all():
                raise ValueError('Query controls must be finite metric ranges and optional gain/base')
            if ((controls[:,0] != -1) & ((controls[:,0]<.8)|(controls[:,0]>20.))).any() or ((controls[:,1]<.3)|(controls[:,1]>6.)).any():
                raise ValueError('Query controls exceed the local observation envelope')
            if controls.shape[1]==4:
                if ((controls[:,2]<0)|(controls[:,2]>4)|(controls[:,3]<0)|(controls[:,3]>10)).any():
                    raise ValueError('Invalid speed-to-lookahead gain or base')
                extent=(command_norm*controls[:,2]+controls[:,3]).clamp(.8,20.)
            extent = torch.where(controls[:,0]<0,extent,controls[:,0])
        fraction = torch.linspace(0, 1, config.patch_height, device=device, dtype=dtype)
        adaptive_forward = config.forward_start_m + (extent[:, None] - config.forward_start_m) * fraction
        forward_flat = adaptive_forward[:, :, None].expand(-1, -1, config.patch_width).reshape(batch, -1)
    else:
        forward_flat = forward_grid.flatten()[None]
    if query_controls is not None:
        if not config.scale_forward_with_intent:
            raise ValueError('Query overrides require the intent-range checkpoint protocol')
        side_grid = side_grid.flatten()[None] * (controls[:,1:2] / (.15*((config.patch_width-1)/2)))
    else:
        side_grid = side_grid.flatten()
    angle = torch.where(command_norm > 1e-8, torch.atan2(command[:, 0], command[:, 1]), torch.zeros_like(command_norm))
    ca, sa = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    direction_local = torch.stack((ca * side_grid + sa * forward_flat,
                                   -sa * side_grid + ca * forward_flat), dim=-1)
    near_local = near_local.unsqueeze(0).expand(batch, -1, -1, -1)
    direction_local = direction_local.reshape(batch, config.patch_height, config.patch_width, 2)
    local_xz = torch.stack((near_local, direction_local), dim=1)

    # R_y(yaw) maps root-local XZ into world XZ; preserve root-local query coordinates.
    cy = torch.cos(yaw)[:, None, None, None]
    sy = torch.sin(yaw)[:, None, None, None]
    world_dx = cy * local_xz[..., 0] + sy * local_xz[..., 1]
    world_dz = -sy * local_xz[..., 0] + cy * local_xz[..., 1]
    world_x = root[:, None, None, None, 0] + world_dx
    world_z = root[:, None, None, None, 2] + world_dz
    world_xz = torch.stack((world_x, world_z), dim=-1)
    if query_path_world_xz is not None:
        route,route_available=sample_prediction_corridor(query_path_world_xz,query_path_valid,
            root[:,[0,2]],world_xz[:,1],forward_flat.reshape(batch,config.patch_height,config.patch_width),
            side_grid.reshape(batch if query_controls is not None else 1,config.patch_height,config.patch_width))
        corridor=torch.where(route_available[:,None,None,None],route,world_xz[:,1])
        world_xz=torch.stack((world_xz[:,0],corridor),1)
        world_x,world_z=world_xz[...,0],world_xz[...,1]
        delta=world_xz-root[:,None,None,None,[0,2]]
        local_xz=torch.stack((cy*delta[...,0]-sy*delta[...,1],
                              sy*delta[...,0]+cy*delta[...,1]),-1)
        local_xz=torch.stack((near_local,local_xz[:,1]),1)

    grid_x = (world_x - origin[:, None, None, None, 0]) / spacing[:, None, None, None]
    grid_z = (world_z - origin[:, None, None, None, 2]) / spacing[:, None, None, None]
    # Nearest-cell bins; index bounds reject map exterior instead of edge clamping.
    cell_x = torch.floor(grid_x + 0.5).to(torch.long)
    cell_z = torch.floor(grid_z + 0.5).to(torch.long)
    inside = (cell_x >= 0) & (cell_x < map_x) & (cell_z >= 0) & (cell_z < map_z)
    safe_x, safe_z = cell_x.clamp(0, map_x - 1), cell_z.clamp(0, map_z - 1)
    batch_ids = torch.arange(batch, device=device)[:, None, None, None]
    sampled = heightfield[batch_ids, safe_z, safe_x]
    valid = inside & known[batch_ids, safe_z, safe_x]
    relative_height = torch.where(
        valid, sampled - root[:, None, None, None, 1], torch.zeros_like(sampled))
    points = torch.stack((local_xz[..., 0], relative_height, local_xz[..., 1]), dim=-1)
    return LocalTerrainObservation(
        points=points,
        valid=valid,
        world_query_xz=world_xz,
        query_envelope_m=(local_xz.norm(dim=-1).flatten(1).amax(1).detach()+1e-5
                          if query_controls is not None or query_path_world_xz is not None else None),
    )


def trained_query_controls(requested,direction,config):
    """Preserve metric semantics within this checkpoint's trained sensor envelope.

    Larger surveys are sampled independently. Rescaling their XYZ into a near
    patch would falsely move distant geometry closer to the body.
    """
    if not config.scale_forward_with_intent:
        raise ValueError('Runtime controls require the trained intent-range protocol')
    controls=requested.clone()
    near_radius=(config.patch_height-1)/2*config.near_spacing_m
    extent=direction.norm(dim=-1)+near_radius
    if controls.shape[1]==4:
        extent=direction.norm(dim=-1)*controls[:,2]+controls[:,3]
    extent=torch.where(controls[:,0]<0,extent,controls[:,0])
    maximum=config.forward_start_m+(config.patch_height-1)*config.forward_spacing_m
    controls[:,0]=extent.clamp(2*near_radius,maximum)
    trained_halfwidth=(config.patch_width-1)/2*(config.forward_lateral_spacing_m or config.forward_spacing_m)
    controls[:,1]=controls[:,1].clamp(max=trained_halfwidth)
    return controls


def sample_prediction_corridor(path,valid,root_xz,straight_grid,forward_grid,side_grid):
    """Sample a model-proposed curve by arc length and its local side normals.

    Only XZ is accepted. Measured scene height is queried by the caller afterward.
    Beyond the proposed curve, scan along its final tangent; this is an explicit
    observation extension and is never substituted for generated motion.
    """
    batch=path.shape[0]
    if path.ndim!=3 or path.shape[-1]!=2 or path.shape[1]<1 or not torch.isfinite(path).all():
        raise ValueError('Prediction corridor requires finite [B,N,2] XZ proposals')
    if valid is None or valid.dtype!=torch.bool or valid.shape!=path.shape[:-1]:
        raise ValueError('Prediction corridor requires a boolean point-valid mask')
    if ((~valid[:,:-1]) & valid[:,1:]).any():
        raise ValueError('Prediction validity must be a contiguous prefix')
    path=path.detach();valid=valid.detach()
    counts=valid.sum(-1)
    last=path.gather(1,(counts-1).clamp_min(0)[:,None,None].expand(-1,1,2))
    path=torch.where(valid[...,None],path,last)
    vertices=torch.cat((root_xz[:,None],path),1)
    segments=vertices[:,1:]-vertices[:,:-1]
    lengths=segments.norm(dim=-1)
    cumulative=torch.cat((lengths.new_zeros(batch,1),lengths.cumsum(-1)),1)
    total=cumulative[:,-1]
    maximum=forward_grid[:,-1,0]
    # Preserve near-path detail when extending a short proposal into a long scan.
    knee=torch.minimum(maximum,torch.minimum(total.clamp_min(.4),total.new_full(total.shape,2.)))
    near_count=(forward_grid.shape[1]+1)//2;far_count=forward_grid.shape[1]-near_count
    near=torch.linspace(0,1,near_count,device=path.device,dtype=path.dtype)[None]*knee[:,None]
    far=knee[:,None]+torch.linspace(1/far_count,1,far_count,device=path.device,dtype=path.dtype)[None]*(maximum-knee)[:,None]
    distances=torch.cat((near,far),-1)
    indices=torch.searchsorted(cumulative.contiguous(),distances.contiguous(),right=True).clamp(1,path.shape[1])
    start=vertices.gather(1,(indices-1)[...,None].expand(-1,-1,2))
    segment=segments.gather(1,(indices-1)[...,None].expand(-1,-1,2))
    size=lengths.gather(1,indices-1)
    tangent=segment/size.clamp_min(1e-6)[...,None]
    along=distances-cumulative.gather(1,indices-1)
    centers=start+tangent*along[...,None]
    last_segment=torch.where(lengths>1e-6,torch.arange(path.shape[1],device=path.device)[None],0).amax(-1)
    direction=segments.gather(1,last_segment[:,None,None].expand(-1,1,2))[:,0]
    direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    straight_direction=straight_grid[:,-1,straight_grid.shape[2]//2]-root_xz
    straight_direction=straight_direction/straight_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    direction=torch.where((total>1e-6)[:,None],direction,straight_direction)
    beyond=distances>=total[:,None]
    centers=torch.where(beyond[...,None],last+direction[:,None]*(distances-total[:,None])[...,None],centers)
    tangent=torch.where(beyond[...,None],direction[:,None],tangent)
    right=torch.stack((tangent[...,1],-tangent[...,0]),-1)
    return centers[:,:,None]+right[:,:,None]*side_grid[...,None],counts>0
