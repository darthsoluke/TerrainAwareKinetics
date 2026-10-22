"""PFNN cell-prism terrain plus authored 3D solids, for observation queries.

Outside solids, certified min-component distances are exact. Inside intersecting
solids, min-SDF is a component distance, not distance to the union boundary.
These fields and their render meshes are not the independent 2 cm quality gate.
Unknown components never become empty space merely because another is known.
"""

from __future__ import annotations

import math

import torch

from .joint_sdf_observation import HeightfieldPrismSDF, PackedPrismSDF
from .scene_geometry_field import SceneGeometryField, SceneManifest, SceneMesh
from .unified_scene_field import AnalyticPlaneField, field_from_heightfield


def _check_points(points):
    if (points.ndim < 1 or points.shape[-1] != 3 or not points.is_floating_point()
            or not torch.isfinite(points).all()):
        raise ValueError('Finite floating XYZ points with shape [...,3] required')


def _combine(terrain, geometry, terrain_clearance, geometry_clearance):
    if terrain is None:
        return geometry[:3]
    if geometry is None:
        return terrain
    td, tn, tv = terrain
    gd, gn, gv = geometry[:3]
    choose_terrain = tv & (~gv | (td <= gd))
    distance = torch.where(choose_terrain, td, gd)
    normal = torch.where(choose_terrain[..., None], tn, gn)
    # A positive candidate is certifiable when every component is either already
    # certified or known empty throughout its distance ball. For interior values
    # require both component certificates; min-SDF is still not a union depth.
    valid = (tv | gv) & (tv | ((distance >= 0) & (distance <= terrain_clearance)))
    valid &= gv | ((distance >= 0) & (distance <= geometry_clearance))
    return (torch.where(valid, distance, 0),
            torch.where(valid[..., None], normal, 0), valid)


class CompositeSceneField:
    """Three-value field API for VolumetricEgoSampler, in SOMA Y-up metres.

    terrain_scene is the existing heightfield scene mapping, or None for a
    manifest-only scene. Neither component is synthesized when absent. prepare()
    preconstructs device/dtype buffers; querying a new device prepares it once.
    Terrain wins equal component distances; manifest face IDs remain unchanged
    in meshes (terrain uses reserved ID 0).
    """

    distance_semantics = 'min_component_sdf; exact_exterior; nonexact_overlap_interior'
    eligible_for_penetration_gate = False

    def __init__(self, terrain_scene, manifest: SceneManifest | None = None):
        if terrain_scene is None and manifest is None:
            raise ValueError('Nonempty terrain geometry or SceneManifest required')
        self.manifest = manifest
        self.terrain_scene = None
        self.terrain = None
        if terrain_scene is not None:
            height = torch.as_tensor(terrain_scene['heightfield_y_m']).detach().cpu().clone()
            origin = torch.as_tensor(terrain_scene['min_point_world_xyz_m']).detach().cpu().clone()
            spacing = float(terrain_scene['dx_m'])
            known = torch.isfinite(height)
            if 'heightfield_valid' in terrain_scene:
                mask = torch.as_tensor(terrain_scene['heightfield_valid']).detach().cpu().bool()
                if mask.shape != height.shape:
                    raise ValueError('Terrain validity must match heightfield shape')
                known &= mask
            if (height.ndim != 2 or min(height.shape) < 1 or not known.any()
                    or origin.shape != (3,) or not torch.isfinite(origin).all()
                    or not math.isfinite(spacing) or spacing <= 0):
                raise ValueError('Known terrain, finite XYZ origin and positive spacing required')
            self.terrain_scene = dict(heightfield_y_m=height,
                                      min_point_world_xyz_m=origin, dx_m=spacing,
                                      heightfield_valid=known)
            self.terrain = field_from_heightfield(self.terrain_scene)
        self._buffers = {}
        self.prepare()

    @property
    def terrain_bounds_xz_m(self):
        if self.terrain is None:
            return None
        if isinstance(self.terrain, AnalyticPlaneField):
            return self.terrain.bounds
        f = self.terrain
        return (float(f.origin[0]) - f.dx / 2,
                float(f.origin[0]) + (f.heights.shape[1] - .5) * f.dx,
                float(f.origin[2]) - f.dx / 2,
                float(f.origin[2]) + (f.heights.shape[0] - .5) * f.dx)

    def prepare(self, device='cpu', dtype=torch.float32):
        device = torch.device(device)
        key = (device, dtype)
        if key not in self._buffers:
            sample = torch.empty(0, 3, device=device, dtype=dtype)
            geometry = None
            if self.manifest is not None:
                geometry = SceneGeometryField(self.manifest)
                for name in ('_centers', '_half', '_rotations', '_lower', '_upper'):
                    setattr(geometry, name, getattr(geometry, name).to(device=device, dtype=dtype))
                geometry._ids = geometry._ids.to(device)
            terrain = self.terrain
            holes = None
            if isinstance(terrain, HeightfieldPrismSDF):
                holes = terrain._tensors(sample)[1][:2]
            bounds = None if terrain is None else sample.new_tensor(self.terrain_bounds_xz_m)
            self._buffers[key] = geometry, bounds, holes
        return self

    def _prepared(self, points):
        self.prepare(points.device, points.dtype)
        return self._buffers[(points.device, points.dtype)]

    def _clearance(self, points, buffers):
        geometry, bounds, holes = buffers
        terrain_margin = geometry_margin = None
        if bounds is not None:
            terrain_margin = torch.stack((points[..., 0] - bounds[0], bounds[1] - points[..., 0],
                                          points[..., 2] - bounds[2], bounds[3] - points[..., 2]), -1).amin(-1)
            if holes is not None:
                centers, half = holes
                for start in range(0, len(centers), 1024):
                    gap = ((points[..., None, [0, 2]] - centers[start:start + 1024]).abs()
                           - half[start:start + 1024]).clamp_min(0).norm(dim=-1).amin(-1)
                    terrain_margin = torch.minimum(terrain_margin, gap)
                # A point in an unknown cell has no empty-ball certificate,
                # including zero-distance candidates on a hole boundary.
                f = self.terrain
                _, _, _, known, origin = f._tensors(points)
                cells = torch.floor((points[..., [0, 2]] - origin[[0, 2]]) / f.dx + .5).long()
                x, z = cells.unbind(-1)
                in_map = (x >= 0) & (x < known.shape[1]) & (z >= 0) & (z < known.shape[0])
                in_map &= known[z.clamp(0, known.shape[0] - 1), x.clamp(0, known.shape[1] - 1)]
                terrain_margin = terrain_margin.masked_fill(~in_map, -torch.inf)
        if geometry is not None:
            geometry_margin = torch.minimum(points - geometry._lower,
                                             geometry._upper - points).amin(-1)
        return terrain_margin, geometry_margin

    def _with_terrain(self, points, terrain):
        if self.manifest is None:
            return terrain
        buffers = self._prepared(points)
        geometry = buffers[0].query(points)
        return _combine(terrain, geometry, *self._clearance(points, buffers))

    def query(self, points):
        _check_points(points)
        # HeightfieldPrismSDF does not support an empty query; preserve its API
        # on nonempty inputs and return correctly shaped empties here.
        if points.numel() == 0:
            return points[..., 0], points.clone(), torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
        terrain = None if self.terrain is None else self.terrain.query(points)
        return self._with_terrain(points, terrain)

    def meshes(self, *, dtype=torch.float32, device='cpu'):
        """Existing flat cell tops and vertical step faces, plus manifest meshes.

        As in the voxel terrain renderer, bounds are cell centers +/- dx/2.
        Unknown cells are omitted. No interpolated slopes, boundary caps, or
        invented terrain bottom are supplied; this is not a closed union mesh.
        """
        result = []
        if self.terrain_scene is not None:
            scene = self.terrain_scene
            hf, known = scene['heightfield_y_m'], scene['heightfield_valid']
            origin, dx = scene['min_point_world_xyz_m'], scene['dx_m']
            # Reuse the prism's equal-height rectangles for flat tops.
            prism = self.terrain if isinstance(self.terrain, HeightfieldPrismSDF) else HeightfieldPrismSDF(scene)
            quads = []
            for x0, x1, z0, z1, h in prism.rectangles:
                a, b = float(origin[0]) + (x0 - .5) * dx, float(origin[0]) + (x1 - .5) * dx
                c, d = float(origin[2]) + (z0 - .5) * dx, float(origin[2]) + (z1 - .5) * dx
                quads.append(((a, h, c), (a, h, d), (b, h, d), (b, h, c)))
            # The renderer connects adjacent cells vertically, never diagonally.
            for axis in (0, 1):
                first = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
                second = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
                changed = known[first] & known[second] & (hf[first] != hf[second])
                for z, x in changed.nonzero().tolist():
                    lo, hi = sorted((float(hf[z, x]), float(hf[z + (axis == 0), x + (axis == 1)])))
                    a, c = float(origin[0]) + (x - .5) * dx, float(origin[2]) + (z - .5) * dx
                    quad = ((a, lo, c + dx), (a, hi, c + dx), (a + dx, hi, c + dx), (a + dx, lo, c + dx)) if axis == 0 else (
                        (a + dx, lo, c), (a + dx, lo, c + dx), (a + dx, hi, c + dx), (a + dx, hi, c))
                    if hf[z, x] > hf[z + (axis == 0), x + (axis == 1)]:
                        quad = quad[::-1]
                    quads.append(quad)
            vertices = torch.tensor(quads, device=device, dtype=dtype).reshape(-1, 3)
            base = torch.arange(len(quads), device=device)[:, None, None] * 4
            triangles = (base + torch.tensor(((0, 1, 2), (0, 2, 3)), device=device)).reshape(-1, 3)
            result.append(SceneMesh(0, 'pfnn_terrain', 'terrain', vertices, triangles,
                                    torch.zeros(len(triangles), dtype=torch.int64, device=device)))
        if self.manifest is not None:
            result.extend(self.manifest.meshes(dtype=dtype, device=device))
        return tuple(result)


class BatchedSceneField:
    """One field per batch row; prepare buffers before timing the query path.

    Float32 nonflat terrain uses existing PackedPrismSDF; planes retain analytic
    normals even at zero distance. Float64 uses the scalar reference terrain
    implementation, avoiding the packed backend's float32 geometry conversion.
    OBB composition is initially per scene; no DDP scalability claim is made.
    """

    def __init__(self, fields, *, device='cpu', dtype=torch.float32):
        self.fields = tuple(fields)
        if not self.fields or not all(isinstance(f, CompositeSceneField) for f in self.fields):
            raise ValueError('Nonempty CompositeSceneField list required')
        self._prism_indices = tuple(i for i, f in enumerate(self.fields)
                                    if isinstance(f.terrain, HeightfieldPrismSDF))
        self._buffers = {}
        self.prepare(device, dtype)

    def prepare(self, device='cpu', dtype=torch.float32):
        device = torch.device(device)
        key = (device, dtype)
        if key not in self._buffers:
            for field in self.fields:
                field.prepare(device, dtype)
            indices = self._prism_indices
            packed = None
            if indices and dtype == torch.float32:
                prisms = [self.fields[i].terrain for i in indices]
                scenes = [dict(heightfield_y_m=torch.where(torch.from_numpy(f.known),
                                                          torch.from_numpy(f.heights), torch.nan),
                               min_point_world_xyz_m=f.origin, dx_m=f.dx) for f in prisms]
                packed = PackedPrismSDF(prisms, scenes, device)
            self._buffers[key] = torch.tensor(indices, dtype=torch.long, device=device), packed
        return self

    def query(self, points):
        _check_points(points)
        if points.ndim < 2 or points.shape[0] != len(self.fields):
            raise ValueError('Batched XYZ points require one row per scene')
        if points.numel() == 0:
            return points[..., 0], points.clone(), torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
        self.prepare(points.device, points.dtype)
        indices, packed = self._buffers[(points.device, points.dtype)]
        terrain_rows = {}
        if packed is not None:
            values = packed.query(points.index_select(0, indices))
            for row, i in enumerate(self._prism_indices):
                terrain_rows[i] = tuple(value[row] for value in values)
        results = []
        for i, field in enumerate(self.fields):
            if i in terrain_rows:
                results.append(field._with_terrain(points[i], terrain_rows[i]))
            else:
                results.append(field.query(points[i]))
        return tuple(torch.stack([r[j] for r in results]) for j in range(3))


def cpu_reference_query(terrain_scene, manifest, points):
    """Scalar CPU reference using the original terrain and OBB query backends."""
    if points.device.type != 'cpu':
        raise ValueError('CPU reference requires CPU points')
    return CompositeSceneField(terrain_scene, manifest).query(points)


@torch.no_grad()
def path_observation(sampler, history_root, root_path, coarse, field):
    """Requery only 512 corridor points along the detached latest Root proposal.

    root_path is [B,8,2] world XZ, already selected by the Root head (not future
    ground truth). Body and actual-velocity points/samples remain factual.
    Interpolation, endpoint tangents and stationary heading match
    PredictedRootObservation._curve exactly.
    """
    from .root_path_terrain import PredictedRootObservation

    batch = len(history_root)
    if (history_root.shape != (batch, 4, 5) or root_path.shape != (batch, 8, 2)
            or sampler.config.grid_size != 8 or coarse['points'].shape != (batch, 3, 512, 3)
            or history_root.device != root_path.device or history_root.dtype != root_path.dtype
            or not torch.isfinite(history_root).all() or not torch.isfinite(root_path).all()):
        raise ValueError('Four factual Root frames, eight world XZ stations and three 512-point regions required')
    history_root, root_path = history_root.detach(), root_path.detach()
    centers, right = PredictedRootObservation._curve(None, root_path, 8, history_root)
    points = coarse['points'].detach().clone()
    ids = torch.arange(512, device=points.device)
    lateral = torch.linspace(-sampler.config.corridor_halfwidth_m, sampler.config.corridor_halfwidth_m,
                             8, device=points.device, dtype=points.dtype)
    xz = centers[:, ids % 8] + right[:, ids % 8] * lateral[ids // 64][None, :, None]
    points[:, 1, :, 0], points[:, 1, :, 2] = xz[..., 0], xz[..., 1]
    changed = field.query(points[:, 1:2])
    samples = tuple(torch.cat((coarse[key][:, :1].detach(), value, coarse[key][:, 2:].detach()), 1)
                    for key, value in zip(('distance', 'normal', 'valid'), changed))
    return sampler.observe_points(history_root, points, field, samples=samples)
