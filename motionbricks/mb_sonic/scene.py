"""One metric scene description for MuJoCo, geometric queries and rendering.

NumPy is the only runtime dependency here. MuJoCo checks live in the export CLI.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np


SCENE_SCHEMA = "motionbricks.mb_sonic.scene.v1"
SOMA_TO_MUJOCO = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
SOMA_TO_MUJOCO.setflags(write=False)


def soma_to_mujoco(xyz):
    """Rotate positions/vectors: (X, Y, Z) -> (X, -Z, Y), in metres."""
    return np.asarray(xyz) @ SOMA_TO_MUJOCO.T


def mujoco_to_soma(xyz):
    return np.asarray(xyz) @ SOMA_TO_MUJOCO


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(path):
    return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size)


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


@dataclass(frozen=True)
class RigidBox:
    """An explicitly authored, static, axis-aligned box in MuJoCo XYZ metres."""

    object_id: int
    name: str
    purpose: str
    center_xyz_m: tuple
    half_size_xyz_m: tuple

    def __post_init__(self):
        if self.object_id < 1 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError("Box needs a positive stable object ID and an MJCF-safe name")
        center = np.asarray(self.center_xyz_m, float)
        half = np.asarray(self.half_size_xyz_m, float)
        if center.shape != (3,) or half.shape != (3,) or not np.isfinite([center, half]).all() or (half <= 0).any():
            raise ValueError("Box needs finite XYZ center and strictly positive half sizes")


RIGID_BOX_FIXTURE = (
    RigidBox(1, "low_ceiling", "low_ceiling", (0., 0., 1.35), (1.2, 1., .1)),
    RigidBox(2, "sidewall", "sidewall", (1.5, 0., 1.2), (.1, 1.5, 1.2)),
    RigidBox(3, "platform", "platform", (-2., 0., .2), (.6, .6, .2)),
)


@dataclass(frozen=True)
class SceneManifest:
    schema: str
    scene_id: str
    units: dict
    transform: dict
    source: dict
    heightfield: dict
    objects: list
    known_domain: dict
    friction: dict
    mjcf: dict

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != SCENE_SCHEMA:
            raise ValueError("Unsupported SceneManifest schema: " + data["schema"])
        return cls(**data)

    def write(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class HeightQuery:
    height_m: np.ndarray
    normal_xyz: np.ndarray
    known: np.ndarray
    normal_unique: np.ndarray
    object_id: np.ndarray


@dataclass(frozen=True)
class SceneMesh:
    object_id: int
    geom_name: str
    vertices_m: np.ndarray
    faces: np.ndarray


def _read_source(path):
    with np.load(path, allow_pickle=False) as data:
        height = data["heightfield_y_m"].copy()
        origin = np.asarray(data["min_point_world_xyz_m"], np.float64)
        dx = float(data["dx_m"])
        metadata = json.loads(str(data["metadata_json"]))
        if "heightfield_valid" in data and not data["heightfield_valid"].all():
            raise ValueError("MuJoCo heightfields cannot represent unknown holes")
    if height.dtype != np.float32 or height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("Expected original float32 metric height[z,x] with at least 2x2 nodes")
    if origin.shape != (3,) or not np.isfinite(origin).all() or not 0 < dx < np.inf:
        raise ValueError("Expected finite SOMA XYZ origin and positive metric spacing")
    if not np.isfinite(height).all():
        raise ValueError("MuJoCo heightfields cannot represent unknown holes; no height fill is allowed")
    if metadata["coordinate_frame"] != "soma_world_y_up":
        raise ValueError("Source must explicitly declare soma_world_y_up")
    return height, origin, dx, metadata


def _include_xml(heightfield, objects, friction, binary_path):
    root = ET.Element("mujocoinclude")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "hfield", name="mb_heightfield", file=binary_path.as_posix(),
                  size=_numbers(heightfield["size_m"]))
    world = ET.SubElement(root, "worldbody")
    for obj in objects:
        attrs = dict(name=obj["geom_name"], type=obj["kind"], pos=_numbers(obj["center_xyz_m"]),
                     friction=_numbers(friction), priority="1", condim="3", contype="1", conaffinity="1",
                     group="0", rgba=_numbers(obj["rgba"]))
        if obj["kind"] == "hfield":
            attrs["hfield"] = "mb_heightfield"
        else:
            attrs["size"] = _numbers(obj["half_size_xyz_m"])
        ET.SubElement(world, "geom", **attrs)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def export_scene(source_npz, output_dir, *, scene_id, friction, friction_source,
                 boxes=(), base_depth_m=.1, catalog=None):
    """Export without resampling, image quantization, meshes-as-convex-floor or training.

    Friction is (sliding dimensionless, torsional metres, rolling metres). It is
    one fixed scene-wide assignment with priority 1, including every rigid box.
    """
    source_npz = Path(source_npz).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    height, origin, dx, metadata = _read_source(source_npz)
    friction = tuple(float(value) for value in friction)
    if len(friction) != 3 or not np.isfinite(friction).all() or min(friction) < 0 or not friction_source.strip():
        raise ValueError("Supply three nonnegative friction coefficients and their source")
    if not 0 < base_depth_m < np.inf:
        raise ValueError("Heightfield base depth must be positive metres")
    if len({box.object_id for box in boxes}) != len(boxes) or len({box.name for box in boxes}) != len(boxes):
        raise ValueError("Box IDs and names must be unique")
    nz, nx = height.shape
    x0, y0 = float(origin[0]), float(-origin[2] - (nz - 1) * dx)
    lo, hi = float(height.min()), float(height.max())
    extent = [(nx - 1) * dx, (nz - 1) * dx]
    scale = hi - lo if hi > lo else 1.
    center = [x0 + extent[0] / 2, y0 + extent[1] / 2, lo]
    objects = [dict(object_id=0, geom_name="mb_ground", kind="hfield", purpose="terrain",
                    center_xyz_m=center, rgba=[.42, .56, .45, 1.], friction_ref="scene")]
    for box in boxes:
        objects.append(dict(object_id=box.object_id, geom_name="mb_" + box.name, kind="box", purpose=box.purpose,
                            center_xyz_m=list(box.center_xyz_m), half_size_xyz_m=list(box.half_size_xyz_m),
                            quat_wxyz=[1., 0., 0., 0.], rgba=[.65, .54, .42, 1.], friction_ref="scene"))
    output_dir.mkdir(parents=True, exist_ok=False)
    source_copy = output_dir / "source_scene.npz"
    shutil.copyfile(source_npz, source_copy)
    binary = output_dir / "heightfield.bin"
    # Custom binary hfield rows are bottom-to-top. R_x(+pi/2) reverses SOMA Z.
    binary.write_bytes(struct.pack("<ii", nz, nx) + height[::-1].astype("<f4").tobytes(order="C"))
    hfield = dict(asset=_asset(binary), source_key="heightfield_y_m", source_shape_zx=[nz, nx],
                  source_dtype="float32", source_origin_soma_xyz_m=origin.tolist(), dx_m=dx,
                  source_height_semantics="absolute_soma_world_Y; origin_Y_is_not_added",
                  source_row_order="increasing_SOMA_Z", binary_row_order="increasing_MuJoCo_Y; source_rows_reversed",
                  binary_format="little_endian_int32_nrow_ncol_then_float32_metric_heights_row_major",
                  normalization=dict(min_m=lo, max_m=hi, elevation_scale_m=scale,
                                     rule="compiler_subtract_min_divide_range; flat_input_becomes_zero"),
                  size_m=[extent[0] / 2, extent[1] / 2, scale, base_depth_m], center_xyz_m=center,
                  triangulation="MuJoCo_XY_cell: (00,10,11),(00,11,01)",
                  base_semantics="solid_to_min_height_minus_base_depth; finite_vertical_boundary_walls",
                  resampling="none", image_quantization="none")
    include = output_dir / "scene_include.xml"
    include.write_text(_include_xml(hfield, objects, friction, binary), encoding="utf-8")
    manifest = SceneManifest(
        schema=SCENE_SCHEMA, scene_id=str(scene_id),
        units=dict(length="m", time="s", angle="rad", friction=["1", "m", "m"]),
        transform=dict(source_frame="soma_world_y_up", target_frame="mujoco_world_z_up",
                       rotation=SOMA_TO_MUJOCO.tolist(), translation_m=[0., 0., 0.],
                       quaternion_wxyz=[float(np.sqrt(.5)), float(np.sqrt(.5)), 0., 0.],
                       rule="(X,Y,Z)->(X,-Z,Y); proper_rotation; no_scale_no_rebase"),
        source=dict(scene=_asset(source_copy), original_scene_path=str(source_npz),
                    original_scene_sha256=sha256(source_npz), catalog=catalog, metadata=metadata),
        heightfield=hfield, objects=objects,
        known_domain=dict(ground_xy_m=[[x0, y0], [x0 + extent[0], y0 + extent[1]]],
                          boundary="closed_last_grid_node; not_n_times_dx", ground_cells="all_four_nodes_finite",
                          outside="unknown; no_clamp_no_flat_extension", object_scope="only_declared_objects",
                          ground_query="top_surface_only; no_signed_distance_claim"),
        friction=dict(coefficients=list(friction), source=friction_source, measured=False,
                      assignment="fixed_scene_wide", geom_priority=1, condim=3),
        mjcf=dict(include=_asset(include), asset_paths="absolute_at_export; reexport_after_relocation",
                  collision="full_resolution_hfield_and_independent_rigid_box_primitives"))
    path = output_dir / "scene_manifest.json"
    manifest.write(path)
    return path


def export_catalog(catalog_path, output_dir, *, friction, friction_source, include_rigid_fixture=True):
    """Export all fifteen catalog entries, plus a separate three-box flat fixture."""
    catalog_path = Path(catalog_path).resolve(strict=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = catalog["entries"]
    if catalog["schema"] != "motionbricks.pfnn_map_catalog.v1" or len(entries) != 15:
        raise ValueError("E2 requires the existing complete fifteen-map PFNN catalog")
    if len({entry["id"] for entry in entries}) != 15:
        raise ValueError("Catalog map IDs must be unique")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_files = [catalog_path.parent / entry["artifact"] / "scene.npz" for entry in entries]
    for entry, source in zip(entries, source_files):
        if sha256(source) != entry["scene_sha256"]:
            raise ValueError("Catalog scene SHA256 mismatch: " + entry["id"])
    provenance = dict(path=str(catalog_path), sha256=sha256(catalog_path))
    paths = [export_scene(source, output_dir / entry["id"], scene_id=entry["id"], friction=friction,
                          friction_source=friction_source, catalog=provenance)
             for entry, source in zip(entries, source_files)]
    if include_rigid_fixture:
        flat = source_files[[entry["id"] for entry in entries].index("000")]
        if np.ptp(_read_source(flat)[0]) != 0:
            raise ValueError("The explicit rigid-box fixture requires the catalog's flat 000 scene")
        paths.append(export_scene(flat, output_dir / "rigid_boxes", scene_id="rigid_boxes",
                                  friction=friction, friction_source=friction_source,
                                  boxes=RIGID_BOX_FIXTURE, catalog=provenance))
    return tuple(paths)


class CanonicalScene:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.manifest = SceneManifest.load(self.manifest_path)
        self.root = self.manifest_path.parent
        for asset in (self.manifest.source["scene"], self.manifest.heightfield["asset"], self.manifest.mjcf["include"]):
            if sha256(self.root / asset["file"]) != asset["sha256"]:
                raise ValueError("Scene asset SHA256 mismatch: " + asset["file"])
        if self.manifest.transform["rotation"] != SOMA_TO_MUJOCO.tolist():
            raise ValueError("Manifest transform differs from the canonical SOMA-to-MuJoCo rotation")
        height, _, _, _ = _read_source(self.root / self.manifest.source["scene"]["file"])
        self.height = height[::-1].astype(np.float64)
        self.height.setflags(write=False)
        self.dx = float(self.manifest.heightfield["dx_m"])
        self.bounds_xy_m = np.asarray(self.manifest.known_domain["ground_xy_m"], float)

    @property
    def scene_include_path(self):
        return self.root / self.manifest.mjcf["include"]["file"]

    def query_height(self, points_xy_m):
        """Ground-only triangular surface in MuJoCo XY, unknown outside the node rectangle.

        Normals at edges/nodes are one incident face normal, not unique normals.
        Overhead/side/platform geometry stays independent of this ground query.
        """
        points = np.asarray(points_xy_m, float)
        if points.shape[-1:] != (2,):
            raise ValueError("Ground query requires (...,2) MuJoCo XY positions")
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2)
        grid = (flat - self.bounds_xy_m[0]) / self.dx
        known = np.isfinite(flat).all(axis=1) & (flat >= self.bounds_xy_m[0]).all(axis=1) & (flat <= self.bounds_xy_m[1]).all(axis=1)
        height = np.full(len(flat), np.nan)
        normal = np.full((len(flat), 3), np.nan)
        unique = np.zeros(len(flat), bool)
        ids = np.flatnonzero(known)
        if len(ids):
            g = grid[ids]
            cell = np.minimum(np.floor(g).astype(int), np.array(self.height.shape[::-1]) - 2)
            u, v = (g - cell).T
            c, r = cell.T
            a, b = self.height[r, c], self.height[r, c + 1]
            d, e = self.height[r + 1, c + 1], self.height[r + 1, c]
            first = u >= v
            du = np.where(first, b - a, d - e)
            dv = np.where(first, d - b, e - a)
            height[ids] = a + u * du + v * dv
            n = np.stack((-du / self.dx, -dv / self.dx, np.ones_like(u)), axis=-1)
            normal[ids] = n / np.linalg.norm(n, axis=-1, keepdims=True)
            unique[ids] = (u > 1e-9) & (u < 1 - 1e-9) & (v > 1e-9) & (v < 1 - 1e-9) & (np.abs(u - v) > 1e-9)
        return HeightQuery(height.reshape(shape), normal.reshape(*shape, 3), known.reshape(shape),
                           unique.reshape(shape), np.where(known, 0, -1).reshape(shape))

    def query_height_point(self, x, y):
        """The same triangular height for one measured Root, without normal arrays."""
        if (not math.isfinite(x) or not math.isfinite(y)
                or x < self.bounds_xy_m[0, 0] or x > self.bounds_xy_m[1, 0]
                or y < self.bounds_xy_m[0, 1] or y > self.bounds_xy_m[1, 1]):
            return None, False
        gx = (x-self.bounds_xy_m[0, 0])/self.dx
        gy = (y-self.bounds_xy_m[0, 1])/self.dx
        c, r = min(math.floor(gx), self.height.shape[1]-2), min(math.floor(gy), self.height.shape[0]-2)
        u, v = gx-c, gy-r
        a, b = self.height[r, c], self.height[r, c+1]
        d, e = self.height[r+1, c+1], self.height[r+1, c]
        du, dv = (b-a, d-b) if u >= v else (d-e, e-a)
        return float(a+u*du+v*dv), True

    def box_signed_distance(self, points_xyz_m, object_id):
        """Exact box SDF; independent closed objects, not a heightfield approximation."""
        obj = next(obj for obj in self.manifest.objects if obj["object_id"] == object_id and obj["kind"] == "box")
        p = np.asarray(points_xyz_m, float) - np.asarray(obj["center_xyz_m"])
        q = np.abs(p) - np.asarray(obj["half_size_xyz_m"])
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)

    def render_meshes(self):
        """Full-resolution physical geometry, including finite hfield base and walls.

        Meshes are for queries/rendering. Physics uses the hfield/box MJCF geoms.
        """
        nr, nc = self.height.shape
        r, c = np.indices((nr, nc))
        xy = self.bounds_xy_m[0] + np.stack((c, r), -1) * self.dx
        vertices = np.concatenate((xy, self.height[..., None]), axis=-1).reshape(-1, 3)
        ids = np.arange(nr * nc).reshape(nr, nc)
        a, b, d, e = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
        faces = np.stack((np.stack((a, b, d), -1), np.stack((a, d, e), -1)), -2).reshape(-1, 3)
        border = np.concatenate((ids[0, :-1], ids[:-1, -1], ids[-1, :0:-1], ids[:0:-1, 0]))
        bottom = vertices[border].copy()
        bottom[:, 2] = self.manifest.heightfield["center_xyz_m"][2] - self.manifest.heightfield["size_m"][3]
        lower = np.arange(len(border)) + len(vertices)
        nxt, low_next = np.roll(border, -1), np.roll(lower, -1)
        walls = np.stack((np.stack((border, lower, low_next), -1), np.stack((border, low_next, nxt), -1)), -2).reshape(-1, 3)
        cap_center = bottom.mean(axis=0, keepdims=True)
        cap = np.stack((np.full(len(lower), len(vertices) + len(bottom)), low_next, lower), -1)
        ground = SceneMesh(0, "mb_ground", np.concatenate((vertices, bottom, cap_center)), np.concatenate((faces, walls, cap)))
        meshes = [ground]
        box_faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                              [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
        signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        for obj in self.manifest.objects:
            if obj["kind"] == "box":
                vertices = np.asarray(obj["center_xyz_m"]) + signs * np.asarray(obj["half_size_xyz_m"])
                meshes.append(SceneMesh(obj["object_id"], obj["geom_name"], vertices, box_faces.copy()))
        return tuple(meshes)


def load_scene(manifest_path):
    return CanonicalScene(manifest_path)
