"""Read-only, hash-bound offline SOMA reconstruction clips for the motion lab."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np


REVIEW_SCHEMA = "parc_tokenizer_terrain_review.v1"
BRANCHES = ("source", "codec")


def load_review_clips(review_root, vertex_count, terrain_mesh):
    """Load exported world poses without changing their roots, rotations or ground."""
    root = Path(review_root).resolve()
    manifest = json.loads((root / "review_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != REVIEW_SCHEMA
            or manifest.get("coordinate_frame") != "soma_y_up_world"
            or manifest.get("fps") != 30):
        raise ValueError("Unsupported offline SOMA review contract")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not 1 <= len(examples) <= 32:
        raise ValueError("Review must contain 1..32 examples")
    clips = {}
    for example in examples:
        name = example.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", name):
            raise ValueError("Invalid review example id")
        relative = Path(example["file"])
        path = (root / relative).resolve()
        if relative.is_absolute() or root not in path.parents or not path.is_file():
            raise ValueError("Review artifact must remain inside the review directory")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != example.get("sha256"):
            raise ValueError("Review artifact hash mismatch")
        with np.load(path, allow_pickle=False) as artifact:
            arrays = {key: artifact[key].copy() for key in (
                "roots_source", "local_source", "roots_codec", "local_codec",
                "hf", "min_point", "dx", "fps")}
        fps = float(arrays["fps"])
        heightfield, origin = arrays["hf"], arrays["min_point"]
        spacing = float(arrays["dx"])
        if (fps != 30 or heightfield.ndim != 2 or min(heightfield.shape) < 1
                or origin.shape != (3,) or not np.isfinite(spacing) or spacing <= 0
                or not all(np.isfinite(array).all() for array in arrays.values())):
            raise ValueError("Invalid world-space review geometry")
        count = len(arrays["roots_source"])
        if not 2 <= count <= 4096:
            raise ValueError("Invalid offline review frame count")
        for branch in BRANCHES:
            roots, local = arrays[f"roots_{branch}"], arrays[f"local_{branch}"]
            if roots.shape != (count, 3) or local.shape != (count, 30, 3, 3):
                raise ValueError("Review poses must use aligned SOMA30 local rotations and roots")
            if (not np.allclose(np.swapaxes(local, -1, -2) @ local, np.eye(3), atol=2e-3, rtol=0)
                    or not np.allclose(np.linalg.det(local), 1, atol=2e-3, rtol=0)):
                raise ValueError("Review local rotations must be proper rotation matrices")
        mesh = terrain_mesh(heightfield, origin, spacing)
        terrain = {"hf": heightfield.tolist(), "min_point": origin.tolist(), "dx": spacing,
                   "mesh": mesh, "mesh_source": "PARC.convert_heightfield_to_voxelized_trimesh"}
        for branch, title in (("source", "Source"), ("codec", "Codec 50k")):
            key = f"review_{name}_{branch}"
            if key in clips:
                raise ValueError("Duplicate review example id")
            clips[key] = {
                "id": key, "label": f"{title} | {example.get('label', name)}",
                "fps": fps, "frame_count": count, "vertex_count": vertex_count,
                "frame_stride_floats": 3 + vertex_count * 3,
                "terrain": terrain, "local": arrays[f"local_{branch}"],
                "roots": arrays[f"roots_{branch}"], "role": branch,
                "task": "offline source-encoded reconstruction, not live generation",
                "review_metrics": example.get("metrics", {}),
                "review_artifact_sha256": digest,
            }
    return clips


def native_terrain_to_parc(heightfield, origin, spacing):
    """Invert the shared [source-X,source-Y] -> native [Z,X] grid convention."""
    heightfield = np.asarray(heightfield)
    origin = np.asarray(origin)
    if (heightfield.ndim != 2 or origin.shape != (3,) or not np.isfinite(heightfield).all()
            or not np.isfinite(origin).all() or not np.isfinite(spacing) or spacing <= 0):
        raise ValueError("Invalid native terrain grid")
    parc_hf = np.ascontiguousarray(heightfield[::-1, :].T + origin[1])
    parc_origin = np.array([origin[0], -origin[2] - (heightfield.shape[0] - 1) * spacing])
    return parc_hf, parc_origin
