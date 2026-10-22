"""Bounded 30 Hz SOMA adapter for the original MotionBricks representation."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from motionbricks.data.bones_seed import resample_rotations_root
from motionbricks.data.synthetic_dataset import collate_batch
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats


FPS = 30.0
REPRESENTATION_NAME = "somaskel30_dual_root_global_joints"
CONTRACT_VERSION = "native-mb-soma30-v1"
STATS_WINDOW_FRAMES = 65
STATS_WINDOW_STRIDE = 32
YAW_AUGMENTATION_DEGREES = (0, 90, 180, 270)


def build_native_representation(stats_folder=None):
    stats = Stats(folder=str(stats_folder)) if stats_folder else None
    return GlobalRootGlobalJoints(FPS, SOMASkeleton30(), REPRESENTATION_NAME, stats=stats)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dual_features(representation, global_features):
    local_features = representation.dual_rep.global_to_local(
        global_features, is_normalized=False, to_normalize=False
    )
    return torch.cat(
        [global_features[:, :5], local_features[:, :4], global_features[:, 5:]], dim=-1
    )


def build_train_stats(dataset_dir, records, manifest_sha256, stride=STATS_WINDOW_STRIDE):
    """Write stats from train-only 65-global-frame windows and four yaw rotations."""
    dataset_dir = Path(dataset_dir)
    stats_dir = dataset_dir / "stats"
    if stats_dir.exists():
        raise FileExistsError(f"Statistics already exist: {stats_dir}")
    representation = build_native_representation()
    count = windows = 0
    mean = m2 = None
    for record in records:
        if record["split"] != "train":
            continue
        with np.load(dataset_dir / record["shard"], allow_pickle=False) as shard:
            motion = torch.from_numpy(shard["global_features"].copy())
        if len(motion) < STATS_WINDOW_FRAMES:
            continue
        starts = sorted(set(list(range(0, len(motion) - STATS_WINDOW_FRAMES + 1, stride)) +
                            [len(motion) - STATS_WINDOW_FRAMES]))
        for start in starts:
            window = motion[start:start + STATS_WINDOW_FRAMES]
            for degrees in YAW_AUGMENTATION_DEGREES:
                augmented = representation.change_first_heading(
                    window, np.deg2rad(degrees), False, False
                )
                data = _dual_features(representation, augmented).numpy().astype(np.float64)
                n = len(data)
                chunk_mean = data.mean(0)
                chunk_m2 = ((data - chunk_mean) ** 2).sum(0)
                if count == 0:
                    count, mean, m2 = n, chunk_mean, chunk_m2
                else:
                    delta = chunk_mean - mean
                    m2 += chunk_m2 + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
                windows += 1
    if count == 0:
        raise RuntimeError("No train split windows of at least 65 frames")
    stats_dir.mkdir()
    std = np.sqrt(m2 / count)
    np.save(stats_dir / "mean.npy", mean.astype(np.float32))
    np.save(stats_dir / "std.npy", std.astype(np.float32))
    contract = {
        "version": CONTRACT_VERSION,
        "source_manifest_sha256": manifest_sha256,
        "split": "train",
        "window_global_frames": STATS_WINDOW_FRAMES,
        "window_local_frames": STATS_WINDOW_FRAMES - 1,
        "stride_frames": stride,
        "yaw_quadrature_degrees": list(YAW_AUGMENTATION_DEGREES),
        "windows_augmented": windows,
        "frames_augmented": count,
        "root_xz_std_m": std[[0, 2]].tolist(),
    }
    (stats_dir / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


class NativeMBMotionDataset(Dataset):
    """Random bounded windows from regenerated 30 Hz native MotionBricks features."""

    def __init__(self, dataset_dir, split, max_frames=200, min_frames=65, cache_max_bytes=0,
                 require_terrain=False, require_online_terrain=False, online_only=False):
        self.dataset_dir = Path(dataset_dir)
        self.require_terrain = bool(require_terrain)
        self.require_online_terrain = bool(require_online_terrain)
        self.online_only = bool(online_only)
        if self.online_only and not self.require_online_terrain:
            raise ValueError('online_only requires bound static terrain scenes')
        if self.require_online_terrain and not self.require_terrain:
            raise ValueError("require_online_terrain requires require_terrain=True")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if min_frames < 1 or max_frames < min_frames:
            raise ValueError("Require 1 <= min_frames <= max_frames")
        manifest = json.loads((self.dataset_dir / "accepted.json").read_text(encoding="utf-8"))
        self.records = [r for r in manifest if r.get("split") == split]
        self.records = [r for r in self.records if r.get("frames_30hz", 0) >= min_frames]
        self._online_artifacts = {}
        if self.require_online_terrain:
            native_manifest_path = self.dataset_dir / "native_manifest.json"
            if not native_manifest_path.is_file():
                raise ValueError("Online terrain requires a native_manifest.json artifact index")
            native_manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
            self._online_artifacts = native_manifest.get("artifact_sha256", {})
            if not isinstance(self._online_artifacts, dict):
                raise ValueError("native_manifest artifact_sha256 must be a mapping")
            for record in self.records:
                self._online_geometry_path(record)
        self.max_frames, self.min_frames = int(max_frames), int(min_frames)
        if cache_max_bytes < 0:
            raise ValueError("Cache size must be nonnegative")
        self.cache_max_bytes = int(cache_max_bytes)
        # Keep motion values directly addressable for existing cache inspectors.
        self._motion_cache = OrderedDict()
        self._terrain_cache = {}
        self._online_terrain_cache = {}
        self._cache_entry_bytes = {}
        self._cached_bytes = 0
        if not self.records:
            raise ValueError(f"No {split} records contain at least {min_frames} frames")
        representation = build_native_representation(self.dataset_dir / "stats")
        self._global_root_pos_indices = tuple(
            int(index) for index in representation.indices["global_root_pos"]
        )
        self._global_root_heading_indices = tuple(
            int(index) for index in representation.indices["global_root_heading"]
        )
        # Keep data-loader workers independent of a model that may move its stats to CUDA.
        self.stats = Stats(eps=representation.stats.eps, legacy=representation.stats.legacy)
        self.stats.register_from_tensors(
            representation.stats.mean.detach().cpu().clone(),
            representation.stats.std.detach().cpu().clone(),
        )

    def __len__(self):
        return len(self.records)

    def _online_geometry_path(self, record):
        relative = record.get("terrain_geometry_snapshot")
        expected = record.get("terrain_geometry_sha256")
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or not isinstance(expected, str) or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected.lower())
                or self._online_artifacts.get(relative) != expected):
            raise ValueError("Accepted record does not bind its terrain geometry snapshot")
        root = self.dataset_dir.resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Terrain geometry snapshot is missing or escapes the dataset: {relative}")
        return path

    def _load_online_terrain(self, record, motion):
        path = self._online_geometry_path(record)
        digest = _sha256_file(path)
        expected = record["terrain_geometry_sha256"]
        if digest != expected:
            raise ValueError(f"Terrain geometry snapshot SHA256 mismatch: {record['terrain_geometry_snapshot']}")

        required = {
            "heightfield_y_m", "min_point_world_xyz_m", "dx_m",
            "canonical_init_heading_quat", "canonical_root_pos_init_xz",
        }
        with np.load(path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Terrain geometry snapshot lacks {sorted(missing)}: {record['terrain_geometry_snapshot']}")
            heightfield = np.asarray(data["heightfield_y_m"])
            origin = np.asarray(data["min_point_world_xyz_m"])
            spacing = np.asarray(data["dx_m"])
            init_heading = np.asarray(data["canonical_init_heading_quat"])
            init_root_xz = np.asarray(data["canonical_root_pos_init_xz"])

        real_numeric = lambda value: (np.issubdtype(value.dtype, np.number)
                                      and not np.issubdtype(value.dtype, np.complexfloating))
        if heightfield.ndim != 2 or min(heightfield.shape) <= 0 or not real_numeric(heightfield):
            raise ValueError("Terrain heightfield must be a nonempty numeric [Z, X] array")
        if (origin.shape != (3,) or not real_numeric(origin)
                or not np.isfinite(origin).all()):
            raise ValueError("Terrain origin must be a finite world XYZ vector")
        if (spacing.ndim != 0 or not real_numeric(spacing)
                or not np.isfinite(spacing).all() or float(spacing) <= 0):
            raise ValueError("Terrain dx_m must be a finite positive scalar")
        if init_heading.shape == (1, 4):
            init_heading = init_heading[0]
        if init_root_xz.shape == (1, 2):
            init_root_xz = init_root_xz[0]
        if (init_heading.shape != (4,) or not real_numeric(init_heading)
                or not np.isfinite(init_heading).all()
                or not np.isclose(np.linalg.norm(init_heading), 1.0, atol=1e-3, rtol=0)):
            raise ValueError("canonical_init_heading_quat must be a finite unit wxyz quaternion")
        if (init_root_xz.shape != (2,) or not real_numeric(init_root_xz)
                or not np.isfinite(init_root_xz).all()):
            raise ValueError("canonical_root_pos_init_xz must be a finite XZ vector")

        scene = {
            "heightfield_y_m": torch.from_numpy(heightfield.astype(np.float32, copy=True)).detach(),
            "min_point_world_xyz_m": torch.from_numpy(origin.astype(np.float32, copy=True)).detach(),
            "dx_m": torch.tensor(float(spacing), dtype=torch.float32),
        }
        root_world = _native_root_world(
            motion,
            torch.as_tensor(init_heading.copy(), dtype=torch.float32),
            torch.as_tensor(init_root_xz.copy(), dtype=torch.float32),
            self._global_root_pos_indices,
            self._global_root_heading_indices,
        )
        return scene, root_world

    def __getitem__(self, index):
        record = self.records[index]
        key = record["shard"]
        if key in self._motion_cache:
            motion = self._motion_cache[key]
            self._motion_cache.move_to_end(key)
            terrain = self._terrain_cache.get(key)
            online_terrain = self._online_terrain_cache.get(key)
        else:
            with np.load(self.dataset_dir / key, allow_pickle=False) as shard:
                motion = torch.from_numpy(shard["global_features"].copy())
                terrain = None
                if self.require_terrain and not self.online_only:
                    from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL
                    if not {"terrain_points", "terrain_valid"}.issubset(shard.files):
                        raise ValueError(f"Missing local terrain observations: {key}")
                    points = torch.from_numpy(shard["terrain_points"].copy())
                    valid = torch.from_numpy(shard["terrain_valid"].copy())
                    if (points.shape != (len(motion), *TERRAIN_PROTOCOL["patch_shape"]) or
                            valid.shape != points.shape[:-1] or valid.dtype != torch.bool):
                        raise ValueError(f"Terrain must align with every motion frame: {key}")
                    terrain = (points, valid)
            online_terrain = self._load_online_terrain(record, motion) if self.require_online_terrain else None

            entry_tensors = (motion,) + (terrain if terrain is not None else ())
            if online_terrain is not None:
                scene, root_world = online_terrain
                entry_tensors += tuple(scene.values()) + (root_world,)
            size = sum(tensor.numel() * tensor.element_size() for tensor in entry_tensors)
            if 0 < size <= self.cache_max_bytes:
                while self._motion_cache and self._cached_bytes + size > self.cache_max_bytes:
                    evicted_key, _ = self._motion_cache.popitem(last=False)
                    self._terrain_cache.pop(evicted_key, None)
                    self._online_terrain_cache.pop(evicted_key, None)
                    self._cached_bytes -= self._cache_entry_bytes.pop(evicted_key)
                self._motion_cache[key] = motion
                if terrain is not None:
                    self._terrain_cache[key] = terrain
                if online_terrain is not None:
                    self._online_terrain_cache[key] = online_terrain
                self._cache_entry_bytes[key] = size
                self._cached_bytes += size
        if len(motion) < self.min_frames:
            raise ValueError(f"{record['filename']} is shorter than min_frames")
        count = min(len(motion), self.max_frames)
        start = int(torch.randint(len(motion) - count + 1, ())) if len(motion) > count else 0
        item = {"keyid": index, "motion": self.stats.normalize(motion[start:start + count])}
        if self.require_terrain and not self.online_only:
            points, valid = terrain if terrain is not None else self._terrain_cache[key]
            item["terrain_points"] = points[start:start + count].float().clone()
            item["terrain_valid"] = valid[start:start + count].clone()
        if self.require_online_terrain:
            scene, root_world = online_terrain
            item["terrain_scene"] = {name: value.detach() for name, value in scene.items()}
            item["terrain_root_world"] = root_world[start:start + count].detach().clone()
        return item


def _native_root_world(motion, init_heading_quat, init_root_xz,
                       root_position_indices, root_heading_indices):
    """Restore source canonical roots to their paired heightfield's SOMA world."""
    from motionbricks.motionlib.core.utils.rotations import quat_apply, quat_mul

    root_position = motion.index_select(
        1, torch.as_tensor(root_position_indices, device=motion.device, dtype=torch.long)
    )
    heading_features = motion.index_select(
        1, torch.as_tensor(root_heading_indices, device=motion.device, dtype=torch.long)
    )
    heading_angle = torch.atan2(heading_features[:, 1], heading_features[:, 0])
    canonical_heading = torch.stack((
        torch.cos(heading_angle / 2), torch.zeros_like(heading_angle),
        torch.sin(heading_angle / 2), torch.zeros_like(heading_angle),
    ), dim=-1)
    initial_rotation = init_heading_quat.to(device=motion.device, dtype=motion.dtype)
    initial_offset = init_root_xz.to(device=motion.device, dtype=motion.dtype)
    initial_rotation = initial_rotation.expand(len(motion), -1)
    world_position = quat_apply(initial_rotation, root_position)
    world_position = world_position.clone()
    world_position[:, [0, 2]] += initial_offset
    world_heading = quat_mul(initial_rotation, canonical_heading)
    yaw = 2 * torch.atan2(world_heading[:, 2], world_heading[:, 0])
    result = torch.cat((world_position, torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]), dim=-1)
    if result.shape != (len(motion), 5) or not torch.isfinite(result).all():
        raise ValueError("Canonical source features did not produce a finite world root trajectory")
    return result.detach()


def collate_native_batch(batch):
    """Official motion padding, plus optional causal scene inputs (not targets)."""
    result = collate_batch(batch)
    keys = ("terrain_points", "terrain_valid")
    has_stored_terrain = any(key in item for item in batch for key in keys)
    if has_stored_terrain and not all(key in item for item in batch for key in keys):
        raise ValueError("Every terrain-conditioned row needs points and validity")
    from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL
    max_frames = result["motion"].shape[1]
    shape = batch[0]['terrain_points'].shape[1:] if has_stored_terrain else None
    for item in batch if has_stored_terrain else ():
        points, valid = item["terrain_points"], item["terrain_valid"]
        if (points.shape[0] != len(item["motion"]) or points.shape[1:] != shape or
                tuple(points.shape[1:]) != tuple(TERRAIN_PROTOCOL['patch_shape']) or
                valid.shape != points.shape[:-1] or valid.dtype != torch.bool):
            raise ValueError("Terrain row geometry or frame alignment mismatch")
    for key in keys if has_stored_terrain else ():
        first = batch[0][key]
        padded = first.new_zeros((len(batch), max_frames, *first.shape[1:]))
        for row, item in enumerate(batch):
            padded[row, :len(item[key])] = item[key]
        result[key] = padded

    online_keys = ("terrain_scene", "terrain_root_world")
    has_online = any(key in item for item in batch for key in online_keys)
    if has_online:
        if not all(all(key in item for key in online_keys) for item in batch):
            raise ValueError("Every online-terrain row needs a scene and world root trajectory")
        for item in batch:
            scene = item["terrain_scene"]
            root_world = item["terrain_root_world"]
            if (not isinstance(scene, dict)
                    or not {"heightfield_y_m", "min_point_world_xyz_m", "dx_m"}.issubset(scene)
                    or scene["heightfield_y_m"].ndim != 2
                    or scene["min_point_world_xyz_m"].shape != (3,)
                    or scene["dx_m"].ndim != 0
                    or root_world.shape != (len(item["motion"]), 5)
                    or not root_world.is_floating_point()):
                raise ValueError("Online terrain scene or root trajectory has invalid geometry")
        max_frames = result["motion"].shape[1]
        first_root = batch[0]["terrain_root_world"]
        root_world = first_root.new_zeros((len(batch), max_frames, 5))
        for row, item in enumerate(batch):
            root_world[row, :len(item["motion"])] = item["terrain_root_world"]
        result["terrain_scene"] = [item["terrain_scene"] for item in batch]
        result["terrain_root_world"] = root_world.detach()
    return result


def native_training_loader(dataset, batch_size, workers, *, paper=False):
    """Full real-window batches even when there are fewer recording files than rows."""
    from torch.utils.data import DataLoader, RandomSampler

    kwargs = {"batch_size": batch_size, "num_workers": workers,
              "collate_fn": collate_native_batch, "persistent_workers": workers > 0, "pin_memory": True}
    if paper:
        # Epoch length is bookkeeping: the paper budget is measured in updates.
        # Each repeated recording lookup still draws an independent real crop.
        batches = max(256, (len(dataset) + batch_size - 1) // batch_size)
        kwargs.update(sampler=RandomSampler(dataset, replacement=True, num_samples=batches * batch_size),
                      drop_last=True)
    else:
        kwargs.update(shuffle=True)
    return DataLoader(dataset, **kwargs)
