import json

import numpy as np
import pytest
import torch

from motionbricks.data import native_mb_data
from motionbricks.data.native_mb_data import NativeMBMotionDataset
from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL


FEATURE_DIM = 366
STATS_DIM = 370


def make_dataset(root, *, frame_counts=(12,), cache_max_bytes=0, require_terrain=True,
                 terrain_overrides=None):
    (root / "stats").mkdir(parents=True)
    np.save(root / "stats" / "mean.npy", np.zeros(STATS_DIM, dtype=np.float32))
    np.save(root / "stats" / "std.npy", np.ones(STATS_DIM, dtype=np.float32))
    records = []
    arrays = {}
    terrain_overrides = terrain_overrides or {}
    for index, frames in enumerate(frame_counts):
        shard = f"shards/clip_{index}.npz"
        motion = np.arange(frames * FEATURE_DIM, dtype=np.float32).reshape(frames, FEATURE_DIM)
        motion[:, 0] = np.arange(frames, dtype=np.float32)
        points = np.broadcast_to(
            np.arange(frames, dtype=np.float32).reshape(frames, 1, 1, 1, 1),
            (frames, *TERRAIN_PROTOCOL["patch_shape"]),
        ).copy()
        valid = np.ones(points.shape[:-1], dtype=np.bool_)
        data = {"global_features": motion, "terrain_points": points, "terrain_valid": valid}
        for name, value in terrain_overrides.get(index, {}).items():
            if value is None:
                data.pop(name, None)
            else:
                data[name] = value
        (root / "shards").mkdir(exist_ok=True)
        np.savez_compressed(root / shard, **data)
        arrays[shard] = {"global_features": motion, "terrain_points": points, "terrain_valid": valid}
        records.append({
            "filename": f"clip_{index}",
            "split": "train",
            "shard": shard,
            "frames_30hz": frames,
        })
    (root / "accepted.json").write_text(json.dumps(records), encoding="utf-8")
    dataset = NativeMBMotionDataset(
        root, "train", max_frames=5, min_frames=4, cache_max_bytes=cache_max_bytes,
        require_terrain=require_terrain,
    )
    return dataset, arrays


def tensor_bytes(array):
    tensor = torch.from_numpy(array.copy())
    return tensor.numel() * tensor.element_size()


def count_np_loads(monkeypatch):
    real_load = native_mb_data.np.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(native_mb_data.np, "load", counted_load)
    return calls


def test_terrain_cache_hot_read_uses_no_npz_and_preserves_seeded_output(tmp_path, monkeypatch):
    dataset, arrays = make_dataset(tmp_path)
    shard, raw = next(iter(arrays.items()))
    entry_bytes = sum(tensor_bytes(value) for value in raw.values())
    dataset.cache_max_bytes = entry_bytes
    loads = count_np_loads(monkeypatch)

    torch.manual_seed(847)
    cold = dataset[0]
    assert len(loads) == 1
    assert dataset._cached_bytes == entry_bytes
    assert list(dataset._motion_cache) == [shard]
    assert set(dataset._terrain_cache) == {shard}

    torch.manual_seed(847)
    hot = dataset[0]
    assert len(loads) == 1
    assert cold.keys() == hot.keys()
    for key in cold:
        if torch.is_tensor(cold[key]):
            torch.testing.assert_close(cold[key], hot[key], rtol=0, atol=0)
        else:
            assert cold[key] == hot[key]
    torch.testing.assert_close(
        dataset.stats.unnormalize(hot["motion"])[:, 0],
        hot["terrain_points"][:, 0, 0, 0, 0],
    )


@pytest.mark.parametrize(
    "override",
    [
        {"terrain_points": np.zeros((12, 2, 8, 9, 3), dtype=np.float32)},
        {"terrain_valid": np.ones((12, 2, 9, 9), dtype=np.uint8)},
        {"terrain_valid": None},
    ],
)
def test_terrain_cache_rejects_invalid_schema(tmp_path, override):
    dataset, _ = make_dataset(tmp_path, terrain_overrides={0: override})

    with pytest.raises(ValueError, match="[Tt]errain"):
        dataset[0]

    assert not dataset._motion_cache
    assert dataset._cached_bytes == 0


def test_zero_budget_disables_motion_and_terrain_cache(tmp_path, monkeypatch):
    dataset, _ = make_dataset(tmp_path, cache_max_bytes=0)
    loads = count_np_loads(monkeypatch)

    torch.manual_seed(29)
    first = dataset[0]
    torch.manual_seed(29)
    second = dataset[0]

    assert len(loads) == 2
    assert not dataset._motion_cache
    assert not dataset._terrain_cache
    assert dataset._cached_bytes == 0
    torch.testing.assert_close(first["motion"], second["motion"], rtol=0, atol=0)
    torch.testing.assert_close(first["terrain_points"], second["terrain_points"], rtol=0, atol=0)
    torch.testing.assert_close(first["terrain_valid"], second["terrain_valid"], rtol=0, atol=0)


def test_terrain_cache_lru_evicts_complete_entries_by_total_tensor_bytes(tmp_path):
    dataset, arrays = make_dataset(tmp_path, frame_counts=(12, 12, 12))
    entry_bytes = sum(tensor_bytes(value) for value in arrays["shards/clip_0.npz"].values())
    dataset.cache_max_bytes = 2 * entry_bytes

    dataset[0]
    dataset[1]
    dataset[0]
    dataset[2]

    expected = ["shards/clip_0.npz", "shards/clip_2.npz"]
    assert list(dataset._motion_cache) == expected
    assert set(dataset._terrain_cache) == set(expected)
    assert dataset._cached_bytes == 2 * entry_bytes
    assert dataset._cached_bytes <= dataset.cache_max_bytes


def test_oversized_terrain_entry_is_returned_but_not_cached_or_evicted_into_cache(tmp_path):
    dataset, arrays = make_dataset(tmp_path, frame_counts=(12, 24))
    first_size = sum(tensor_bytes(value) for value in arrays["shards/clip_0.npz"].values())
    dataset.cache_max_bytes = first_size

    first = dataset[0]
    second = dataset[1]

    assert first["terrain_points"].shape == (5, *TERRAIN_PROTOCOL["patch_shape"])
    assert second["terrain_points"].shape == (5, *TERRAIN_PROTOCOL["patch_shape"])
    assert list(dataset._motion_cache) == ["shards/clip_0.npz"]
    assert set(dataset._terrain_cache) == {"shards/clip_0.npz"}
    assert dataset._cached_bytes == first_size


def test_flat_dataset_reads_and_caches_only_global_features(tmp_path, monkeypatch):
    dataset, arrays = make_dataset(tmp_path, require_terrain=False)
    motion_bytes = arrays["shards/clip_0.npz"]["global_features"].nbytes
    dataset.cache_max_bytes = motion_bytes
    accessed = []
    original_getitem = np.lib.npyio.NpzFile.__getitem__

    def tracked_getitem(archive, key):
        accessed.append(key)
        return original_getitem(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked_getitem)
    item = dataset[0]

    assert item["motion"].shape == (5, FEATURE_DIM)
    assert accessed == ["global_features"]
    assert dataset._cached_bytes == motion_bytes
    assert not dataset._terrain_cache
