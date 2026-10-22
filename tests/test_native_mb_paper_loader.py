import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from motionbricks.data.native_mb_data import (
    NativeMBMotionDataset,
    native_training_loader,
)


FEATURE_DIM = 366
DUAL_STATS_DIM = 370


def make_native_dataset(root, *, train_records=3, frames=80):
    (root / "stats").mkdir(parents=True)
    (root / "shards").mkdir()
    np.save(root / "stats" / "mean.npy", np.full(DUAL_STATS_DIM, 10., dtype=np.float32))
    np.save(root / "stats" / "std.npy", np.full(DUAL_STATS_DIM, 2., dtype=np.float32))
    records = []
    raw_by_shard = {}
    specs = [(f"train_{index}", "train") for index in range(train_records)]
    specs.extend([("validation_heldout", "validation"), ("test_heldout", "test")])
    for index, (name, split) in enumerate(specs):
        shard = f"shards/{name}.npz"
        features = np.full((frames, FEATURE_DIM), index + 20., dtype=np.float32)
        features[:, 0] = index * 1000. + np.arange(frames, dtype=np.float32)
        np.savez_compressed(root / shard, global_features=features)
        raw_by_shard[shard] = features
        records.append({
            "filename": name,
            "recording_group": name,
            "behavior": "walk",
            "split": split,
            "shard": shard,
            "frames_30hz": frames,
        })
    (root / "accepted.json").write_text(json.dumps(records), encoding="utf-8")
    return raw_by_shard


def test_paper_loader_yields_full_256_row_batch_with_two_files(tmp_path):
    make_native_dataset(tmp_path, train_records=2)
    dataset = NativeMBMotionDataset(tmp_path, "train", max_frames=65, min_frames=65)

    loader = native_training_loader(dataset, batch_size=256, workers=0, paper=True)
    batch = next(iter(loader))

    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.replacement is True
    assert loader.sampler.num_samples == 256 * 256
    assert len(loader) == 256
    assert batch["batch_size"] == 256
    assert batch["motion"].shape == (256, 65, FEATURE_DIM)
    assert batch["motion_len"].tolist() == [65] * 256
    assert batch["motion_pad_mask"].all()


def test_repeated_same_record_reads_draw_independent_crops(tmp_path, monkeypatch):
    make_native_dataset(tmp_path, train_records=1)
    dataset = NativeMBMotionDataset(tmp_path, "train", max_frames=65, min_frames=65)
    starts = iter((0, 15))
    monkeypatch.setattr(torch, "randint", lambda _high, _size: torch.tensor(next(starts)))

    first = dataset[0]
    second = dataset[0]

    assert first["keyid"] == second["keyid"] == 0
    assert first["motion"].shape == second["motion"].shape == (65, FEATURE_DIM)
    assert first["motion"][0, 0].item() != second["motion"][0, 0].item()
    assert second["motion"][0, 0].item() - first["motion"][0, 0].item() == pytest.approx(
        15. / np.sqrt(2. ** 2 + dataset.stats.eps)
    )


def test_normalization_and_returned_mutation_do_not_change_cached_raw_features(tmp_path):
    raw_by_shard = make_native_dataset(tmp_path, train_records=1)
    dataset = NativeMBMotionDataset(
        tmp_path, "train", max_frames=65, min_frames=65,
        cache_max_bytes=raw_by_shard["shards/train_0.npz"].nbytes,
    )
    raw = torch.from_numpy(raw_by_shard["shards/train_0.npz"].copy())

    normalized = dataset[0]["motion"]
    cached = dataset._motion_cache["shards/train_0.npz"]
    torch.testing.assert_close(cached, raw)
    assert not torch.equal(normalized, raw[:65])
    normalized.add_(1000.)
    torch.testing.assert_close(cached, raw)

    dataset.max_frames = len(raw)
    restored = dataset[0]["motion"]
    expected = dataset.stats.normalize(raw)
    torch.testing.assert_close(restored, expected)
    torch.testing.assert_close(cached, raw)


def test_raw_motion_cache_obeys_byte_limit_and_least_recently_used_order(tmp_path):
    raw_by_shard = make_native_dataset(tmp_path, train_records=3)
    shard_size = raw_by_shard["shards/train_0.npz"].nbytes
    dataset = NativeMBMotionDataset(
        tmp_path, "train", max_frames=65, min_frames=65,
        cache_max_bytes=2 * shard_size,
    )

    dataset[0]
    dataset[1]
    dataset[0]
    dataset[2]

    assert list(dataset._motion_cache) == ["shards/train_0.npz", "shards/train_2.npz"]
    assert dataset._cached_bytes == 2 * shard_size
    assert dataset._cached_bytes <= dataset.cache_max_bytes


def test_training_dataset_excludes_validation_and_test_records(tmp_path):
    make_native_dataset(tmp_path, train_records=2)

    dataset = NativeMBMotionDataset(tmp_path, "train", max_frames=65, min_frames=65)

    assert len(dataset) == 2
    assert [record["split"] for record in dataset.records] == ["train", "train"]
    assert all("heldout" not in record["filename"] for record in dataset.records)


def test_nonpaper_loader_retains_release_sampler_and_drop_last_behavior(tmp_path):
    make_native_dataset(tmp_path, train_records=3)
    dataset = NativeMBMotionDataset(tmp_path, "train", max_frames=65, min_frames=65)

    loader = native_training_loader(dataset, batch_size=2, workers=0)

    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.replacement is False
    assert loader.sampler.num_samples == len(dataset) == 3
    assert loader.drop_last is False
    assert len(loader) == 2
