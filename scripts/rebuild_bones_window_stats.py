"""Versioned train-only statistics matching recentered, yaw-augmented MB windows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import build_representation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", default="stats-window-yaw-v1")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Statistics name must be one directory name")
    torch.set_num_threads(4)
    output = args.dataset / args.name
    output.mkdir(exist_ok=False)
    manifest = args.dataset / "accepted.json"
    records = json.loads(manifest.read_text())
    rep = build_representation()
    count, windows, mean, m2 = 0, 0, None, None
    for record in records:
        if record["split"] != "train":
            continue
        with np.load(args.dataset / record["shard"], allow_pickle=False) as shard:
            motion = torch.from_numpy(shard["global_features"].copy())
        starts = sorted(set(list(range(0, len(motion) - 128, 64)) + [len(motion) - 129]))
        for start in starts:
            if start < 0:
                continue
            for angle in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                window = rep.change_first_heading(motion[start:start + 129], angle, False, False)
                local = rep.dual_rep.global_to_local(window, False, False)
                data = torch.cat([window[:, :5], local[:, :4], window[:, 5:]], -1).numpy().astype(np.float64)
                n = len(data)
                clip_mean = data.mean(0)
                clip_m2 = ((data - clip_mean) ** 2).sum(0)
                if not count:
                    count, mean, m2 = n, clip_mean, clip_m2
                else:
                    delta = clip_mean - mean
                    m2 += clip_m2 + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
                windows += 1
    if count == 0:
        raise RuntimeError("No usable training windows")
    np.save(output / "mean.npy", mean.astype(np.float32))
    np.save(output / "std.npy", np.sqrt(m2 / count).astype(np.float32))
    contract = {"source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "split": "train", "window_frames": 129, "stride_frames": 64,
                "yaw_quadrature_degrees": [0, 90, 180, 270], "windows_augmented": windows,
                "frames_augmented": count, "root_xz_std_m": np.sqrt(m2 / count)[[0, 2]].tolist()}
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    report_path = args.dataset / "quality_report.json"
    report = json.loads(report_path.read_text())
    report["stats_folder"] = args.name
    report["normalization_contract"] = contract
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
