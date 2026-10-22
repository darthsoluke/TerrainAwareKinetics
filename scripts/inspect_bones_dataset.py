"""Render held-out source motion and sole/contact traces for pipeline review."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from motionbricks.data.bones_seed import build_representation, official_sole_offsets, sole_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    records = json.loads((args.dataset / "accepted.json").read_text())
    rep = build_representation()
    skeleton = rep.skeleton
    offsets, _ = official_sole_offsets(skeleton)
    categories = sorted({record["behavior"] for record in records})
    figure = plt.figure(figsize=(15, 3 * len(categories)))
    reviewed = []
    for row, category in enumerate(categories):
        record = next((record for record in records if record["behavior"] == category and record["split"] == "validation"),
                      next(record for record in records if record["behavior"] == category))
        with np.load(args.dataset / record["shard"], allow_pickle=False) as shard:
            features = torch.from_numpy(shard["global_features"].copy())
            confidence = shard["contact_confidence"].copy()
        with torch.no_grad():
            output = rep.inverse(features, False, return_all=True)
            sole = sole_positions(output["posed_joints"], output["global_joint_rots"], skeleton, offsets)
        posed = output["posed_joints"].numpy()
        frames = np.linspace(0, min(len(posed) - 1, 180), 4).astype(int)
        for column, frame in enumerate(frames):
            axis = figure.add_subplot(len(categories), 5, row * 5 + column + 1, projection="3d")
            points = posed[frame].copy()
            points[:, [0, 2]] -= points[0, [0, 2]]
            for joint, parent in enumerate(skeleton.joint_parents.tolist()):
                if parent < 0:
                    continue
                color = "#2171b5" if "Left" in skeleton.bone_order_names[joint] else "#c0392b"
                axis.plot(points[[parent, joint], 0], points[[parent, joint], 2], points[[parent, joint], 1], color=color)
            axis.plot([-0.7, 0.7], [0, 0], [0, 0], color="gray", alpha=0.5)
            axis.set(xlim=(-0.75, 0.75), ylim=(-0.75, 0.75), zlim=(0, 2.05))
            axis.set_box_aspect((1, 1, 1.4))
            axis.view_init(elev=12, azim=-45)
            axis.set_axis_off()
            axis.set_title(f"{category} | {frame / 60:.2f}s", fontsize=10)
        axis = figure.add_subplot(len(categories), 5, row * 5 + 5)
        colors = ("#2171b5", "#63a6d8", "#c0392b", "#ee856e")
        time = np.arange(len(sole)) / 60
        for foot, color in enumerate(colors):
            height = sole[:, foot, 1].numpy()
            axis.plot(time, height, color=color, linewidth=0.8, label=skeleton.foot_joint_names[foot])
            contact = output["foot_contacts"][:, foot].numpy() > 0.5
            axis.scatter(time[contact], height[contact], color=color, s=2)
        axis.axhline(0, color="black", linewidth=0.5)
        axis.set(xlabel="Time (s)", ylabel="Sole height (m)", title=f"{record['split']} | contacts marked")
        axis.tick_params(labelsize=8)
        if row == 0:
            axis.legend(fontsize=6)
        reviewed.append({"filename": record["filename"], "split": record["split"], "behavior": category,
                         "confidence_fraction": float(confidence.mean())})
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=130)
    args.output.with_suffix(".json").write_text(json.dumps({
        "examples": reviewed, "status": "review examples; not manual frame-level contact ground truth"
    }, indent=2))
    plt.close(figure)


if __name__ == "__main__":
    main()
