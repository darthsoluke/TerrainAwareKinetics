"""Headless official-SOMA surface comparison, without retargeting or motion correction."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, require_same_snapshot
from motionbricks.data.bones_seed import build_representation
from motionbricks.data.soma_skin_reference import load_official_skinner, skinned_vertices
from scripts.evaluate_sole_region_ab import WINDOWS, build_model, reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--behavior", choices=[b for b, _ in WINDOWS], default="walk")
    parser.add_argument("--stride", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--filename", help="Optional held-out failure clip instead of the showcase prefix")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.dataset / "quality_report.json").read_text())
    rep = build_representation(args.dataset / report.get("stats_folder", "stats"))
    snapshot = dataset_snapshot(args.dataset)
    filename = args.filename or dict(WINDOWS)[args.behavior]
    records = {r["filename"]: r for r in json.loads((args.dataset / "accepted.json").read_text())}
    if records[filename]["split"] != "validation":
        raise ValueError("The showcase must remain held out")
    with np.load(args.dataset / records[filename]["shard"], allow_pickle=False) as data:
        if args.start < 0 or args.start + 129 > len(data["global_features"]):
            raise ValueError("Rendered window exceeds the held-out shard")
        features = torch.from_numpy(data["global_features"][args.start:args.start + 129].copy())
    arms = {}
    with torch.inference_mode():
        global_motion = rep.normalize(features)[None]
        local = rep.dual_rep.global_to_local(global_motion, True, True, lengths=torch.tensor([129]))[:, :128]
        arms["source"] = rep.dual_rep.local_motion_rep.inverse(local, True, return_all=True)
        for entry in args.runs.split(","):
            name, path = entry.split("=", 1)
            run = Path(path)
            require_same_snapshot(json.loads((run / "run_contract.json").read_text())["dataset_snapshot"], snapshot)
            model, network = build_model(run, rep)
            arms[name] = reconstruct(model, network, rep, features)[0]
        skinner = load_official_skinner(rep.skeleton)
        meshes = {name: skinned_vertices(skinner, pose["local_joint_rots"][0], pose["posed_joints"][0]).numpy()
                  for name, pose in arms.items()}
    faces = skinner.faces.numpy()
    source_root = arms["source"]["posed_joints"][0, :, rep.skeleton.root_idx].numpy()
    colors = {"source": "#888888", "Initial": "#43895b", "Point": "#328ac2", "Region": "#dd8d35"}
    light = np.array([0.5, -0.6, 1.0])
    light /= np.linalg.norm(light)
    figure = plt.figure(figsize=(4.0 * len(arms), 4.8))
    artists = []
    for column, name in enumerate(arms):
        axis = figure.add_subplot(1, len(arms), column + 1, projection="3d")
        axis.set(xlim=(-0.85, 0.85), ylim=(-0.85, 0.85), zlim=(-0.05, 2.0))
        axis.set_box_aspect((1, 1, 1.25))
        axis.view_init(elev=12, azim=-50)
        axis.set_axis_off()
        for tick in np.arange(-0.75, 1, 0.25):
            axis.plot([-0.85, 0.85], [tick, tick], [0, 0], color="#dddddd", linewidth=0.5)
            axis.plot([tick, tick], [-0.85, 0.85], [0, 0], color="#dddddd", linewidth=0.5)
        mesh = Poly3DCollection([], linewidth=0, edgecolors="none")
        axis.add_collection3d(mesh)
        title = axis.set_title(name, fontsize=11)
        artists.append((name, mesh, title))
    figure.suptitle(f"SOMA | {args.behavior} | recorded source / offline reconstruction | no IK or foot locking", fontsize=11)
    figure.tight_layout()

    def update(frame):
        origin = source_root[frame].copy()
        origin[1] = 0
        for name, mesh, title in artists:
            points = meshes[name][frame] - origin
            triangles = points[:, [0, 2, 1]][faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-10)
            intensity = 0.45 + 0.55 * np.maximum(normals @ light, 0)
            base = np.array(matplotlib.colors.to_rgb(colors.get(name, "#328ac2")))
            mesh.set_verts(triangles)
            mesh.set_facecolor(intensity[:, None] * base)
            title.set_text(f"{name} | {frame / 60:.2f} s")
        return [mesh for _, mesh, _ in artists]

    update(32)
    stem = args.behavior + ("_worst_skinned_" if args.filename else "_skinned_") + "_".join(arms)
    figure.savefig(args.output / f"{stem}.png", dpi=120)
    animation = FuncAnimation(figure, update, frames=range(0, 128, args.stride), interval=1000 * args.stride / 60)
    animation.save(args.output / f"{stem}.mp4", writer=FFMpegWriter(
        fps=60 / args.stride, codec="libx264", extra_args=["-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(figure)
    (args.output / f"{stem}.json").write_text(json.dumps({
        "behavior": args.behavior, "source": filename, "mode": "offline tokenizer reconstruction",
        "skin": "official SOMASkin", "post_processing": None,
        "camera": "All panels follow the same source-root position; meshes are not aligned independently",
        "source_fps": 60, "render_stride": args.stride, "playback_fps": 60 / args.stride,
        "start_frame": args.start,
        "dataset_snapshot_sha256": snapshot["sha256"], "runs": args.runs,
    }, indent=2))
    print(str(args.output / f"{stem}.mp4"), flush=True)


if __name__ == "__main__":
    main()
