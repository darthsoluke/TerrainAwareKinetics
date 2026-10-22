"""Frozen-weight playback comparison, using official SOMA FK and sole skinning."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.native_mb_live import NativeSomaLiveBackend
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from motionbricks.data.sole_region_contact import material_point_slip, sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from scripts.portable_native_mb_soma import load_bundle


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None}


def commands(index):
    if index < 12 or index >= 72:
        return [0, 0], 0.0
    if index < 36:
        return [0, 1.2], 0.0
    if index < 48:
        return [0, 2.5], 0.0
    if index < 60:
        return [-1.2, 0], -np.pi / 2
    return [0, -1.2], 0.0


@torch.inference_mode()
def motion_geometry(rep, raw, skinner, indices):
    result = rep.inverse(raw, False, return_quat=False, return_all=True)
    height, horizontal = region_geometry_from_motion(
        result["local_joint_rots"][0], raw[0, :, :3], skinner, indices)
    rotations = result["local_joint_rots"][0]
    relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    return {"height": height.cpu(), "horizontal": horizontal.cpu(),
            "joints": result["posed_joints"][0].cpu(),
            "contacts": result["foot_contacts"][0].cpu(),
            "angle_deg": torch.rad2deg(angle).cpu(),
            "root": raw[0, :, :3].cpu()}


def geometry_metrics(geometry, generated_starts, fixed_masks=None):
    joints = geometry["joints"]
    speed = torch.diff(geometry["root"][:, [0, 2]], dim=0).norm(dim=-1) * 30
    acceleration = torch.diff(joints, n=2, dim=0).norm(dim=-1) * 30 ** 2
    # Exclude the four genuine initialization frames, identically in every case.
    scored = torch.arange(len(speed)) >= 4
    seam = torch.zeros(len(speed), dtype=torch.bool)
    for start in generated_starts:
        if start > 4:
            seam[start - 1] = True
    result = {"root_speed_mps": stats(speed[scored]),
              "local_rotation_step_deg": stats(geometry["angle_deg"][scored]),
              "generation_seam_rotation_step_deg": stats(geometry["angle_deg"][seam & scored]),
              "joint_acceleration_mps2": stats(acceleration[3:]),
              "neural_generations": len(generated_starts), "feet": {}}
    masks = {}
    for foot in range(2):
        height = geometry["height"][:, foot * 98:(foot + 1) * 98]
        horizontal = geometry["horizontal"][:, foot * 98:(foot + 1) * 98]
        slip = material_point_slip(height, horizontal, 30)
        mask = slip["vertex_measured"] & scored[:, None]
        masks[foot] = mask
        own_speed = torch.diff(horizontal, dim=0).norm(dim=-1) * 30
        fixed = mask if fixed_masks is None else fixed_masks[foot]
        result["feet"][str(foot)] = {
            "own_geometric_mask_slip_mps": stats(own_speed[mask]),
            "fixed_baseline_mask_slip_mps": stats(own_speed[fixed]),
            "fixed_baseline_mask_mean_gap_m": stats(height[1:].amin(-1)[fixed.any(-1)]),
            "region_gap_m": stats(height[4:].amin(-1)),
            "penetration_m": stats(torch.relu(-height[4:].amin(-1))),
            "measured_interval_fraction": float(mask.any(-1)[scored].float().mean())}
    return result, masks


def render(geometries, parents, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    names = list(geometries)
    fig = plt.figure(figsize=(5 * len(names), 5))
    axes = [fig.add_subplot(1, len(names), row + 1, projection="3d") for row in range(len(names))]
    lines = [[axis.plot([], [], [], color="#227764", linewidth=2)[0] for _ in parents[1:]] for axis in axes]
    for axis, name in zip(axes, names):
        axis.set_title(name)
        axis.set_xlim(-1, 1); axis.set_ylim(-1, 1); axis.set_zlim(0, 2)
        axis.set_box_aspect((1, 1, 1)); axis.view_init(18, -60)
        axis.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0] * 5, color="#999999")
    frame_count = min(len(value["joints"]) for value in geometries.values())

    def update(frame):
        for name, bone_lines in zip(names, lines):
            points = geometries[name]["joints"][frame].numpy().copy()
            points[:, [0, 2]] -= geometries[name]["root"][frame, [0, 2]].numpy()
            for bone, line in enumerate(bone_lines, 1):
                segment = points[[parents[bone], bone]]
                line.set_data(segment[:, 0], segment[:, 2]); line.set_3d_properties(segment[:, 1])
        return [line for bone_lines in lines for line in bone_lines]

    animation = FuncAnimation(fig, update, frames=range(4, frame_count, 2), interval=2000 / 30, blit=False)
    animation.save(path, writer=PillowWriter(fps=15))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans", type=int, default=90)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--buffer-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 45 <= args.plans <= 90:
        raise ValueError("Use a fresh output and 45-90 plans")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cudnn.allow_tf32 = False
    infer, bundle = load_bundle(args.bundle)
    surfaces, _, _, _ = official_sole_surfaces(infer.motion_rep.skeleton)
    indices = np.concatenate([sole_region_indices(surfaces, side) for side in ("Left", "Right")])
    if len(indices) != 196:
        raise ValueError("Expected 98 official sole vertices per foot")
    skinner = move_skinner(load_official_skinner(infer.motion_rep.skeleton), infer.device)
    geometries, metrics, arrays, baseline_masks = {}, {}, {}, None
    cases = (("previous_every8", False, True, False), ("paper_motion_buffer", True, True, False),
             ("buffer_no_source_realignment", True, False, False),
             ("buffer_skip_decoder_endpoint", True, True, True))
    if args.buffer_only:
        cases = (cases[1],)
    for name, buffered, realign, skip_endpoint in cases:
        torch.manual_seed(2333)
        backend = NativeSomaLiveBackend(infer, bundle, reuse_motion_buffer=buffered)
        backend.controller._source_root_realignment = realign
        backend.controller.SKIP_ENDING_TARGET_COND = skip_endpoint
        pieces = [backend.history.clone()]
        generations, records = [], []
        with torch.inference_mode():
            for index in range(args.plans):
                velocity, heading = commands(index)
                reply = backend.predict_plan([velocity], [heading])
                motion = reply["planned_motion"]
                if backend.last_generation["neural_generation"]:
                    generations.append(4 + index * 8)
                records.append({"plan": index, "command": velocity, "heading": heading,
                                "inference_ms": reply["generation_seconds"] * 1000,
                                **backend.last_generation})
                pieces.append(motion)
                backend.history = motion[:, -4:].clone()
            played = torch.cat(pieces, dim=1)
            geometry = motion_geometry(infer.global_motion_rep, played, skinner, indices)
            report, masks = geometry_metrics(geometry, generations, baseline_masks)
            per_joint = geometry["angle_deg"][4:].amax(0)
            worst = torch.topk(per_joint, 5)
            report["worst_rotation_joints"] = [
                {"name": infer.motion_rep.skeleton.bone_order_names[int(index)], "max_step_deg": float(value)}
                for index, value in zip(worst.indices, worst.values)]
            if baseline_masks is None:
                baseline_masks = masks
            metrics[name] = {**report, "records": records}
            geometries[name] = geometry
            arrays[name + "_motion"] = played.cpu().numpy()
            arrays[name + "_joints"] = geometry["joints"].numpy()
            arrays[name + "_sole_height"] = geometry["height"].numpy()
            arrays[name + "_sole_xz"] = geometry["horizontal"].numpy()
        print(json.dumps({"case": name, **{key: value for key, value in report.items() if key != "feet"}}), flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "motions.npz", **arrays)
    report = {"bundle_sha256": file_sha256(args.bundle), "gpu": torch.cuda.get_device_name(),
              "matmul_precision": torch.get_float32_matmul_precision(),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "script_sha256": file_sha256(Path(__file__)),
              "fps": 30, "post_processing": None, "quality_approved": False,
              "fixed_mask_status": "previous generated motion geometric evidence, NOT ground truth; counts/gap reported",
              "ground_m": 0.0, "case_metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.render:
        parents = infer.motion_rep.skeleton.joint_parents.cpu().tolist()
        render(geometries, parents, args.output / "raw_playback.gif")


if __name__ == "__main__":
    main()
