"""Measure unmodified publication seams separately from browser playback timing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation


def summarize(values, incoming):
    values = np.asarray(values)
    boundary = incoming % 8 == 0
    order = np.argsort(values)[-8:][::-1]
    return dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)), maximum=float(values.max()),
        boundary_mean=float(values[boundary].mean()) if boundary.any() else None,
        interior_mean=float(values[~boundary].mean()),
        peaks=[dict(frame=int(incoming[i]), value=float(values[i]), boundary=bool(boundary[i])) for i in order])


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    rep = build_native_representation()
    reports = {}
    for spec in args.recordings:
        label, filename = spec.split('=', 1)
        with np.load(filename, allow_pickle=False) as archive:
            world = torch.from_numpy(archive['world'].copy())[None]
        pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
        joints = pose['posed_joints'][0]
        relative = joints - world[0, :, None, :3]
        roots = world[0, :, :3]
        rotation = pose['local_joint_rots'][0]
        delta = rotation[:-1].transpose(-1, -2) @ rotation[1:]
        angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)) * (180 / torch.pi)
        incoming = np.arange(1, len(roots))
        velocity = torch.diff(roots, dim=0) * 30
        relative_velocity = torch.diff(relative, dim=0) * 30
        reports[label] = dict(file=filename, frames=len(roots), original_fps=30,
            root_step_cm=summarize(torch.diff(roots, dim=0).norm(dim=-1).numpy() * 100, incoming),
            root_velocity_change_mps=summarize(torch.diff(velocity, dim=0).norm(dim=-1).numpy(), incoming[1:]),
            body_step_cm=summarize(torch.diff(relative, dim=0).norm(dim=-1).mean(-1).numpy() * 100, incoming),
            body_velocity_change_mps=summarize(torch.diff(relative_velocity, dim=0).norm(dim=-1).mean(-1).numpy(), incoming[1:]),
            max_joint_angle_step_deg=summarize(angle.amax(-1).numpy(), incoming),
            worst_bones=[dict(frame=int(i + 1), bone=rep.skeleton.bone_order_names[int(angle[i].argmax())],
                              angle_deg=float(angle[i].max())) for i in torch.argsort(angle.amax(-1), descending=True)[:8]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(recordings=reports, motion_modified=False,
        note='Frame indices are zero-based published arrays. Source multiples of8 are a comparison partition, not model seams.'), indent=2))
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
