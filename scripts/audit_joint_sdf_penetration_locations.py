"""Locate exact skin penetration in unchanged recordings, including loss coverage."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    rep = build_native_representation()
    geometry = JointGeometryObjective(rep, 'cpu')
    dominant = geometry.skin.lbs_indices.gather(1, geometry.skin.lbs_weights.argmax(1)[:, None]).squeeze(1)
    names = geometry.skin.skeleton_skin.bone_order_names
    sampled = set(geometry.indices[::3].tolist())
    rows = []
    for item in args.recordings:
        label, filename = item.split('=', 1)
        path = Path(filename)
        with np.load(path, allow_pickle=False) as a:
            world = torch.from_numpy(a['world'].copy())[None]
            scene = {k:a[k].copy() for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
        vertices = geometry.skin.skin(pose['local_joint_rots'][0], world[0,:,:3,None].transpose(-1,-2).expand(-1,30,-1), rot_is_global=False)
        field = HeightfieldPrismSDF(scene)
        peaks, samples = [], []
        for frame in range(len(vertices)-1):
            points = swept_surface_points(vertices[frame:frame+2], 8)[:-1]
            gap, known = heightfield_gap(points, scene)
            inside = known & (gap < 0)
            depth = torch.zeros_like(gap)
            if inside.any():
                distance, _, certified = field.query(points[inside])
                if not certified.all():
                    raise ValueError('Uncertified inside sample')
                depth[inside] = (-distance).clamp_min(0)
            value, index = depth.flatten().max(0)
            subframe, vertex = divmod(int(index), depth.shape[1])
            peaks.append(dict(frame=frame+subframe/8, depth_m=float(value), vertex=vertex,
                dominant_joint=names[int(dominant[vertex])], in_sparse_collision_loss=vertex in sampled))
            samples.append(depth[:,geometry.indices[::3]].max())
        rows.append(dict(label=label, path=str(path), sha256=sha256(path),
            top_intervals=sorted(peaks,key=lambda x:x['depth_m'],reverse=True)[:12],
            sparse_vertices_peak_m=float(torch.stack(samples).max()), quality_approved=False))
    report = dict(rows=rows, updates=0, motion_modified=False,
        scope='8substeps per interval excluding final endpoint; exact prism relative to current scene. Sparse membership does not prove absence of indirect gradients.')
    (args.output/'complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
