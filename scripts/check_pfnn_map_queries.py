"""Check spatial-query equivalence on the actual original PFNN map."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.bvh_heightfield import BVHHeightfieldPrismSDF
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.unified_scene_field import AnalyticPlaneField, field_from_heightfield
from scripts.package_pfnn_demo_map import sha


def main(args):
    scene_path = args.artifact/'scene.npz'
    with np.load(scene_path,allow_pickle=False) as archive:
        scene = {key:archive[key].copy() for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
    torch.set_num_threads(4)
    started = time.perf_counter()
    native = field_from_heightfield(scene)
    field = native if isinstance(native,AnalyticPlaneField) else BVHHeightfieldPrismSDF(scene)
    height,origin,dx = scene['heightfield_y_m'],scene['min_point_world_xyz_m'],float(scene['dx_m'])
    rng = np.random.default_rng(20261017)
    x = rng.integers(1,height.shape[1]-1,48)
    z = rng.integers(1,height.shape[0]-1,48)
    points = np.stack((origin[0]+x*dx, height[z,x]+rng.uniform(-2,3,48),origin[2]+z*dx),-1)
    points = np.concatenate((points, [[2, y, z] for y in (-1.,0.,1.,3.) for z in (-.3,0.,.3)],
        [[origin[0]-.5*dx,y,origin[2]] for y in (0.,1.)]))
    points = torch.tensor(points,dtype=torch.float32,device='cuda')
    with torch.inference_mode():
        expected = native.query(points)
        actual = field.query(points)
        errors = []
        for left,right in zip(actual,expected):
            torch.testing.assert_close(left,right,atol=0,rtol=0)
            errors.append(0. if left.dtype==torch.bool else float((left-right).abs().max()))
        hot = points.repeat(25,1)[:1536].contiguous()
        for _ in range(3):field.query(hot)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(10):field.query(hot)
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter()-begin)*100
    report = dict(passed=True,scene_sha256=sha(scene_path),
        implementation_sha256=sha(Path(__file__).parents[1]/'motionbricks/data/bvh_heightfield.py'),
        points_checked=len(points), distance_max_abs=errors[0], normal_max_abs=errors[1],valid_equal=True,
        rectangle_count=len(field.rectangles) if hasattr(field,'rectangles') else 1, query_1536_mean_ms=milliseconds,
        elapsed_seconds=time.perf_counter()-started,
        geometry='native bounded analytic plane' if isinstance(field,AnalyticPlaneField) else 'unchanged native downward/upward prisms',
        actual_optimizer_updates=0, quality_approved=False, delivery_approved=False)
    (args.artifact/'geometry_query_review.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    main(p.parse_args())
