"""Create a large authored trial world; no new motion or training data generation."""

import argparse
import json
from pathlib import Path

import numpy as np

from motionbricks.data.terrain_test_world import build_world,merged_surface
from scripts.train_native_mb import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new authored scene artifact')
    with np.load(args.seed_scene,allow_pickle=False) as source:
        history=source['initial_history'].copy()
        metadata=json.loads(str(source['metadata_json']))
    hf,origin,spawns=build_world()
    world_meta={'schema':'motionbricks.authored_test_scene.v1',
        'seed_dataset_manifest_sha256':metadata['dataset_manifest_sha256'],
        'seed_scene_sha256':sha256(args.seed_scene),'seed_motion_source':metadata.get('window'),
        'history_frames':4,'future_motion_included':False,'terrain_source':'procedurally authored test arena',
        'training_data':False,'size_m':[256,256],'cell_spacing_m':.2,
        'geometry_scope':'single-valued heightfield; no overhangs','spawns':spawns}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,heightfield_y_m=hf,min_point_world_xyz_m=origin,
        dx_m=np.float32(.2),initial_history=history,metadata_json=np.asarray(json.dumps(world_meta)))
    world_meta['scene_sha256']=sha256(args.output)
    geometry=merged_surface(hf,origin,.2)
    mesh_path=args.output.with_suffix('.mesh.json')
    mesh_path.write_text(json.dumps(geometry,separators=(',',':'),allow_nan=False))
    world_meta.update(mesh_file=mesh_path.name,mesh_sha256=sha256(mesh_path),
        height_min_m=float(hf.min()),height_max_m=float(hf.max()),
        terrain_cells=int(hf.size),render_triangles=len(geometry['faces'])//3)
    args.output.with_suffix('.json').write_text(json.dumps(world_meta,indent=2))
    print(json.dumps({k:v for k,v in world_meta.items() if k!='spawns'},indent=2))


if __name__=='__main__':main()
