"""Package an explicitly configured continuous-reference interactive prototype."""

import argparse
import json
from pathlib import Path
import shutil

from motionbricks.mb_sonic.planner_process import source_identity
from motionbricks.mb_sonic.reference import file_sha256
from scripts.isolate_mb_continuous_reference import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--consumer-horizon',type=float,default=.6)
    parser.add_argument('--scene-manifest',type=Path)
    parser.add_argument('--initial-scene')
    parser.add_argument('--limb-rotation-source',choices=('markers','native_material_axes'),default='native_material_axes')
    root_mode=parser.add_mutually_exclusive_group()
    root_mode.add_argument('--root-reference-acceleration',type=float)
    root_mode.add_argument('--learned-root',action='store_true')
    parser.add_argument('--tick-log-pending',type=int,default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config = json.loads(args.base.read_text())
    config.update(mb_conditioning_mode='B1',reference_stream_policy='continuous_native_AR_v3',
        mb_replan_interval_s=.02,mb_adopt_interval_s=0.,consumer_horizon_s=args.consumer_horizon,
        inference_rng_seed=20261020,retarget_limb_rotation_source=args.limb_rotation_source,
        recording_dir=str((args.output/'recording').resolve()))
    config['reference_constraints']['pose_task_frame'] = 'root'
    config['logging']['tick_pending_records']=args.tick_log_pending
    log=config['logging']
    reserve=((config['raw_log_pending_plans']+log['tick_pending_records']+4)
             *(log['max_numeric_record_bytes']+log['max_json_record_bytes'])+4*log['max_json_record_bytes'])
    log['required_log_reserve_bytes']=max(log['required_log_reserve_bytes'],reserve)
    if args.root_reference_acceleration is not None:
        config['root_reference_acceleration_mps2']=args.root_reference_acceleration
    if args.learned_root:
        config.pop('root_reference_acceleration_mps2',None)
    if args.scene_manifest is not None:
        manifest_path = args.scene_manifest.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        scene_id = manifest['scene_id']
        seed = manifest_path.parent/manifest['source']['scene']['file']
        config.update(scene_root=str(manifest_path.parent.parent),initial_scene_id=scene_id,
            enabled_scene_ids=[scene_id],scene_manifest_sha256={scene_id:file_sha256(manifest_path)},
            scene_seeds={scene_id:dict(path=str(seed),sha256=file_sha256(seed))},
            scene_query_backend='native',scene_query_proofs={})
        config.pop('scene_catalog',None)
    if args.initial_scene is not None:
        if args.initial_scene not in config['enabled_scene_ids']:
            raise ValueError('Initial scene must be one of the enabled maps')
        config['initial_scene_id']=args.initial_scene
    for key in ('state_feedback_segments','state_feedback_stop_segments','state_feedback_robot_prior'):
        config.pop(key,None)
    save_json(args.output/'config.json',config)
    sources = source_identity()
    root = Path(__file__).resolve().parents[1]
    (args.output/'source').mkdir()
    for name in sources:
        shutil.copy2(root/'motionbricks/mb_sonic'/name,args.output/'source'/name)
    server_source=root/'scripts/mb_sonic_lab_server.py'
    shutil.copy2(server_source,args.output/'source'/server_source.name)
    save_json(args.output/'package.json',dict(schema='motionbricks.continuous_stream_prototype.v1',
        config_sha256=file_sha256(args.output/'config.json'),source_sha256=sources,
        server_source_sha256=file_sha256(server_source),
        preserved_maps=config['enabled_scene_ids'],optimizer_updates=0,
        physical_state_writes=False,pose_filter=False,actuator_filter=False,
        forecast_history='continuous_MB_handoff_context',feedback='actual_Root_scene_query_anchor',
        untrained_observation='G1_body_not_relabelled_as_SOMA_ground_truth',
        raw_reference_retained=True,automatic_quality_approval=False))
    print(args.output/'config.json')


if __name__ == '__main__':
    main()
