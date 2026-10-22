"""Startup proof binding and current-scene checks for the dual policy."""
import hashlib
import json
from pathlib import Path

import torch
from motionbricks.data.dual_horizon_execution import DualHorizonExecution,SCHEMA
from motionbricks.data.native_mb_online_live import online_command_goal


ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=('motionbricks/data/dual_horizon_execution.py','motionbricks/data/root_path_terrain.py',
    'motionbricks/data/root_path_split_execution.py','motionbricks/data/soma_terrain_execution.py',
    'motionbricks/data/soma_static_execution.py','motionbricks/data/pfnn_tuning.py',
    'motionbricks/data/unified_committed_playback.py','motionbricks/data/soma_lab.py',
    'scripts/soma_pfnn_maps_server.py','scripts/dual_horizon_deployment.py')


def source_hashes():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def load_proof(path,model_sha256):
    if path is None:raise ValueError('Dual execution requires its verified proof receipt')
    proof=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (not proof['completed'] or proof['policy_schema']!=SCHEMA or proof['model_sha256']!=model_sha256
            or proof['source_hashes']!=source_hashes() or proof['torch_version']!=torch.__version__
            or proof['device']!=torch.cuda.get_device_name() or not proof['production_matches_candidate_bitwise']):
        raise ValueError('Dual execution proof does not bind this model, source and device')
    if not proof['candidate_parity'] or any(not row['tokens_equal'] or row['fk_max_m']>3e-5
            or row['skin_max_abs_m']>1e-5 or row['history_max_abs']>1e-5 for row in proof['candidate_parity']):
        raise ValueError('Dual execution proof fails the original numerical gates')
    return proof


@torch.inference_mode()
def check_scene(runner,history,scene_sha256):
    engine=DualHorizonExecution.from_runner(runner)
    if engine.scene is not runner.scene or engine.field is not runner.field:
        raise ValueError('Dual scene view is not bound to the current runner')
    query=history.new_tensor([[-1.,.6,1.,.4]])
    goal=online_command_goal(history[...,:5],[0.,3.],0.,8,engine.protocol_version)
    prepared=engine.prepare_dual(history,goal,query,16)
    result=engine.compiled_suffix(history,goal,query,prepared)
    if not torch.isfinite(result['motion'][:,:32]).all() or not (result['count']==8).all():
        raise ValueError('Current-scene dual execution is invalid')
    geometry=engine.geometry(result['motion'][:,4:12].contiguous())
    vertices=engine.skinner.lbs(geometry['rig'])
    if vertices.shape[-2]!=18056 or not torch.isfinite(vertices).all():
        raise ValueError('Current-scene official geometry is invalid')
    return dict(passed=True,scene_sha256=scene_sha256,execute_tokens=8,lookahead_tokens=16,
        current_field_bound=engine.field is runner.field,official_vertices=18056,
        scope='current scene input/count/geometry; numerical parity is bound by the source proof')
