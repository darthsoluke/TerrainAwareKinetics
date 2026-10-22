"""Inspect fresh predictions at saved histories; no rollout, sampling or training."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.full_corpus_inference import tensor_digest
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainBackend, PROFILE
from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
from scripts.train_native_mb import sha256


@torch.inference_mode()
def run(args):
    args.output.mkdir(exist_ok=False)
    runtime=FullCorpusAcceptanceRuntime(args.bundle,args.artifact,acceptance_test=True,
        execution='eager',spatial_index='bvh')
    before=tensor_digest(runtime.unified_controller.state_dict())
    report=json.loads((args.recordings/'complete.json').read_text())
    if runtime.acceptance_receipt['scene_sha256']!=report['scene_sha256']:
        raise ValueError('Saved histories belong to a different map')
    rows=[]
    try:
        for case in ('forward_walk','forward_fast'):
            saved=torch.load(args.recordings/f'{case}-24.pt',weights_only=False,map_location='cpu')
            speed=1.6 if case=='forward_walk' else 4.
            for offset in (0,8,16):
                history=saved['world'][:,offset:offset+4].to(runtime.inference.device)
                backend=UnifiedTerrainBackend(runtime.unified_controller,history,runtime.terrain_scene)
                backend.configure_runtime(dict(PROFILE))
                result=backend.predict_plan([0.,speed],0.)
                debug=result['debug']
                root=torch.tensor(debug['root_stage_world_xyz_m'],device=history.device)
                decoded=torch.tensor(debug['decoded_world_xyz_m'],device=history.device)
                root_gap,root_valid=heightfield_gap(root,runtime.terrain_scene)
                decoded_gap,decoded_valid=heightfield_gap(decoded,runtime.terrain_scene)
                observation=runtime.unified_controller.observation
                volume=runtime.unified_controller.volume_observation
                current_gap,current_valid=heightfield_gap(history[0,-1:,:3],runtime.terrain_scene)
                row=dict(case=case,history_offset=offset,actual_optimizer_updates=0,
                    input_root=history[0,-1,:3].cpu().tolist(),
                    source_history_sha256=sha256(args.recordings/f'{case}-24.pt'),
                    current_root_clearance_m=float(current_gap[0]),
                    observed_height_relative_to_root_range_m=[float(observation['points'][...,1][observation['valid']].min()),
                        float(observation['points'][...,1][observation['valid']].max())],
                    volume_known_fraction=float(volume['valid'].float().mean()),
                    root_prediction_clearance_m=root_gap.cpu().tolist(),decoded_root_clearance_m=decoded_gap.cpu().tolist(),
                    root_y=root[:,1].cpu().tolist(),decoded_y=decoded[:,1].cpu().tolist(),
                    query_known=bool(root_valid.all() & decoded_valid.all() & current_valid.all()),
                    root_vs_decoded_y_max_m=float((root[:,1]-decoded[:,1]).abs().max()))
                rows.append(row)
                print(json.dumps(dict(case=case,offset=offset,root_min=float(root_gap.min()),
                    decoded_min=float(decoded_gap.min()),known=row['query_known'])),flush=True)
        if tensor_digest(runtime.unified_controller.state_dict())!=before:raise ValueError('Weights changed')
        (args.output/'complete.json').write_text(json.dumps(dict(rows=rows,weights_unchanged=True,
            model_state_sha256=before,scene_sha256=report['scene_sha256'],
            script_sha256=sha256(Path(__file__)),actual_optimizer_updates=0,
            scope='Fresh native predictions from saved generated histories; offline map queries are diagnostic only.'),indent=2))
    finally:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','artifact','recordings','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
