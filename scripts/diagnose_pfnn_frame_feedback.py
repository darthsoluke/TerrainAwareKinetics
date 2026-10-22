"""Offline one/four-frame feedback feasibility; never changes the live protocol."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from motionbricks.data.full_corpus_inference import tensor_digest
from motionbricks.data.native_mb_online_live import online_command_goal
from scripts.diagnose_pfnn_terrain_refresh import measure, save
from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
from scripts.train_native_mb import sha256


@torch.inference_mode()
def run(args):
    args.output.mkdir(exist_ok=False)
    cases=[('idle',0.,0.,48),('forward_fast',4.,0.,72),('left_fast',4.,-math.pi/2,72)]
    save(args.output/'plan.json',dict(cases=cases,execute_frames=[1,4],actual_optimizer_updates=0,
        policy='Offline counterfactual. Predict32 with the same8tokens/4passes; execute1or4 then observe.',
        inference_timing_scope='Local wall times while the live viewer may share the GPU.',
        not_live_packet_protocol=True,quality_approved=False))
    runtime=FullCorpusAcceptanceRuntime(args.bundle,args.artifact,acceptance_test=True,
        execution='verified-cuda',spatial_index='bvh',reusable_scenes=True)
    before=tensor_digest(runtime.unified_controller.state_dict())
    runner=runtime.inference._terrain_execution
    results=[]
    try:
        for name,speed,heading,count in cases:
            for stride in (1,4):
                history=runtime.terrain_initial_history.clone()
                parts,rows=[history.clone()],[]
                failure=None
                for offset in range(0,count,stride):
                    try:
                        goal=online_command_goal(history[...,:5],
                            [speed*math.sin(heading),speed*math.cos(heading)],heading,8,runner.protocol_version)
                        debug={}
                        started=time.perf_counter()
                        motion,_=runner.predict(history,goal,debug)
                        emitted=motion[:,4:4+stride].clone()
                        history=runner.observe(torch.cat((history,emitted),1)[:,-4:].contiguous())
                        parts.append(emitted)
                        rows.append(dict(first_frame=4+offset,neural_generation=True,emitted_frames=stride,
                            wall_ms=(time.perf_counter()-started)*1000))
                    except Exception as error:
                        failure=repr(error)
                        break
                world=torch.cat(parts,1)
                torch.save(dict(world=world.cpu(),steps=rows,source='offline_generated_feedback',
                    executed_frames_per_prediction=stride),args.output/f'{name}-{stride}.pt')
                metrics=measure(runtime,world,rows) if rows else None
                row=dict(case=name,refresh_frames=stride,requested_speed_mps=speed,
                    frames=world.shape[1]-4,failure=failure,steps=rows,metrics=metrics)
                results.append(row)
                save(args.output/f'{name}-{stride}.json',row)
                print(json.dumps(dict(case=name,refresh=stride,failure=failure,
                    peak_m=metrics['body_prism_max_m'] if metrics else None,
                    distance_m=metrics['travelled_xz_m'] if metrics else None)),flush=True)
        if tensor_digest(runtime.unified_controller.state_dict())!=before:raise ValueError('Weights changed')
        save(args.output/'complete.json',dict(completed=True,rows=results,weights_unchanged=True,
            actual_optimizer_updates=0,model_state_sha256=before,script_sha256=sha256(Path(__file__)),
            scene_sha256=runtime.acceptance_receipt['scene_sha256'],quality_approved=False,
            delivered_runtime_changed=False))
    finally:runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','artifact','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.is_dir():save(args.output/'failure.json',dict(error=repr(error)))
        raise
