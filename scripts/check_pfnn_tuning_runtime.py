"""Verify experimental replanning packets against independent serial predictions."""

import argparse
import json
from pathlib import Path
import time

import torch

from motionbricks.data.full_corpus_inference import tensor_digest
from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.pfnn_tuning import TunablePFNNBackend,PROFILE
from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
from scripts.train_native_mb import sha256


@torch.inference_mode()
def run(args):
    args.output.mkdir(exist_ok=False)
    runtime=FullCorpusAcceptanceRuntime(args.bundle,args.artifact,acceptance_test=True,
        execution='verified-cuda',spatial_index='bvh',reusable_scenes=True)
    before=tensor_digest(runtime.unified_controller.state_dict())
    runner=runtime.inference._terrain_execution
    rows=[]
    try:
        for refresh in (1,2,4,8,16,24):
            backend=TunablePFNNBackend(runtime.unified_controller,runtime.terrain_initial_history.clone(),runtime.terrain_scene)
            backend.configure_runtime(dict(PROFILE,refresh_frames=refresh))
            history=backend.history.clone()
            started=time.perf_counter()
            result=backend.predict_plan([0.,1.6],0.)
            elapsed=time.perf_counter()-started
            torch.testing.assert_close(backend.history,history,atol=0,rtol=0)
            parts=[]
            step=min(8,refresh)
            for _ in range(8//step):
                goal=online_command_goal(history[...,:5],[0.,1.6],0.,8,runner.protocol_version)
                motion,_=runner.predict(history,goal,{})
                emitted=motion[:,4:4+step]
                parts.append(emitted)
                history=runner.observe(torch.cat((history,emitted),1)[:,-4:].contiguous())
            expected=torch.cat(parts,1)
            torch.testing.assert_close(result['planned_motion'],expected,atol=0,rtol=0)
            canonical=runner.geometry(expected)['history']
            torch.testing.assert_close(result['published_history'],canonical,atol=0,rtol=0)
            torch.testing.assert_close(backend._last_published_history,canonical,atol=0,rtol=0)
            saved_history=backend.history.clone()
            candidate=backend.prepare_next_plan([0.,1.6],0.,saved_history,result['published_history'])
            torch.testing.assert_close(backend.history,saved_history,atol=0,rtol=0)
            serial=TunablePFNNBackend(runtime.unified_controller,runtime.terrain_initial_history.clone(),runtime.terrain_scene)
            serial.configure_runtime(dict(PROFILE,refresh_frames=refresh))
            previous=serial.predict_plan([0.,1.6],0.)
            serial.history=previous['published_history'].clone()
            following=serial.predict_plan([0.,1.6],0.)
            torch.testing.assert_close(candidate['planned_motion'],following['planned_motion'],atol=0,rtol=0)
            backend.history=result['published_history'].clone()
            backend.commit_prepared_plan(candidate)
            torch.testing.assert_close(backend._future_buffer,serial._future_buffer,atol=0,rtol=0)
            rows.append(dict(refresh_frames=refresh,packet_frames=8,seconds=elapsed,
                serial_equal=True,prefetch_equal=True,history_unchanged_until_commit=True))
            print(json.dumps(rows[-1]),flush=True)
        for changes in (dict(num_tokens=12),dict(inference_steps=2),
                        dict(lookahead_m=2.8,lateral_halfwidth_m=.8),dict(refresh_frames=4,inference_steps=2)):
            backend=TunablePFNNBackend(runtime.unified_controller,runtime.terrain_initial_history.clone(),runtime.terrain_scene)
            backend.configure_runtime(dict(PROFILE,**changes))
            result=backend.predict_plan([0.,1.6],0.)
            if result['planned_motion'].shape!=(1,8,runner.history_shape[-1]) or not torch.isfinite(result['planned_motion']).all():
                raise ValueError('Tuned native packet invalid')
            rows.append(dict(native_settings=changes,packet_frames=8,finite=True,
                applied=backend.last_generation['runtime_controls']))
            print(json.dumps(rows[-1]),flush=True)
        if tensor_digest(runtime.unified_controller.state_dict())!=before:raise ValueError('Weights changed')
        (args.output/'complete.json').write_text(json.dumps(dict(passed=True,rows=rows,weights_unchanged=True,
            actual_optimizer_updates=0,model_state_sha256=before,scene_sha256=runtime.acceptance_receipt['scene_sha256'],
            script_sha256=sha256(Path(__file__)),quality_approved=False,
            scope='All-frame serial/prefetch equivalence and native setting execution. Not a motion-quality approval.'),indent=2))
    finally:runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','artifact','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.is_dir():(args.output/'failure.json').write_text(json.dumps(dict(error=repr(error))))
        raise
