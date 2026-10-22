"""Compare native and separately captured scene queries at saved factual histories."""

import argparse
import json
from pathlib import Path
import time

import torch
from motionbricks.data.soma_cuda_graph import StaticCudaCall
from motionbricks.data.soma_static_execution import trace_fixed_call

from motionbricks.data.native_mb_online_live import online_command_goal
from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
from scripts.train_native_mb import sha256


@torch.inference_mode()
def run(args):
    args.output.mkdir(exist_ok=False)
    runtime=FullCorpusAcceptanceRuntime(args.bundle,args.artifact,acceptance_test=True,
        execution='verified-cuda',spatial_index='bvh',reusable_scenes=True)
    runner=runtime.inference._terrain_execution
    world=torch.load(args.recording,weights_only=False,map_location='cpu')['world'].to(runtime.inference.device)
    histories=[world[:,offset:offset+4].contiguous() for offset in (0,8,16,32)]
    # This rejected experiment stays outside the delivered runtime.
    cases=[]
    for history in histories:
        for velocity,heading in (([0.,0.],0.),([0.,4.],0.),([-4.,0.],-1.57)):
            goal=online_command_goal(history[...,:5],velocity,heading,8,runner.protocol_version)
            cases.append(((history,goal),runner._scene_inputs(history,goal)))
    def query(history,goal):
        return {str(index):value for index,value in enumerate(runner._scene_inputs(history,goal))}
    graph=StaticCudaCall(trace_fixed_call(query,cases[0][0],cuda_constants=True))
    differences=[]
    for inputs,expected in cases:
        actual=graph(*inputs)
        for index,value in enumerate(expected):
            torch.testing.assert_close(actual[str(index)],value,atol=1e-5,rtol=1e-5)
            if value.is_floating_point():differences.append(float((actual[str(index)]-value).abs().max()))
    native_predict=runner._prediction
    def captured_predict(history,goal):
        observed=graph(history,goal)
        if not observed['11'].flatten(1).any(-1).all():raise ValueError('All scene observations are unknown')
        return runner._compiled_prediction(history,goal,*(observed[str(index)] for index in range(13)))
    rows=[]
    for index in range(24):
        history=histories[index%len(histories)]
        goal=online_command_goal(history[...,:5],[0.,4.],0.,8,runner.protocol_version)
        results={}
        timings={}
        for mode in (('native','captured') if index%2==0 else ('captured','native')):
            runner._prediction=native_predict if mode=='native' else captured_predict
            started=time.perf_counter()
            motion,count=runner.predict(history,goal,{})
            observed=runner.observe(torch.cat((history,motion[:,4:5]),1)[:,-4:].contiguous())
            torch.cuda.synchronize()
            timings[mode]=(time.perf_counter()-started)*1000
            results[mode]=(motion,count,observed,runner._compiled_prediction.output['tokens'].clone())
        left,right=results['native'],results['captured']
        for index_of_value in (1,3):torch.testing.assert_close(left[index_of_value],right[index_of_value],atol=0,rtol=0)
        torch.testing.assert_close(left[2],right[2],atol=1e-5,rtol=1e-5)
        poses=[runtime.inference.motion_rep.inverse(value[0],False,return_all=False)['posed_joints'] for value in (left,right)]
        fk=float((poses[0]-poses[1]).norm(dim=-1).max())
        if fk>3e-5:raise ValueError('Original compiled FK gate failed')
        rows.append(dict(iteration=index,history_index=index%len(histories),tokens_equal=True,fk_max_m=fk,
            motion_bitwise=torch.equal(left[0],right[0]),**timings))
    report=dict(rows=rows,scene_query_parity=dict(cases=len(cases),maximum_input_difference=max(differences)),
        scene_sha256=runtime.acceptance_receipt['scene_sha256'],recording_sha256=sha256(args.recording),
        script_sha256=sha256(Path(__file__)),execution_sha256=sha256(Path(__file__).parents[1]/
            'motionbricks/data/soma_reusable_terrain_execution.py'),actual_optimizer_updates=0,
        timing_scope='Alternating paired calls with shared local GPU, includes diagnostics/history but excludes HTTP/rendering.',
        quality_approved=False)
    (args.output/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({mode:sum(row[mode] for row in rows)/len(rows) for mode in ('native','captured')}),flush=True)
    runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','artifact','recording','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
