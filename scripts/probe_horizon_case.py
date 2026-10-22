"""Isolate the first005 idle observation discrepancy without a route run."""
import argparse
import json
from pathlib import Path

import torch
from motionbricks.data.root_path_split_execution import SplitRootPathExecution,VOLUME_KEYS
from motionbricks.data.root_path_terrain import RootPathTerrainExecution
from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.local_terrain_observation import LocalTerrainObservation
from motionbricks.data.soma_static_execution import restore_native_transformers,use_dense_transformers
from scripts.soma_pfnn_maps_server import PFNNMapRuntime


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    runtime=PFNNMapRuntime(a.bundle,a.catalog,execution='eager',initial_map='005')
    runner=object.__new__(SplitRootPathExecution);runner.graphs={}
    RootPathTerrainExecution.__init__(runner,runtime.inference,runtime.terrain_scene,runtime.skinner,
        runtime.terrain_initial_history,controller=runtime.unified_controller)
    history=runtime.terrain_initial_history
    query=history.new_tensor([[-1.,.6,1.,.4]])
    tokens=a.tokens
    goal=online_command_goal(history[...,:5],[0.,0.],0.,tokens,runner.protocol_version)
    saved={}
    for label,dense in (('native',False),('dense_eager',True)):
        (use_dense_transformers if dense else restore_native_transformers)(runtime.inference)
        with runtime.unified_controller.context(history,runtime.terrain_scene):
            result=RootPathTerrainExecution._prediction(runner,history,goal,num_tokens=tokens,query_controls=query)
        saved[label]={key:value.clone() for key,value in result.items()}
        saved[label]['path']=runner.observer.path.clone()
    runner._build_graphs(tokens,history,goal,query)
    actual=runner._prediction(history,goal,torch.cat((history.new_tensor([[tokens]]),query),-1))
    saved['shared_graph']={key:value.clone() for key,value in actual.items()}
    saved['shared_graph']['path']=runner.observer.path.clone()
    coarse=runner._scene_inputs(history,goal,tokens,query)
    _,same_root_height,same_root_path=runner._path_inputs(history,saved['native']['root'],coarse,tokens)
    report=dict(tokens=tokens,command=[0.,0.],query=query.cpu().tolist(),map='005',comparisons={})
    for label in ('dense_eager','shared_graph'):
        report['comparisons'][label]={key:float((saved[label][key]-saved['native'][key]).abs().max())
            for key in ('root','points','world_query_xz','sdf_distance','path')}
    report['graph_vs_dense_root_max_abs']=float((actual['root']-saved['dense_eager']['root']).abs().max())
    report['identical_native_root_points_max_abs']=float((same_root_height.points-saved['native']['points']).abs().max())
    path=saved['native']['path'];segments=path[:,1:]-path[:,:-1]
    report['native_path']=dict(total_length=float(segments.norm(dim=-1).sum()),last_segment_length=float(segments[:,-1].norm()),
        endpoint_displacement=(path[:,-1]-path[:,0]).cpu().tolist(),points=path.cpu().tolist())
    report['shared_storage']=runner.constant_storage_report()
    torch.save({k:{name:value.cpu() for name,value in item.items()} for k,item in saved.items()},a.output/'tensors.pt')
    (a.output/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True);runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=8);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
