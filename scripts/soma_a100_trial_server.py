"""One verified A100 checkpoint in the existing local TPS viewer, without legacy models."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from scripts.soma_lab_server import (ROOT, SomaLabRuntime, create_app,
                                     create_native_terrain_backend, load_terrain_live_bundle)


class A100TrialRuntime(SomaLabRuntime):
    default_mode = 'terrain'

    def __init__(self,bundle_path,expected_hash,device,scene_path,label,execution,interactive=False,joint_checkpoint=None,committed_prefetch=False,start_paused=False):
        if file_sha256(bundle_path) != expected_hash:
            raise ValueError('A100 checkpoint hash differs from export receipt')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        inference, scene, history = load_terrain_live_bundle(bundle_path, device, scene_path)
        backend_builder=create_native_terrain_backend
        self.joint_controller=None
        if joint_checkpoint is not None:
            from motionbricks.data.terrain_model_v3 import JointTerrainController,SCHEMA
            from motionbricks.data.joint_sdf_controller import JointSDFController,SDF_VERSION
            from motionbricks.data.terrain_control import CONTROL_VERSION
            from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
            checkpoint=torch.load(joint_checkpoint,map_location='cpu',mmap=True,weights_only=False)
            if (checkpoint.get('schema') not in (SCHEMA,SDF_VERSION) or checkpoint['contract']['source_bundle_sha256']!=expected_hash
                    or checkpoint['contract'].get('control_version')!=CONTROL_VERSION):
                raise ValueError('Joint checkpoint source or online-control contract differs')
            controller=JointSDFController if checkpoint['schema']==SDF_VERSION else JointTerrainController
            model=controller(inference,torch.load(bundle_path,map_location='cpu',weights_only=False))
            from motionbricks.data.locomotion_command_adapter import install_checkpoint_command
            install_checkpoint_command(model,checkpoint)
            model.load_state_dict(checkpoint['model_state'],strict=True);model.requires_grad_(False);model.eval()
            self.joint_controller=model
            def backend_builder(_inference,seed,terrain):
                backend=JointTerrainOnlineBackend(model,seed,terrain)
                backend.committed_history_prefetch=committed_prefetch
                return backend
        self.inference = inference
        self.skinner = move_skinner(load_official_skinner(inference.motion_rep.skeleton), torch.device(device))
        self.skinner.skeleton_skin.float().to(device)
        self.faces = self.skinner.faces.cpu().numpy().astype(np.uint32)
        self.vertex_count = len(self.skinner.bind_vertices)
        self.clips = {}
        self.world_metadata = None
        self.allow_runtime_controls = interactive
        self.committed_prefetch=committed_prefetch
        self.start_paused=start_paused
        self.entry_command=None
        self.max_speed_mps = 12. if interactive else 3.
        terrain_view = None
        if scene_path is not None:
            receipt = json.loads(scene_path.with_suffix('.json').read_text())
            self.entry_command=receipt.get('entry_command')
            if receipt.get('schema') == 'motionbricks.authored_test_scene.v1':
                self.world_metadata = receipt
                mesh_path = scene_path.parent / receipt['mesh_file']
                if mesh_path.resolve().parent != scene_path.resolve().parent or file_sha256(mesh_path) != receipt['mesh_sha256']:
                    raise ValueError('Authored terrain mesh identity differs')
                terrain_view = {'hf':torch.as_tensor(scene['heightfield_y_m']).cpu().tolist(),
                    'min_point':torch.as_tensor(scene['min_point_world_xyz_m']).cpu().tolist(),
                    'dx':float(scene['dx_m']),'mesh':json.loads(mesh_path.read_text()),
                    'spawns':receipt['spawns'],'world_size_m':receipt['size_m']}
        self.spawn_histories = {}
        self.spawn_ready = {}
        self.default_spawn = 'zone-03' if self.world_metadata else None
        if self.world_metadata:
            rep = inference.global_motion_rep
            for spawn in self.world_metadata['spawns']:
                placed = rep.change_first_heading(history.clone().to(device),float(spawn['heading']),False,False)
                placed[...,[0,2]] += placed.new_tensor(spawn['spawn_xz']) - placed[:,-1:,[0,2]]
                if not torch.equal(placed[...,1],history.to(device)[...,1]):
                    raise ValueError('Scene reset changed source body height')
                self.spawn_histories[spawn['id']] = placed
            history = self.spawn_histories[self.default_spawn]
        self.configure_terrain_live(backend_builder, scene, history, inference,
            terrain_quality={'vertical_frame':inference._pose_model.args.get('vertical_frame'),
                             'model_file':bundle_path.name, 'model_label':label,
                             'model_sha256':expected_hash,
                             'scene_file':scene_path.name if scene_path else None}, terrain_view=terrain_view)
        self.initial_root = self.terrain_initial_root
        self.initial_vertices = self.terrain_initial_vertices
        self.initial_joints = self.terrain_initial_joints
        self.terrain_keep_ready = False
        if joint_checkpoint is not None:
            from motionbricks.data.joint_terrain_live import joint_checkpoint_progress
            self.terrain_quality.update(joint_checkpoint_sha256=file_sha256(joint_checkpoint),
                control_version=CONTROL_VERSION,**joint_checkpoint_progress(checkpoint))
            export_path=joint_checkpoint.parent/'export.json'
            if export_path.is_file():
                exported=json.loads(export_path.read_text())
                if (exported.get('schema')=='motionbricks.approved_parc_live.v1'
                        and exported.get('user_visual_approval') is True
                        and exported.get('checkpoint_sha256')==self.terrain_quality['joint_checkpoint_sha256']
                        and exported.get('source_bundle_sha256')==expected_hash):
                    self.terrain_quality.update(visual_baseline_approved=True,
                        quality_note='Visual baseline accepted; interactive validation pending')
        if execution == 'cuda-graph':
            if self.joint_controller is not None and hasattr(self.joint_controller,'sdf_enabled'):
                raise ValueError('Joint SDF requires --execution eager until its dynamic queries are separately compiled and verified')
            print('Compiling and checking this checkpoint against native inference', flush=True)
            if self.joint_controller is not None:
                from motionbricks.data.joint_terrain_execution import JointTerrainExecution
                runner=JointTerrainExecution(self.joint_controller,self.terrain_scene,self.skinner,self.terrain_initial_history,flexible=interactive)
            else:
                from motionbricks.data.soma_terrain_execution import TerrainExecution
                runner = TerrainExecution(inference,self.terrain_scene,self.skinner,self.terrain_initial_history,
                                          flexible=interactive)
            object.__setattr__(inference, '_terrain_execution', runner)
            self.terrain_quality.update(execution='verified-static-cuda-v1', execution_parity=runner.validation)
        else:
            self.terrain_quality['execution'] = 'native-eager'
        self.rig_transport=joint_checkpoint is not None and execution=='eager'
        if self.rig_transport:
            with torch.inference_mode():
                sample=self.terrain_initial_history
                geometry=self.skin_rig_geometry(sample,inference)
                _,reference,_=self.skin_features_with_joints(sample,inference)
                actual=self.skinner.lbs(geometry['rig'])
                torch.testing.assert_close(actual,torch.as_tensor(reference,device=actual.device),atol=1e-5,rtol=1e-5)
                self.terrain_quality['rig_transport_max_vertex_error_m']=float((actual-torch.as_tensor(reference,device=actual.device)).abs().max())
    @torch.inference_mode()
    def warm_live_entry(self):
        if self.start_paused and self.entry_command is not None:
            from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS
            begin=time.perf_counter()
            temporary=self.create_backend('terrain');temporary.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS))
            warm=temporary.predict_plan(self.entry_command['velocity_xz_mps'],self.entry_command['heading_rad'])
            self.skin_rig_geometry(warm['planned_motion'],self.inference)
            if self.terrain_initial_history.is_cuda:torch.cuda.synchronize(self.terrain_initial_history.device)
            self.terrain_quality['warmup_ms']=(time.perf_counter()-begin)*1000
            del temporary,warm

    @torch.inference_mode()
    def skin_rig_geometry(self,features,inference):
        pose=inference.global_motion_rep.inverse(features,False,return_all=False,joint_positions_from='global_rot_data')
        captured={};original=self.skinner.lbs
        def capture(rig):
            captured['rig']=rig
            return rig.new_empty(0)
        try:
            self.skinner.lbs=capture
            self.skinner.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
        finally:
            self.skinner.lbs=original
        return {'rig':captured['rig'],'joints':pose['posed_joints'][0]}

    def config(self):
        value = super().config()
        value['default_mode'] = 'terrain'
        value['start_paused']=self.start_paused
        if self.entry_command is not None:value['entry_command']=self.entry_command
        if self.committed_prefetch:
            value['terrain_mode']['prefetch']={'version':'committed-buffer-v3','played_frame_count':0}
        value['modes'] = [{'id':'terrain','label':self.terrain_quality['model_label']}]
        value['live_model'] = {'source':'A100 exported checkpoint','terrain_conditioned':True}
        if self.world_metadata:
            value['world'] = {'spawns':self.world_metadata['spawns'],'default_spawn':self.default_spawn,
                'size_m':self.world_metadata['size_m'],'scene_hash':self.world_metadata['scene_sha256'],
                'authored_test_world':True}
        if self.allow_runtime_controls:
            from motionbricks.data.terrain_runtime_controls import RUNTIME_DEFAULTS,JOINT_RUNTIME_DEFAULTS
            defaults=JOINT_RUNTIME_DEFAULTS if self.joint_controller is not None else RUNTIME_DEFAULTS
            value['runtime_controls'] = {'defaults':defaults,'max_speed_mps':self.max_speed_mps,
                'apply':'next_fully_played_boundary',
                'trained_defaults':JOINT_RUNTIME_DEFAULTS if self.joint_controller is not None else {'num_tokens':8,'inference_steps':4,'lookahead_m':-1.,
                    'lateral_halfwidth_m':.6,'refresh_frames':24,'lookahead_gain':1.,'lookahead_base_m':.4,
                    'path_refinements':0}}
        return value

    def create_backend(self, mode='terrain', spawn_id=None):
        if mode != 'terrain':
            raise ValueError('This endpoint serves only the named A100 terrain model')
        if spawn_id is not None:
            if not isinstance(spawn_id,str) or spawn_id not in self.spawn_histories:
                raise ValueError('Unknown terrain entry point')
            return self.terrain_backend_builder(self.terrain_inference,self.spawn_histories[spawn_id].clone(),self.terrain_scene)
        return super().create_backend(mode)

    def ready(self, mode='terrain', spawn_id=None):
        if spawn_id is not None:
            if mode!='terrain' or spawn_id not in self.spawn_histories:
                raise ValueError('Unknown terrain entry point')
            if spawn_id not in self.spawn_ready:
                history=self.spawn_histories[spawn_id]
                roots,vertices,joints=self.skin_features_with_joints(history[:,-1:],self.terrain_inference)
                heading=history[0,-1,3:5]
                self.spawn_ready[spawn_id]={'type':'ready','mode':'terrain','fps':self.fps,
                    'vertex_count':self.vertex_count,'root':roots[0].tolist(),'vertices':vertices[0].reshape(-1).tolist(),
                    'joints_world_xyz_m':joints[0].tolist(),'heading':float(torch.atan2(heading[1],heading[0])),
                    'spawn_id':spawn_id}
            return self.spawn_ready[spawn_id]
        return super().ready(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--scene',type=Path)
    p.add_argument('--label',required=True)
    p.add_argument('--port',type=int,default=28930)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execution',choices=('eager','cuda-graph'),default='cuda-graph')
    p.add_argument('--interactive',action='store_true')
    p.add_argument('--joint-checkpoint',type=Path)
    p.add_argument('--committed-prefetch',action='store_true')
    p.add_argument('--start-paused',action='store_true')
    p.add_argument('--trace-worker',action='store_true')
    p.add_argument('--default-stream',action='store_true')
    args=p.parse_args()
    runtime=A100TrialRuntime(args.bundle,args.sha256,args.device,args.scene,args.label,args.execution,args.interactive,args.joint_checkpoint,args.committed_prefetch,args.start_paused)
    runtime.trace_worker=args.trace_worker
    runtime.prefer_default_cuda_stream=args.default_stream
    print(json.dumps({'ready':True,'port':args.port,'model':runtime.terrain_quality}),flush=True)
    web.run_app(create_app(runtime,ROOT/'web/soma_lab'),host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
