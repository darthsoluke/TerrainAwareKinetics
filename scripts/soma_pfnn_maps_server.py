"""All original PFNN maps with one active, hash-bound final-model runtime."""

import argparse
import gc
import json
from pathlib import Path
import time
import weakref

import numpy as np
import torch
from aiohttp import web

from motionbricks.data.full_corpus_acceptance import validate_scene
from motionbricks.data.pfnn_tuning import TunablePFNNBackend,tuning_contract
from motionbricks.data.anticipatory_speed import TerrainSpeedMap,DEFAULTS as SPEED_DEFAULTS,LIMITS as SPEED_LIMITS
from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
from scripts.soma_lab_server import ROOT, create_app
from scripts.train_native_mb import sha256


class PFNNMapRuntime:
    def __init__(self,bundle,catalog,device='cuda:0',execution='verified-cuda',initial_map=None,
                 *,planning_policy='full',dual_proof=None):
        self.bundle_path,self.catalog_path,self.device,self.execution=bundle,catalog,device,execution
        self.catalog=json.loads(catalog.read_text())
        self.planning_policy_mode=planning_policy
        self.policy_scene_reports={};self.policy_proof=None
        if planning_policy not in ('full','dual'):raise ValueError('Unknown planning policy')
        if planning_policy=='dual':
            if execution!='root-path-live':raise ValueError('Dual policy requires Root-path execution')
            from scripts.dual_horizon_deployment import load_proof
            self.policy_proof=load_proof(dual_proof,self.catalog['checkpoint_sha256'])
        if self.catalog['schema']!='motionbricks.pfnn_map_catalog.v1':
            raise ValueError('Unknown PFNN catalog')
        self.entries={entry['id']:entry for entry in self.catalog['entries']}
        if len(self.entries)!=self.catalog['unique_map_count'] or len(self.entries)!=len(self.catalog['entries']):
            raise ValueError('Duplicate or missing PFNN maps')
        self.assets={}
        self.speed_maps={}
        for map_id,entry in self.entries.items():
            folder=(catalog.parent/entry['artifact']).resolve()
            if folder.parent!=catalog.parent.resolve():
                raise ValueError('Map artifact must be directly inside its catalog')
            receipt=json.loads((folder/'acceptance.json').read_text())
            if (receipt['checkpoint_sha256']!=self.catalog['checkpoint_sha256']
                    or receipt['scene_sha256']!=entry['scene_sha256']
                    or sha256(folder/'scene.npz')!=entry['scene_sha256']):
                raise ValueError('Catalog model or map identity differs')
            with np.load(folder/'scene.npz',allow_pickle=False) as data:
                scene={key:data[key].copy() for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
                history=data['initial_history'].copy()
                metadata=json.loads(str(data['metadata_json']))
            validate_scene(scene,history,metadata,receipt['bundle_sha256'])
            self.assets[map_id]=(folder,scene,history,metadata)
        self.scene_revision=0
        self._sessions=weakref.WeakSet()
        self._active=None
        self.current_map=None
        self.load_reports={}
        self.select_map(initial_map or self.catalog['default_map'])

    def __getattr__(self,name):
        active=object.__getattribute__(self,'_active')
        if active is None:
            raise AttributeError(name)
        return getattr(active,name)

    def track_session(self,session):
        self._sessions.add(session)

    def _release(self):
        self.scene_revision+=1
        for session in self._sessions:
            session.backend=None
            session.pending_playback=None
        self._sessions.clear()
        if self._active is not None:
            torch.cuda.synchronize(self.device)
            self._active.close()
            object.__setattr__(self._active.inference,'_terrain_execution',None)
            self._active=None
            gc.collect()
            torch.cuda.empty_cache()

    def select_map(self,map_id):
        if not isinstance(map_id,str) or map_id not in self.entries:
            raise ValueError('Unknown PFNN map')
        if map_id==self.current_map and self._active is not None:
            return
        started=time.monotonic()
        previous=self.current_map
        self.scene_revision+=1
        for session in self._sessions:
            session.backend=None
            session.pending_playback=None
        self._sessions.clear()
        folder=self.assets[map_id][0]
        try:
            if self._active is None:
                self._active=FullCorpusAcceptanceRuntime(self.bundle_path,folder,self.device,
                    acceptance_test=True,execution=self.execution,spatial_index='bvh',reusable_scenes=True)
            else:
                self._active.switch_artifact(folder)
            # Large-map queries already keep each real plan active long enough.
            self._active.terrain_keep_ready=False
            self.current_map=map_id
            if self.planning_policy_mode=='dual' and map_id not in self.policy_scene_reports:
                from scripts.dual_horizon_deployment import check_scene
                self.policy_scene_reports[map_id]=check_scene(self.inference._terrain_execution,
                    self.terrain_initial_history,self._active.acceptance_receipt['scene_sha256'])
            self.load_reports[map_id]=dict(map_id=map_id,seconds=time.monotonic()-started,
                model_sha256=self._active.acceptance_receipt['checkpoint_sha256'],
                scene_sha256=self._active.acceptance_receipt['scene_sha256'],
                execution_parity=self._active.terrain_quality.get('execution_parity'),
                rig_error_m=self._active.terrain_quality['rig_transport_max_vertex_error_m'],
                actual_optimizer_updates=0,quality_approved=False)
        except Exception:
            self.current_map=None
            if previous is not None and previous!=map_id:
                self.select_map(previous)
            raise

    def keep_ready_current_scene(self):
        return None

    def create_backend(self,mode='terrain',spawn_id=None):
        if mode!='terrain' or spawn_id not in (None,'hub'):
            raise ValueError('PFNN viewer requires its declared terrain spawn')
        backend=TunablePFNNBackend(self._active.unified_controller,
            self._active.terrain_initial_history.clone(),self._active.terrain_scene,
            speed_map=self.speed_map())
        backend.configure_runtime(self.runtime_contract()['defaults'])
        if self.planning_policy_mode=='dual':
            from motionbricks.data.dual_horizon_execution import DualHorizonExecution,DualHorizonBackend
            engine=DualHorizonExecution.from_runner(self.inference._terrain_execution)
            engine.validation=self.policy_proof['candidate_parity']
            backend=DualHorizonBackend.from_backend(backend,engine)
        return backend

    def runtime_contract(self):
        contract=tuning_contract()
        if self.execution=='root-path-live':
            contract['defaults']['refresh_frames']=4
            contract['accelerated_horizons']=dict(tokens=[8,12,16],inference_steps=4,refresh_frames=4,
                usable_seconds=[.8,40/30,56/30],prefetch_switch=True)
            contract['speed_governor']=dict(defaults=dict(SPEED_DEFAULTS),limits=dict(SPEED_LIMITS),
                apply='next_fully_played_boundary',schema='motionbricks.anticipatory_speed.v1')
            if self.planning_policy_mode=='dual':
                from motionbricks.data.dual_horizon_execution import policy
                contract['planning_policy']=dict(defaults=policy(8),lookahead_choices=[8,12,16],
                    apply='prefetch_candidate_then_ack',coarse='shared8',execution_frames=32)
                contract['limits'].update(num_tokens=[8,8],inference_steps=[4,4],refresh_frames=[4])
        return contract

    def speed_map(self):
        if self.execution!='root-path-live':return None
        if self.current_map not in self.speed_maps:
            self.speed_maps[self.current_map]=TerrainSpeedMap(self.assets[self.current_map][1])
        return self.speed_maps[self.current_map]

    def tuning_config(self):
        return dict(runtime=self.runtime_contract(),map_id=self.current_map,
            checkpoint_sha256=self.catalog['checkpoint_sha256'],quality_approved=False,
            observation=dict(volume_grid_size=8,points_per_region=512,regions=3,
                body_halfwidth_m=.8,corridor_halfwidth_m=.7,vertical_below_m=2.,vertical_above_m=2.))

    def config(self):
        result=self._active.config()
        if self.execution=='root-path-live':
            from motionbricks.data.root_path_terrain import POLICY
            result['live_model']['observation_policy']=POLICY
            result['terrain_mode']['prefetch'].update(played_frame_count=2,allowed_played_frame_counts=[0,2,4],
                selection='measured-request-to-binary',margin_ms=18,window_size=12,decision='packet-start')
        result['map_catalog']=dict(current=self.current_map,
            entries=[{key:entry[key] for key in ('id','label','category','size_m')} for entry in self.entries.values()],
            total=len(self.entries),all_maps_loaded=True)
        result['runtime_controls']=self.runtime_contract()
        result['tuning_page']='/settings'
        return result

    def selection_config(self,cached_scene=False):
        result=self.config()
        for key in ('soma_skin','faces','clips'):
            result.pop(key,None)
        if cached_scene:result.pop('terrain_scene',None)
        result['map_switch_seconds']=self.load_reports[self.current_map]['seconds']
        return result

    def prepare_maps(self):
        initial=self.current_map
        samples={}
        runner=getattr(self._active.inference,'_terrain_execution',None)
        graph=None
        for map_id in self.entries:
            self.select_map(map_id)
            if runner is not None:
                from motionbricks.data.native_mb_online_live import online_command_goal
                with torch.inference_mode():
                    history=self._active.terrain_initial_history
                    goal=online_command_goal(history[...,:5],[0.,1.6],0.,8,runner.protocol_version)
                    values=runner._prediction(history,goal)
                    samples[map_id]={key:value.clone() for key,value in values.items()}
                    current=runner._compiled_prediction.graph
                    if graph is not None and current is not graph:raise ValueError('Map preparation replaced the shared graph')
                    graph=current
            print(json.dumps(dict(prepared_map=map_id,seconds=self.load_reports[map_id]['seconds'])),flush=True)
        self.cache_replay_checks={}
        for map_id in self.entries:
            self.select_map(map_id)
            if runner is not None:
                with torch.inference_mode():
                    history=self._active.terrain_initial_history
                    goal=online_command_goal(history[...,:5],[0.,1.6],0.,8,runner.protocol_version)
                    actual=runner._prediction(history,goal)
                    for key in actual:torch.testing.assert_close(actual[key],samples[map_id][key],atol=0,rtol=0)
                    if runner._compiled_prediction.graph is not graph:raise ValueError('Cached switch replaced the graph')
                self.cache_replay_checks[map_id]=dict(all_outputs_equal=True,shared_graph=True,
                    switch_seconds=self.load_reports[map_id]['seconds'])
        self.select_map(initial)

    def close(self):
        self._release()


def main(args):
    if args.port in (28936,28941,28942):
        raise ValueError('Preserved viewer ports cannot be used')
    execution='root-path-live' if args.root_path_observation else args.execution
    runtime=PFNNMapRuntime(args.bundle,args.catalog,args.device,execution,args.initial_map,
        planning_policy=args.planning_policy,dual_proof=args.dual_proof)
    if args.latency_review:
        from scripts.profile_root_path_runtime import profile_runtime
        profile_runtime(runtime,args.latency_review,args.latency_review_reference)
    if args.check_all:
        args.check_all.mkdir(parents=True,exist_ok=False)
        try:
            for map_id in runtime.entries:
                runtime.select_map(map_id)
                (args.check_all/'progress.json').write_text(json.dumps(runtime.load_reports,indent=2))
                print(json.dumps(dict(checked=map_id,completed=len(runtime.load_reports))),flush=True)
            first_loads=dict(runtime.load_reports)
            for map_id in runtime.entries:
                runtime.select_map(map_id)
            (args.check_all/'complete.json').write_text(json.dumps(dict(passed=True,first_loads=first_loads,
                catalog_sha256=sha256(args.catalog),maps=runtime.load_reports,
                actual_optimizer_updates=0,quality_approved=False,delivery_approved=False),indent=2))
        except Exception as error:
            (args.check_all/'failure.json').write_text(json.dumps(dict(error=repr(error),completed=runtime.load_reports)))
            raise
        finally:runtime.close()
        return
    runtime.cache_replay_checks={}
    if not args.root_path_observation:
        runtime.prepare_maps()
    if args.readiness_report:
        args.readiness_report.write_text(json.dumps(dict(passed=True,maps=runtime.load_reports,
            cache_replay=runtime.cache_replay_checks,catalog_sha256=sha256(args.catalog),
            horizon_storage=(runtime.inference._terrain_execution.constant_storage_report()
                             if args.root_path_observation else None),
            planning_policy=args.planning_policy,policy_scene_reports=runtime.policy_scene_reports,
            actual_optimizer_updates=0,quality_approved=False,delivery_approved=False),indent=2))
    gc.collect()
    torch.cuda.empty_cache()
    app=create_app(runtime,ROOT/'web/full_corpus_lab')
    async def close(_app):runtime.close()
    app.on_cleanup.append(close)
    print(json.dumps(dict(ready=True,port=args.port,maps=len(runtime.entries),current_map=runtime.current_map,
        checkpoint_sha256=runtime.catalog['checkpoint_sha256'],actual_optimizer_updates=0)),flush=True)
    web.run_app(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=28943)
    p.add_argument('--execution',choices=('eager','verified-cuda'),default='verified-cuda')
    p.add_argument('--check-all',type=Path)
    p.add_argument('--readiness-report',type=Path)
    p.add_argument('--root-path-observation',action='store_true')
    p.add_argument('--initial-map')
    p.add_argument('--planning-policy',choices=('full','dual'),default='full')
    p.add_argument('--dual-proof',type=Path)
    p.add_argument('--latency-review',type=Path)
    p.add_argument('--latency-review-reference',type=Path)
    main(p.parse_args())
