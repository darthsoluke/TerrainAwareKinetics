"""Independent loopback acceptance endpoint for the full-corpus model schema."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

from motionbricks.data.full_corpus_acceptance import (
    artifact_name,load_acceptance_candidate,validate_acceptance,validate_fixed_profile,validate_scene,
)
from motionbricks.data.unified_terrain import UnifiedTerrainController,PROFILE
from motionbricks.data.unified_committed_playback import CommittedUnifiedBackend
from scripts.soma_a100_trial_server import A100TrialRuntime
from scripts.soma_lab_server import ROOT,create_app
from scripts.train_native_mb import sha256


class FixedFullCorpusBackend(CommittedUnifiedBackend):
    def configure_runtime(self,values):
        return super().configure_runtime(validate_fixed_profile(values,PROFILE))


class FullCorpusAcceptanceRuntime(A100TrialRuntime):
    def __init__(self,bundle_path,artifact,device='cuda:0',*,allow_mechanics=False,acceptance_test=False,
                 execution='eager',spatial_index='native',reusable_scenes=False):
        if not acceptance_test:
            raise ValueError('This endpoint requires an explicit local acceptance-test session')
        if execution not in ('eager','verified-cuda','root-path-live'):
            raise ValueError('Unknown full-corpus acceptance execution mode')
        if spatial_index not in ('native','bvh'):
            raise ValueError('Unknown scene spatial index')
        receipt=json.loads((artifact/'acceptance.json').read_text())
        weights=artifact/artifact_name(receipt['checkpoint_file'])
        scene_path=artifact/artifact_name(receipt['scene_file'])
        bundle_hash=sha256(bundle_path);checkpoint_hash=sha256(weights)
        if sha256(scene_path)!=receipt['scene_sha256']:raise ValueError('Scene artifact changed')
        saved=torch.load(weights,map_location='cpu',mmap=True,weights_only=False)
        validate_acceptance(saved,receipt,bundle_hash,checkpoint_hash,allow_mechanics=allow_mechanics)
        with np.load(scene_path,allow_pickle=False) as archive:
            expected={'heightfield_y_m','min_point_world_xyz_m','dx_m','initial_history','metadata_json'}
            if set(archive.files)!=expected:raise ValueError('Scene contains undeclared inputs')
            scene={k:torch.from_numpy(archive[k].copy()) for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
            history=torch.from_numpy(archive['initial_history'].copy())
            metadata=json.loads(str(archive['metadata_json']))
        validate_scene(scene,history,metadata,bundle_hash)
        super().__init__(bundle_path,bundle_hash,device,None,'MotionBricks','eager',
            interactive=True,start_paused=True,committed_prefetch=True)
        if self.vertex_count!=18056:
            raise ValueError('Acceptance requires the complete official18056-vertex SOMA skin')
        bundle=torch.load(bundle_path,map_location='cpu',mmap=True,weights_only=False)
        self.unified_controller=UnifiedTerrainController(self.inference,bundle)
        load_acceptance_candidate(self.unified_controller,saved,receipt,bundle_hash,checkpoint_hash,
            allow_mechanics=allow_mechanics)
        validate_scene(scene,history,metadata,bundle_hash,len(self.inference.global_motion_rep.indices['all']))
        self._bundle_hash,self._checkpoint_hash=bundle_hash,checkpoint_hash
        self._execution_mode,self._spatial_index=execution,spatial_index
        self._reusable_scenes=reusable_scenes
        self._scene_cache={}
        self._install_scene(artifact,receipt,scene,history,metadata)

    def _install_scene(self,artifact,receipt,scene,history,metadata):
        bundle_hash,checkpoint_hash=self._bundle_hash,self._checkpoint_hash
        execution,spatial_index=self._execution_mode,self._spatial_index
        device=self.inference.device
        weights=artifact/artifact_name(receipt['checkpoint_file'])
        def build(_inference,seed,terrain):
            backend=FixedFullCorpusBackend(self.unified_controller,seed,terrain)
            backend.configure_runtime(dict(PROFILE));return backend
        self.configure_terrain_live(build,scene,history,self.inference,terrain_quality=dict(
            model_label='MotionBricks',model_file=weights.name,model_sha256=checkpoint_hash,
            bundle_sha256=bundle_hash,scene_sha256=receipt['scene_sha256'],
            model_state_sha256=receipt['model_state_sha256'],codec_state_sha256=receipt['codec_state_sha256'],
            acceptance_only=True,mechanics_only=receipt['mechanics_only'],quality_approved=False,
            delivery_approved=False,execution='full-corpus-committed-eager-v1'))
        if spatial_index=='bvh':
            from motionbricks.data.bvh_heightfield import BVHHeightfieldPrismSDF
            proof=json.loads((artifact/'geometry_query_review.json').read_text())
            implementation=Path(__file__).parents[1]/'motionbricks/data/bvh_heightfield.py'
            if (not proof.get('passed') or proof.get('scene_sha256')!=receipt['scene_sha256']
                    or proof.get('implementation_sha256')!=sha256(implementation)
                    or proof.get('distance_max_abs')!=0 or proof.get('normal_max_abs')!=0
                    or proof.get('valid_equal') is not True):
                raise ValueError('Scene BVH requires exact native-query evidence for this map and source')
            from motionbricks.data.unified_scene_field import AnalyticPlaneField,field_from_heightfield
            height=self.terrain_scene['heightfield_y_m']
            field=(field_from_heightfield(self.terrain_scene) if torch.equal(height,height[:1,:1].expand_as(height))
                   else BVHHeightfieldPrismSDF(self.terrain_scene,device))
            self.unified_controller._fields[id(self.terrain_scene['heightfield_y_m'])]=(self.terrain_scene,field)
            self.terrain_quality.update(spatial_index='native-analytic-plane' if isinstance(field,AnalyticPlaneField) else 'exact-native-prism-bvh',
                geometry_query_review_sha256=sha256(artifact/'geometry_query_review.json'))
        if metadata.get('display_mesh'):
            descriptor=metadata['display_mesh']
            mesh_path=artifact/artifact_name(descriptor['file'])
            if sha256(mesh_path)!=descriptor['sha256']:
                raise ValueError('Static terrain display mesh changed')
            with np.load(mesh_path,allow_pickle=False) as archive:
                vertices,faces,colors=(archive[key] for key in ('vertices','faces','colors'))
                if (vertices.ndim!=2 or vertices.shape[1]!=3 or colors.shape!=vertices.shape
                        or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(vertices).all()
                        or not np.isfinite(colors).all() or faces.min()<0 or faces.max()>=len(vertices)):
                    raise ValueError('Invalid bound terrain mesh')
                self.terrain_view['mesh']=dict(vertices=vertices.reshape(-1).tolist(),
                    faces=faces.reshape(-1).tolist(),colors=colors.reshape(-1).tolist(),
                    semantics=descriptor['semantics'])
        if metadata.get('world'):
            world=metadata['world']
            if (len(world['spawns'])!=1 or world['spawns'][0]['id']!='hub'
                    or world.get('default_spawn')!='hub'):
                raise ValueError('This acceptance map requires one bound factual spawn')
            self.world_metadata=dict(world,scene_sha256=receipt['scene_sha256'])
            self.default_spawn='hub'
            self.spawn_histories['hub']=self.terrain_initial_history.clone()
        self.acceptance_receipt=receipt
        self.initial_root,self.initial_vertices,self.initial_joints=(
            self.terrain_initial_root,self.terrain_initial_vertices,self.terrain_initial_joints)
        self.rig_transport=True;self.prefer_default_cuda_stream=True
        with torch.inference_mode():
            rig=self.skin_rig_geometry(self.terrain_initial_history,self.inference)['rig']
            _,reference,_=self.skin_features_with_joints(self.terrain_initial_history,self.inference)
            actual=self.skinner.lbs(rig);reference=torch.as_tensor(reference,device=actual.device)
            torch.testing.assert_close(actual,reference,atol=1e-5,rtol=1e-5)
            self.terrain_quality['rig_transport_max_vertex_error_m']=float((actual-reference).abs().max())
        if execution=='verified-cuda':
            if self._reusable_scenes:
                from motionbricks.data.soma_reusable_terrain_execution import ReusableTerrainExecution
                runner=getattr(self.inference,'_terrain_execution',None)
                if runner is None:
                    runner=ReusableTerrainExecution(self.inference,self.terrain_scene,self.skinner,
                        self.terrain_initial_history,controller=self.unified_controller)
                else:
                    runner.bind_scene(self.terrain_scene,self.terrain_initial_history)
            else:
                from motionbricks.data.soma_terrain_execution import TerrainExecution
                runner=TerrainExecution(self.inference,self.terrain_scene,self.skinner,
                    self.terrain_initial_history,controller=self.unified_controller,
                    independent_validation_seeds=True,fk_tolerance_m=3e-5)
            object.__setattr__(self.inference,'_terrain_execution',runner)
            self.terrain_keep_ready=True
            self.terrain_quality.update(execution='full-corpus-committed-verified-cuda-v1',
                execution_parity=runner.validation)
        elif execution=='root-path-live':
            from motionbricks.data.root_path_terrain import POLICY
            from motionbricks.data.root_path_split_execution import SplitRootPathExecution
            runner=getattr(self.inference,'_terrain_execution',None)
            if runner is None:
                runner=SplitRootPathExecution(self.inference,self.terrain_scene,self.skinner,
                    self.terrain_initial_history,controller=self.unified_controller)
            else:
                runner.bind_scene(self.terrain_scene,self.terrain_initial_history)
            object.__setattr__(self.inference,'_terrain_execution',runner)
            self.terrain_keep_ready=False
            self.terrain_quality.update(execution='split-cuda-neural-compiled-geometry-root-path-v2',
                observation_policy=POLICY,execution_parity=runner.validation)
        if self._reusable_scenes:
            names=('terrain_scene','terrain_view','terrain_quality','terrain_initial_history',
                'terrain_initial_root','terrain_initial_vertices','terrain_initial_joints',
                'initial_root','initial_vertices','initial_joints','world_metadata','default_spawn',
                'spawn_histories','spawn_ready','acceptance_receipt')
            self._scene_cache[str(artifact.resolve())]={name:getattr(self,name) for name in names}

    def switch_artifact(self,artifact):
        import gc
        cached=self._scene_cache.get(str(artifact.resolve()))
        if cached is not None:
            torch.cuda.synchronize(self.inference.device)
            for name,value in cached.items():setattr(self,name,value)
            runner=getattr(self.inference,'_terrain_execution',None)
            if runner is not None:
                runner.scene=self.terrain_scene
                runner.field=self.unified_controller.scene_field(self.terrain_scene)
                runner.validation=self.terrain_quality['execution_parity']
            return
        receipt=json.loads((artifact/'acceptance.json').read_text())
        fixed=lambda row:{key:value for key,value in row.items() if key!='scene_sha256'}
        if fixed(receipt)!=fixed(self.acceptance_receipt):
            raise ValueError('Map switching cannot change the model, runtime or completed training contract')
        path=artifact/artifact_name(receipt['scene_file'])
        if sha256(path)!=receipt['scene_sha256']:
            raise ValueError('Selected map changed')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'heightfield_y_m','min_point_world_xyz_m','dx_m','initial_history','metadata_json'}:
                raise ValueError('Selected map contains undeclared inputs')
            scene={key:torch.from_numpy(data[key].copy()) for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
            history=torch.from_numpy(data['initial_history'].copy())
            metadata=json.loads(str(data['metadata_json']))
        validate_scene(scene,history,metadata,self._bundle_hash,len(self.inference.global_motion_rep.indices['all']))
        torch.cuda.synchronize(self.inference.device)
        if not self._reusable_scenes:
            object.__setattr__(self.inference,'_terrain_execution',None)
            self.unified_controller._fields.clear()
        self.spawn_histories={};self.spawn_ready={}
        self.terrain_scene=None;self.terrain_view=None;self.world_metadata=None
        if not self._reusable_scenes:
            gc.collect();torch.cuda.empty_cache()
        from motionbricks.data.soma_static_execution import restore_native_transformers
        restore_native_transformers(self.inference)
        self._install_scene(artifact,receipt,scene,history,metadata)

    @torch.inference_mode()
    def skin_rig_geometry(self,features,inference):
        geometry=super().skin_rig_geometry(features,inference)
        if hasattr(self,'unified_controller'):
            field=self.unified_controller.scene_field(self.terrain_scene)
            distance,_,valid=field.query(geometry['joints'])
            geometry.update(bone_sdf_distance=distance,bone_sdf_valid=valid)
        return geometry

    def config(self):
        value=super().config()
        value['runtime_controls']=dict(defaults=dict(PROFILE),trained_defaults=dict(PROFILE),
            max_speed_mps=self.max_speed_mps,locked=True,apply='next_fully_played_boundary')
        value['live_model']=dict(source='PFNN / PARC / BONES',full_model=True,terrain_conditioned=True,
            candidate_sha256=self.acceptance_receipt['checkpoint_sha256'],
            mechanics_only=self.acceptance_receipt['mechanics_only'],visual_approved=False,
            quality_approved=False,delivery_approved=False,acceptance_only=True)
        if self.world_metadata:
            value['world'].update(authored_test_world=False,label=self.world_metadata['label'])
        return value

    def close(self):
        self.unified_controller.close()


def main(a):
    if a.port in (28936,28941,28942):raise ValueError('Preserved viewer ports cannot be used')
    runtime=FullCorpusAcceptanceRuntime(a.bundle,a.artifact,a.device,
        allow_mechanics=a.allow_mechanics,acceptance_test=a.acceptance_test,execution=a.execution,
        spatial_index=a.spatial_index)
    app=create_app(runtime,ROOT/'web/full_corpus_lab')
    async def close(_app):runtime.close()
    app.on_cleanup.append(close)
    print(json.dumps(dict(ready=True,port=a.port,purpose='local_acceptance_test',delivery_approved=False)))
    web.run_app(app,host='127.0.0.1',port=a.port)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--port',type=int,default=28943)
    parser.add_argument('--execution',choices=('eager','verified-cuda'),default='eager')
    parser.add_argument('--spatial-index',choices=('native','bvh'),default='native')
    parser.add_argument('--acceptance-test',action='store_true');parser.add_argument('--allow-mechanics',action='store_true')
    main(parser.parse_args())
