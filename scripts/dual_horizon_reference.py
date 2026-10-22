"""Native reference and far-removal ablation, excluded from serving imports."""
import torch
from motionbricks.data.dual_horizon_execution import DualHorizonExecution
from motionbricks.data.local_terrain_observation import LocalTerrainObservation
from motionbricks.data.native_mb_online_live import predict_planar_terrain
from motionbricks.data.root_path_split_execution import VOLUME_KEYS,_RootBoundary
from motionbricks.data.soma_terrain_execution import SETTINGS


class DualHorizonReference(DualHorizonExecution):
    def native_root(self,history,goal,query,coarse,tokens):
        original=self.inference._predict_root_trajectories
        def boundary(batch,config):raise _RootBoundary(original(batch,config))
        self.inference._predict_root_trajectories=boundary;self.observer.deferred=True
        try:
            with self.controller.context(history,self.scene,volume_observation=dict(zip(VOLUME_KEYS,coarse[8:]))):
                try:
                    predict_planar_terrain(self.inference,history,goal,self.scene,num_tokens=tokens,config=SETTINGS,
                        query_controls=query,prepared_observations=(LocalTerrainObservation(*coarse[:4]),LocalTerrainObservation(*coarse[4:8])))
                except _RootBoundary as stop:return dict(zip(('count','root','local'),stop.values))
        finally:
            self.inference._predict_root_trajectories=original;self.observer.deferred=False

    def prepare_dual(self,history,goal,query,tokens,*,native=False):
        if not native:return super().prepare_dual(history,goal,query,tokens)
        coarse=self._scene_inputs(history,goal,8,query)
        short=self.native_root(history,goal,query,coarse,8)
        long=self.native_root(history,goal,query,coarse,tokens)
        return self._prepare_observation(history,short,long,coarse,tokens)

    def native_suffix(self,history,goal,query,prepared,*,remove_far=False):
        inference,controller=self.inference,self.controller
        short,coarse,height=prepared['short'],prepared['coarse'],prepared['height']
        volume=prepared['volume']
        if remove_far:
            volume={key:value.clone() for key,value in volume.items()}
            ids=torch.arange(512,device=history.device)%8>=4
            volume['valid'][:,1,ids]=False;volume['features'][:,1,ids]=0
        captured={}
        original_root=inference._predict_root_trajectories
        original_decode=inference._vqvae_pose_model.forward_decoder
        def supplied_root(batch,config):
            controller.last_root=dict(pred_global_root_values=short['root'])
            self.observer.height=height;self.observer.context=controller._context
            return short['count'],short['root'],short['local']
        def decode(ids,*a,**kw):
            captured['tokens']=ids.clone();return original_decode(ids,*a,**kw)
        def height_input(_module,args):captured['height_features']=args[0].clone()
        handle=controller.adaptation['height_encoder'].register_forward_pre_hook(height_input)
        inference._predict_root_trajectories=supplied_root;inference._vqvae_pose_model.forward_decoder=decode
        try:
            with controller.context(history,self.scene,volume_observation=volume):
                motion,count=predict_planar_terrain(inference,history,goal,self.scene,num_tokens=8,config=SETTINGS,
                    query_controls=query,prepared_observations=(LocalTerrainObservation(*coarse[:4]),LocalTerrainObservation(*coarse[4:8])))
                captured.update(memory=controller.memory.clone(),memory_valid=controller.valid.clone(),
                    logits=controller.last_logits.clone(),motion=motion,count=count,condition_frames=controller.frames,
                    root=short['root'].clone(),local=short['local'].clone())
        finally:
            handle.remove();inference._predict_root_trajectories=original_root
            inference._vqvae_pose_model.forward_decoder=original_decode
        return captured
