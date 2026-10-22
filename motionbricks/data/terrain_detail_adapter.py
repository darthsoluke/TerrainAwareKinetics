"""Small terrain-relief adaptation with a frozen native motion backbone."""

import torch

from .terrain_model_v3 import JointTerrainController,FarSceneResidual


def near_relief(observation):
    points=observation['points'];valid=observation['valid']
    return {'points':points,'far_points':points.flatten(1,2),'far_valid':valid.flatten(1,2),
        'far_min_y':points[...,1].flatten(1,2),'far_views':points.shape[1],
        'speed_mps':observation['speed_mps'],'horizon_seconds':observation['horizon_seconds']}


class TerrainDetailController(JointTerrainController):
    def __init__(self,inference,bundle):
        super().__init__(inference,bundle)
        self.root_detail=FarSceneResidual(self.root.args['n_embd']).to(inference.device)
        self.pose_detail=FarSceneResidual(self.pose._args['n_embd']).to(inference.device)
        self.decoder_detail=FarSceneResidual(self.codec.code_dim).to(inference.device)
        self.root_detail_ref=FarSceneResidual(self.root.args['n_embd']).to(inference.device).requires_grad_(False)
        self.pose_detail_ref=FarSceneResidual(self.pose._args['n_embd']).to(inference.device).requires_grad_(False)
        self.decoder_detail_ref=FarSceneResidual(self.codec.code_dim).to(inference.device).requires_grad_(False)
        self.eval()

    def initialize_detail_from_far(self):
        for name in ('root','pose','decoder'):
            state=getattr(self,name+'_far').state_dict()
            getattr(self,name+'_detail').load_state_dict(state)
            getattr(self,name+'_detail_ref').load_state_dict(state)

    def _detail(self,name,queries):
        observation=near_relief(self._observation)
        update=getattr(self,name+'_detail')(queries,observation)
        reference=getattr(self,name+'_detail_ref')(queries,observation)
        return queries+(update-reference)

    def _root_scene_hook(self,module,args,output):
        result=super()._root_scene_hook(module,args,output)
        return self._detail('root',result)

    def _pose_scene_hook(self,module,args,output):
        result=super()._pose_scene_hook(module,args,output)
        return self._detail('pose',result)

    def _decoder_hook(self,module,args):
        result=super()._decoder_hook(module,args)
        latent=self._detail('decoder',result[0].transpose(1,2))
        return (latent.transpose(1,2),*result[1:])
