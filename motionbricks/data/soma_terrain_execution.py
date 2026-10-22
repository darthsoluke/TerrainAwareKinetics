"""Fixed-shape execution of the existing native MB and official SOMA functions.

The live contract is one four-frame history, eight tokens, and four argmax
passes. Tracing specializes Python validation/control flow to that contract;
commands and all motion/terrain tensor calculations remain live inputs. No
checkpoint, token selection rule, decoded motion, or playback boundary changes.
"""

import torch
from contextlib import nullcontext

from .local_terrain_observation import LocalTerrainObservation
from .native_mb_observed_history import reencode_played_history
from .native_mb_online_live import (online_command_goal, predict_planar_terrain,
                                    intent_diagnostics,add_survey_diagnostics,_future_xyz)
from .soma_cuda_graph import StaticCudaCall
from .soma_static_execution import trace_fixed_call,use_dense_transformers


SETTINGS = {'num_inference_step': 4, 'pose_token_sampling_use_argmax': True,
            'cache_terrain_encoder': True}


class TerrainExecution:
    """Single-worker inference; published arrays never alias replay storage."""

    @torch.inference_mode()
    def __init__(self, inference, scene, skinner, seed, *, flexible=False, controller=None,
                 independent_validation_seeds=False, fk_tolerance_m=2e-5):
        if not 0 < fk_tolerance_m <= 1e-4:
            raise ValueError('Compiled FK tolerance must remain at most0.1mm')
        if not seed.is_cuda or seed.shape[:2] != (1, 4):
            raise ValueError('Terrain execution requires one CUDA four-frame seed')
        self.inference = inference
        self.flexible = flexible
        self.controller = controller
        if controller is not None:
            if flexible:
                raise ValueError('Unified capture initially supports the verified fixed horizon only')
            controller.eval().requires_grad_(False)
        field = controller.scene_field(scene) if controller is not None else None
        self.protocol_version = inference._root_model.args['online_conditioning']
        inference.eval().requires_grad_(False)
        width = len(inference.global_motion_rep.indices['all'])
        self.history_shape = (1, 4, width)
        captured = {}
        native_decode = inference._vqvae_pose_model.forward_decoder

        def decode(ids, *args, **kwargs):
            captured['tokens'] = ids.clone()
            return native_decode(ids, *args, **kwargs)

        inference._vqvae_pose_model.forward_decoder = decode
        native_lbs = skinner.lbs

        def capture_rig(transforms):
            captured['rig'] = transforms
            return native_lbs(transforms)

        skinner.lbs = capture_rig

        def prediction(history,goal,controls=None,route=None,route_valid=None):
            tensors = {}
            tokens = controls[:,0:1].long() if flexible else 8
            query = controls[:,1:] if flexible else None
            if controller is not None:
                query = history.new_tensor([[-1., .6, 1., .4]])
            with (controller.context(history, scene) if controller is not None else nullcontext()):
                motion, count = predict_planar_terrain(inference, history, goal, scene,
                    num_tokens=tokens,config=SETTINGS,tensor_diagnostics=tensors,query_controls=query,
                    query_path_world_xz=route,query_path_valid=route_valid)
                if controller is not None:
                    observation = controller.volume_observation
                    tensors.update(sdf_points=observation['points'], sdf_distance=observation['distance'],
                                   sdf_valid=observation['valid'])
            return {'motion': motion, 'count': count, 'tokens': captured['tokens'], **tensors}

        def geometry(motion):
            pose = inference.motion_rep.inverse(motion, False, return_all=False)
            vertices = skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
            checks = {}
            history = reencode_played_history(motion[:, -4:], inference.global_motion_rep, tensor_checks=checks)
            result = {'rig': captured['rig'], 'joints': pose['posed_joints'][0], 'history': history, **checks}
            if field is not None:
                distance, normal, valid = field.query(result['joints'])
                result.update(bone_sdf_distance=distance, bone_sdf_valid=valid)
            return result

        def observe(motion):
            return reencode_played_history(motion, inference.global_motion_rep)

        def observed_with_checks(motion):
            checks = {}
            history = reencode_played_history(motion, inference.global_motion_rep, tensor_checks=checks)
            return {'history': history, **checks}

        try:
            cases, current = [], seed.clone()
            probes = [([0.,0.],0.,8,-1.,.6),([0.,1.6],0.,8,-1.,.6),
                      ([1.6,0.],1.57,8,-1.,.6),([0.,-1.6],3.14,8,-1.,.6),
                      ([-1.6,0.],-1.57,8,-1.,.6)]
            if flexible:
                probes += [([0.,1.6],0.,12,-1.,.6),([0.,1.6],0.,16,3.4,1.),
                           ([0.,1.6],0.,6,.8,.3),([0.,1.6],0.,10,2.4,.8),
                           ([0.,12.],0.,16,20.,6.)]
            for velocity,heading,tokens,ahead,width_m in probes:
                if independent_validation_seeds:
                    current = seed.clone()
                goal = online_command_goal(current[..., :5],velocity,heading,tokens,self.protocol_version)
                route=current[:,-1:,[0,2]].expand(-1,56,-1).clone()
                route_valid=torch.zeros((1,56),device=current.device,dtype=torch.bool)
                if flexible and tokens==10:
                    theta=torch.linspace(0,1.57,56,device=current.device)
                    route=route+torch.stack((1.5*(1-torch.cos(theta)),1.5*torch.sin(theta)),-1)[None]
                    route_valid[:]=True
                inputs = (current,goal,current.new_tensor([[tokens,ahead,width_m,1.,.4]]),route,route_valid) if flexible else (current,goal)
                expected = prediction(*inputs)
                cases.append((inputs,expected,tokens))
                current = observe(expected['motion'][:,tokens*4-8:tokens*4-4])
            use_dense_transformers(inference)
            self._prediction = StaticCudaCall(trace_fixed_call(prediction,cases[0][0],cuda_constants=True))
            sample = cases[0][1]['motion'][:, 4:12].contiguous()
            self._geometry = StaticCudaCall(trace_fixed_call(geometry, (sample,), cuda_constants=True))
            self._observe = StaticCudaCall(trace_fixed_call(observed_with_checks, (seed,), cuda_constants=True))
            self.validation = []
            for inputs,expected,tokens in cases:
                history,goal = inputs[:2]
                actual = self._prediction(*(value.contiguous() for value in inputs))
                expected_pose = inference.motion_rep.inverse(expected['motion'][:, :tokens*4],False,return_all=False)
                actual_pose = inference.motion_rep.inverse(actual['motion'][:, :tokens*4],False,return_all=False)
                error = float((actual_pose['posed_joints']-expected_pose['posed_joints']).norm(dim=-1).max())
                tokens_equal = torch.equal(actual['tokens'][:, :tokens],expected['tokens'][:, :tokens])
                if not tokens_equal or error > fk_tolerance_m:
                    count = int((actual['tokens'][:, :tokens] != expected['tokens'][:, :tokens]).sum())
                    raise ValueError(f'Compiled/native mismatch: probe={len(self.validation)}, '
                        f'changed_tokens={count}, fk_max_m={error:.9g}')
                if controller is not None:
                    for key in ('sdf_points', 'sdf_distance', 'sdf_valid'):
                        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
                motion = expected['motion'][:, 4:12].contiguous()
                compiled_geo, native_geo = self._geometry(motion), geometry(motion)
                for key, value in native_geo.items():
                    torch.testing.assert_close(compiled_geo[key], value, atol=1e-5, rtol=1e-5)
                # The browser will use the asset's full LBS weights. Compare
                # against the official full skin before enabling rig transport.
                reference_pose = inference.motion_rep.inverse(motion, False, return_all=False)
                reference_vertices = skinner.skin(reference_pose['global_joint_rots'][0],
                    reference_pose['posed_joints'][0], rot_is_global=True)
                torch.testing.assert_close(native_lbs(compiled_geo['rig']), reference_vertices,
                                           atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(self._observe(history)['history'], observe(history), atol=1e-5, rtol=1e-5)
                self.validation.append({'tokens_equal':tokens_equal,'fk_max_error_m':error,'num_tokens':tokens,
                    'fk_tolerance_m':fk_tolerance_m,'independent_validation_seed':independent_validation_seeds})
        finally:
            inference._vqvae_pose_model.forward_decoder = native_decode
            skinner.lbs = native_lbs

    def observe(self, frames):
        if frames.shape != self.history_shape or not torch.isfinite(frames).all():
            raise ValueError('Compiled observer requires four finite played frames')
        result = self._observe(frames.contiguous())
        self._validate_observed(result)
        return result['history']

    @torch.inference_mode()
    def keep_ready(self):
        """Keep the active viewer's GPU execution warm; discard every result.

        No backend/session state, published frames, or observed history is read
        or updated. Immutable outputs were cloned before any subsequent replay.
        This runs only on the same serialized inference worker as real plans.
        """
        self._prediction.graph.replay()

    @staticmethod
    def _validate_observed(result):
        # Tracing specializes Python guards. Keep their tensor results alive
        # in the graph and enforce the original FK round-trip bound per call.
        valid = (torch.isfinite(result['history']).all()
                 & (result['position_error'] <= 2e-5) & (result['rotation_error'] <= 2e-5))
        if not valid:
            raise ValueError('Observed-history re-encoding changed the displayed FK state')

    def geometry(self, frames):
        if frames.shape != (1, 8, self.history_shape[2]) or not torch.isfinite(frames).all():
            raise ValueError('Compiled geometry requires eight finite native frames')
        result = self._geometry(frames.contiguous())
        self._validate_observed(result)
        return result

    def predict(self,history,goal,diagnostics,*,num_tokens=8,query_controls=None,path_refinements=0):
        if (history.shape != self.history_shape or goal.shape != (1, 4)
                or not torch.isfinite(history).all() or not torch.isfinite(goal).all()):
            raise ValueError('Compiled prediction requires a finite four-frame history and planar goal')
        if self.flexible:
            if not isinstance(num_tokens,int) or not 6<=num_tokens<=16:
                raise ValueError('Compiled prediction horizon must be 6..16 tokens')
            query = history.new_tensor([[-1.,.6,1.,.4]]) if query_controls is None else query_controls
            controls = torch.cat((history.new_tensor([[num_tokens]]),query),-1)
            route=history[:,-1:,[0,2]].expand(-1,56,-1).clone()
            route_valid=torch.zeros((1,56),device=history.device,dtype=torch.bool)
            result=self._prediction(history.contiguous(),goal.contiguous(),controls.contiguous(),route,route_valid)
            for _ in range(path_refinements):
                route=result['motion'][:,4:60,[0,2]].detach().clone().contiguous()
                route_valid=(torch.arange(56,device=history.device)[None]<num_tokens*4-8)
                result=self._prediction(history.contiguous(),goal.contiguous(),controls.contiguous(),route,route_valid)
        else:
            if self.controller is not None and query_controls is not None:
                expected_query = history.new_tensor([[-1., .6, 1., .4]])
                if not torch.equal(query_controls, expected_query):
                    raise ValueError('Compiled unified query controls differ')
            elif query_controls is not None:
                raise ValueError('This fixed graph does not take custom query controls')
            if num_tokens!=8 or path_refinements:
                raise ValueError('This graph is bound to the original observation contract')
            result = self._prediction(history.contiguous(),goal.contiguous())
        return self._describe_prediction(history,goal,diagnostics,result,num_tokens,
            path_refinements=path_refinements,route=route if self.flexible else None,
            route_valid=route_valid if self.flexible else None)

    def _describe_prediction(self,history,goal,diagnostics,result,num_tokens,*,path_refinements=0,
                             route=None,route_valid=None):
        valid = torch.isfinite(result['motion']).all() & (result['count']==num_tokens).all()
        if not valid:
            raise ValueError('Compiled prediction violated its requested horizon')
        predicted_root = self._root_world(history,result)
        stop = num_tokens*4-4
        values = [result['points'], result['valid'].float(), result['world_query_xz'],
                  history[..., :5], goal, result['motion'][:,4:stop,:3], predicted_root[:,4:stop,:3]]
        if self.flexible:
            values += [result['survey_points'],result['survey_valid'].float(),
                       result['survey_world_query_xz'],result['actor_query_controls']]
        flat = torch.cat([value.reshape(-1) for value in values]).cpu()
        restored, start = [], 0
        for value in values:
            restored.append(flat[start:start+value.numel()].reshape(value.shape))
            start += value.numel()
        points,valid_points,query,root,display_goal,decoded,root_stage=restored[:7]
        observation = LocalTerrainObservation(points, valid_points.bool(), query)
        diagnostics.update(intent_diagnostics(observation,root,display_goal,num_tokens,self.protocol_version))
        if self.flexible:
            surveyed,known,survey_query,actor_controls=restored[7:]
            survey=LocalTerrainObservation(surveyed,known.bool(),survey_query)
            add_survey_diagnostics(diagnostics,survey,root,display_goal,num_tokens,self.protocol_version,actor_controls)
        diagnostics['decoded_world_xyz_m'] = decoded[0].tolist()
        diagnostics['root_stage_world_xyz_m'] = root_stage[0].tolist()
        diagnostics['terrain_sampling']='predicted_decoded_path_corridor' if path_refinements else 'input_direction'
        diagnostics['path_refinements']=path_refinements
        if self.controller is not None:
            from .unified_terrain import sdf_observation_debug
            diagnostics['unified_sdf'] = sdf_observation_debug(
                result['sdf_points'], result['sdf_distance'], result['sdf_valid'], history)
        if path_refinements:
            diagnostics['query_proposal_world_xz_m']=route[0,route_valid[0]].cpu().tolist()
        return result['motion'], result['count']

    def _root_world(self,history,result):
        predicted_root=self.inference.global_motion_rep.unnormalize(result['root']).clone()
        if self.inference._pose_model.args.get('vertical_frame'):
            from .native_mb_vertical_frame import TERRAIN_VERTICAL_FRAME,observed_terrain_origin
            if self.inference._pose_model.args['vertical_frame']==TERRAIN_VERTICAL_FRAME:
                origin=observed_terrain_origin(history[:,3,1],result)
            else:origin=history[:,3,1]-self.inference.global_motion_rep.stats.mean[1].to(history)
            predicted_root[...,1]+=origin[:,None]
        predicted_root[...,[0,2]]+=history[:,:1,[0,2]]
        return predicted_root

    def supports(self, controls):
        if self.controller is None:
            return True
        from .native_preserving_terrain import PROFILE
        return controls is not None and all(controls.get(key) == value for key, value in PROFILE.items()
                                           if key != 'refresh_frames')
