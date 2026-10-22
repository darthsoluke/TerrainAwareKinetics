"""One verified model graph with current-scene observations as explicit inputs."""

import torch

from .local_terrain_observation import LocalTerrainObservation, extract_local_terrain_observation, trained_query_controls
from .native_mb_online_conditions import VELOCITY_PROTOCOL, velocity_direction_local
from .native_mb_online_live import online_command_goal, predict_planar_terrain
from .native_mb_observed_history import reencode_played_history
from .native_mb_terrain_protocol import terrain_observation_config
from .soma_cuda_graph import StaticCudaCall
from .soma_static_execution import trace_fixed_call, use_dense_transformers, restore_native_transformers
from .soma_terrain_execution import TerrainExecution, SETTINGS


class ReusableTerrainExecution(TerrainExecution):
    """Scene queries stay native; only the scene-independent model is captured.

    A single serialized worker owns scene changes and replay. No map tensors,
    field handles, BVH pointers or source poses are embedded in the model graph.
    """

    @torch.inference_mode()
    def __init__(self, inference, scene, skinner, seed, *, controller):
        if inference._root_model.args['online_conditioning'] != VELOCITY_PROTOCOL['version']:
            raise ValueError('Reusable execution requires the fixed velocity contract')
        self.inference, self.controller, self.skinner = inference, controller, skinner
        self.flexible = False
        self.protocol_version = VELOCITY_PROTOCOL['version']
        self.history_shape = (1, 4, len(inference.global_motion_rep.indices['all']))
        self.scene = scene
        self.field = controller.scene_field(scene)
        self._compiled_prediction = self._geometry = self._observe = None
        self.bind_scene(scene, seed)

    def _scene_inputs(self, history, goal, num_tokens=8, query=None):
        root = history[..., :5]
        seconds=(4*num_tokens-4)/30
        direction = velocity_direction_local(root, goal, seconds)
        config = terrain_observation_config(self.inference._root_model.args['observation_range_policy'])
        query = history.new_tensor([[-1., .6, 1., .4]]) if query is None else query
        actor = trained_query_controls(query, direction, config)

        def sample(controls):
            return extract_local_terrain_observation(self.scene['heightfield_y_m'],
                self.scene['min_point_world_xyz_m'], self.scene['dx_m'], root[:, -1, :3],
                torch.atan2(root[:, -1, 4], root[:, -1, 3]), direction,
                heightfield_valid=self.scene.get('heightfield_valid'), config=config, query_controls=controls)

        observation, survey = sample(actor), sample(query)
        volume = self.controller.sampler.observe(root, goal[:, :2], self.field, seconds)
        # Every optional envelope is present for this declared fixed query profile.
        return (observation.points, observation.valid, observation.world_query_xz, observation.query_envelope_m,
                survey.points, survey.valid, survey.world_query_xz, survey.query_envelope_m,
                volume['points'], volume['distance'], volume['normal'], volume['valid'], volume['features'])

    def _prediction(self, history, goal):
        values = self._scene_inputs(history, goal)
        return self._compiled_prediction(history, goal, *values)

    @torch.inference_mode()
    def bind_scene(self, scene, seed):
        """Rebind and validate all five commands without retracing any model."""
        if seed.shape != self.history_shape or not seed.is_cuda:
            raise ValueError('Exactly four CUDA factual frames required')
        self.scene, self.field = scene, self.controller.scene_field(scene)
        inference, controller, skinner = self.inference, self.controller, self.skinner
        captured = {}
        native_decode, native_lbs = inference._vqvae_pose_model.forward_decoder, skinner.lbs

        def decode(ids, *args, **kwargs):
            captured['tokens'] = ids.clone()
            return native_decode(ids, *args, **kwargs)

        def lbs(transforms):
            captured['rig'] = transforms
            return native_lbs(transforms)

        def prediction(history, goal, *values):
            prepared = (LocalTerrainObservation(*values[:4]), LocalTerrainObservation(*values[4:8])) if values else None
            volume = dict(zip(('points', 'distance', 'normal', 'valid', 'features'), values[8:])) if values else None
            tensors = {}
            with controller.context(history, scene, volume_observation=volume):
                motion, count = predict_planar_terrain(inference, history, goal, scene, num_tokens=8,
                    config=SETTINGS, tensor_diagnostics=tensors, query_controls=history.new_tensor([[-1., .6, 1., .4]]),
                    prepared_observations=prepared)
                observed = controller.volume_observation
                tensors.update(sdf_points=observed['points'], sdf_distance=observed['distance'], sdf_valid=observed['valid'])
            return dict(motion=motion, count=count, tokens=captured['tokens'], **tensors)

        def geometry(motion):
            pose = inference.motion_rep.inverse(motion, False, return_all=False)
            skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
            checks = {}
            history = reencode_played_history(motion[:, -4:], inference.global_motion_rep, tensor_checks=checks)
            return dict(rig=captured['rig'], joints=pose['posed_joints'][0], history=history, **checks)

        def observe(motion):
            checks = {}
            history = reencode_played_history(motion, inference.global_motion_rep, tensor_checks=checks)
            return dict(history=history, **checks)

        inference._vqvae_pose_model.forward_decoder, skinner.lbs = decode, lbs
        try:
            restore_native_transformers(inference)
            cases = []
            for velocity, heading in (([0., 0.], 0.), ([0., 1.6], 0.), ([1.6, 0.], 1.57),
                                      ([0., -1.6], 3.14), ([-1.6, 0.], -1.57)):
                history = seed.clone()
                goal = online_command_goal(history[..., :5], velocity, heading, 8, self.protocol_version)
                expected = prediction(history, goal)
                values = self._scene_inputs(history, goal)
                prepared = prediction(history, goal, *values)
                for key in expected:
                    torch.testing.assert_close(prepared[key], expected[key], atol=0, rtol=0)
                cases.append(((history, goal, *values), expected))
            use_dense_transformers(inference)
            if self._compiled_prediction is None:
                self._compiled_prediction = StaticCudaCall(trace_fixed_call(prediction, cases[0][0], cuda_constants=True))
                sample = cases[0][1]['motion'][:, 4:12].contiguous()
                self._geometry = StaticCudaCall(trace_fixed_call(geometry, (sample,), cuda_constants=True))
                self._observe = StaticCudaCall(trace_fixed_call(observe, (seed,), cuda_constants=True))
            self.validation = []
            for inputs, expected in cases:
                actual = self._compiled_prediction(*inputs)
                expected_pose = inference.motion_rep.inverse(expected['motion'][:, :32], False, return_all=False)
                actual_pose = inference.motion_rep.inverse(actual['motion'][:, :32], False, return_all=False)
                error = float((actual_pose['posed_joints']-expected_pose['posed_joints']).norm(dim=-1).max())
                tokens_equal = torch.equal(actual['tokens'][:, :8], expected['tokens'][:, :8])
                if not tokens_equal or error > 3e-5:
                    raise ValueError(f'Reused graph/native mismatch: tokens_equal={tokens_equal}, fk={error}')
                for key in ('sdf_points', 'sdf_distance', 'sdf_valid', 'points', 'valid', 'world_query_xz'):
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                motion = expected['motion'][:, 4:12].contiguous()
                compiled, native = self._geometry(motion), geometry(motion)
                for key in native:
                    torch.testing.assert_close(compiled[key], native[key], atol=1e-5, rtol=1e-5)
                pose = inference.motion_rep.inverse(motion, False, return_all=False)
                vertices = skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
                torch.testing.assert_close(native_lbs(compiled['rig']), vertices, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(self._observe(inputs[0])['history'], observe(inputs[0])['history'], atol=1e-5, rtol=1e-5)
                self.validation.append(dict(tokens_equal=tokens_equal, fk_max_error_m=error, num_tokens=8,
                    fk_tolerance_m=3e-5, independent_validation_seed=True, scene_inputs_equal=True))
        finally:
            inference._vqvae_pose_model.forward_decoder, skinner.lbs = native_decode, native_lbs
            use_dense_transformers(inference)

    def geometry(self, frames):
        result = super().geometry(frames)
        distance, _, valid = self.field.query(result['joints'])
        result.update(bone_sdf_distance=distance, bone_sdf_valid=valid)
        return result

    @torch.inference_mode()
    def keep_ready(self):
        self._compiled_prediction.graph.replay()
