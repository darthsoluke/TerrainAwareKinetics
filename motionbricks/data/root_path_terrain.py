"""Post-Root terrain observations for the independent local control experiment."""

import torch
import torch.nn.functional as F

from .local_terrain_observation import extract_local_terrain_observation
from .native_mb_observed_history import reencode_played_history
from .native_mb_online_live import predict_planar_terrain
from .native_mb_terrain_protocol import terrain_observation_config
from .soma_cuda_graph import StaticCudaCall
from .soma_static_execution import trace_fixed_call, restore_native_transformers
from .soma_terrain_execution import TerrainExecution, SETTINGS


POLICY = 'motionbricks.predicted_root_terrain.v1'


class PredictedRootObservation:
    def __init__(self, controller):
        self.controller = controller
        self.context = self.height = self.path = None
        self.deferred = False
        controller._handles.extend((
            controller.root.register_forward_hook(self._after_root),
            controller.pose.register_forward_pre_hook(self._before_pose, with_kwargs=True),
        ))

    def _curve(self, path, count, history):
        centers = F.interpolate(path.transpose(1, 2), size=count,
                                mode='linear', align_corners=True).transpose(1, 2)
        delta = torch.cat((centers[:, 1:2]-centers[:, :1],
                           centers[:, 2:]-centers[:, :-2],
                           centers[:, -1:]-centers[:, -2:-1]), 1)
        length = delta.norm(dim=-1, keepdim=True)
        heading = history[:, -1, 3:5]
        forward = torch.stack((heading[:, 1], heading[:, 0]), -1)[:, None]
        tangent = torch.where(length > 1e-5, delta/length.clamp_min(1e-5), forward)
        right = torch.stack((tangent[..., 1], -tangent[..., 0]), -1)
        return centers, right

    @torch.no_grad()
    def _after_root(self, _module, _args, output):
        controller = self.controller
        context = controller._context
        if self.deferred or context is None or not controller._adapting or context['training']:
            return
        history, scene = context['history'], context['scene']
        volume, self.height, self.path = self.sample(history, scene, output['pred_global_root_values'],
                                                   controller.volume_observation, controller.observation)
        controller.memory, controller.valid = controller.adaptation['encoder'](volume)
        controller.volume_observation = volume
        self.context = context

    def sample(self, history, scene, normalized_root, coarse_volume, observation, *, reuse_static_regions=False,
               corridor_stations=None):
        controller = self.controller
        root = controller.inference.global_motion_rep.unnormalize(normalized_root)
        world_xz = root[:, :controller.frames, [0, 2]] + history[:, :1, [0, 2]]
        # Frame3 is the factual boundary; frames4..T-5 are the playable proposal.
        path = torch.cat((history[:, -1:, [0, 2]], world_xz[:, 4:controller.frames-4]), 1)
        sampler = controller.sampler
        side = sampler.config.grid_size
        centers, right = self._curve(path, side, history) if corridor_stations is None else corridor_stations
        points = coarse_volume['points'].clone()
        ids = torch.arange(side**3, device=history.device)
        lateral = torch.linspace(-sampler.config.corridor_halfwidth_m,
                                 sampler.config.corridor_halfwidth_m, side,
                                 device=history.device, dtype=history.dtype)
        xz = centers[:, ids % side] + right[:, ids % side]*lateral[ids//(side*side)][None, :, None]
        points[:, 1, :, 0], points[:, 1, :, 2] = xz[..., 0], xz[..., 1]
        field = controller.scene_field(scene)
        samples = None
        if reuse_static_regions:
            changed = field.query(points[:, 1:2])
            samples = tuple(torch.cat((coarse_volume[key][:, :1], value, coarse_volume[key][:, 2:]), 1)
                            for key, value in zip(('distance', 'normal', 'valid'), changed))
        volume = sampler.observe_points(history[..., :5], points, field, samples=samples)
        coarse = observation['points'][:, 1, ..., [0, 2]]
        extent = coarse[:, -1, coarse.shape[2]//2].norm(dim=-1)
        width = (coarse[:, 0, -1]-coarse[:, 0, 0]).norm(dim=-1)*.5
        config = terrain_observation_config(controller.inference._root_model.args['observation_range_policy'])
        # Recover the original sensor bounds after a rotated FP32 norm. The
        # trained minimum can otherwise become 0.79999995 and fail validation.
        extent = extent.clamp((config.patch_height-1)*config.near_spacing_m,
            config.forward_start_m+(config.patch_height-1)*config.forward_spacing_m)
        controls = torch.stack((extent, width, torch.ones_like(extent), torch.full_like(extent, .4)), -1)
        displacement = path[:, -1]-history[:, -1, [0, 2]]
        heading = history[:, -1, 3:5]
        direction = torch.stack((heading[:, 0]*displacement[:, 0]-heading[:, 1]*displacement[:, 1],
                                 heading[:, 1]*displacement[:, 0]+heading[:, 0]*displacement[:, 1]), -1)
        height = extract_local_terrain_observation(scene['heightfield_y_m'],
            scene['min_point_world_xyz_m'], scene['dx_m'], history[:, -1, :3],
            torch.atan2(heading[:, 1], heading[:, 0]), direction,
            heightfield_valid=scene.get('heightfield_valid'), config=config, query_controls=controls,
            query_path_world_xz=path[:, 1:], query_path_valid=torch.ones(path[:, 1:].shape[:-1],
                                                                      device=history.device, dtype=torch.bool))
        return volume, height, path

    def _before_pose(self, _module, args, kwargs):
        if self.controller._adapting and self.context is self.controller._context:
            return args, dict(kwargs, terrain_observation=self.height.as_model_inputs())


class RootPathTerrainExecution(TerrainExecution):
    """One native Root pass, path queries, then native Pose/decoder; compiled FK."""

    def __init__(self, inference, scene, skinner, seed, *, controller):
        self.inference, self.controller, self.skinner = inference, controller, skinner
        self.flexible = False
        self.neural_execution = 'native'
        self.protocol_version = inference._root_model.args['online_conditioning']
        self.history_shape = tuple(seed.shape)
        self.observer = PredictedRootObservation(controller)
        self._geometry = self._observe = None
        restore_native_transformers(inference)
        self.bind_scene(scene, seed)

    @torch.inference_mode()
    def bind_scene(self, scene, seed):
        self.scene, self.field = scene, self.controller.scene_field(scene)
        if self._geometry is not None:
            return
        captured = {}
        native_lbs = self.skinner.lbs

        def lbs(transforms):
            captured['rig'] = transforms
            return native_lbs(transforms)

        def observe(motion):
            checks = {}
            history = reencode_played_history(motion, self.inference.global_motion_rep, tensor_checks=checks)
            return dict(history=history, **checks)

        def geometry(motion):
            pose = self.inference.motion_rep.inverse(motion, False, return_all=False)
            vertices = self.skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
            return dict(rig=captured['rig'], joints=pose['posed_joints'][0],
                        vertices=vertices, **observe(motion[:, -4:]))

        self.skinner.lbs = lbs
        try:
            sample = seed.repeat(1, 2, 1).contiguous()
            expected = geometry(sample)
            self._geometry = StaticCudaCall(trace_fixed_call(geometry, (sample,), cuda_constants=True))
            self._observe = StaticCudaCall(trace_fixed_call(observe, (seed,), cuda_constants=True))
            actual = self._geometry(sample)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(native_lbs(actual['rig']), expected['vertices'], atol=1e-5, rtol=1e-5)
            self.validation = [dict(neural_execution='native', geometry_only=True, geometry_verified=True,
                                   rig_max_error_m=float((actual['vertices']-expected['vertices']).abs().max()),
                                   observation_policy=POLICY)]
        finally:
            self.skinner.lbs = native_lbs

    def _prediction(self, history, goal, *, num_tokens=8, query_controls=None):
        tensors = {}
        motion, count = predict_planar_terrain(self.inference, history, goal, self.scene,
            num_tokens=num_tokens, config=SETTINGS, tensor_diagnostics=tensors,
            query_controls=(history.new_tensor([[-1., .6, 1., .4]]) if query_controls is None else query_controls))
        height = self.observer.height
        volume = self.controller.volume_observation
        tensors.update(points=height.points, valid=height.valid, world_query_xz=height.world_query_xz,
                       sdf_points=volume['points'], sdf_distance=volume['distance'], sdf_valid=volume['valid'])
        return dict(motion=motion, count=count, **tensors)

    @torch.inference_mode()
    def predict(self, history, goal, diagnostics, **kwargs):
        motion, count = super().predict(history, goal, diagnostics, **kwargs)
        diagnostics.update(terrain_sampling=POLICY, root_neural_passes=1,
                           query_proposal_world_xz_m=self.observer.path[0].cpu().tolist())
        diagnostics['unified_sdf']['region_names'] = ['body', 'predicted_root_path', 'actual_velocity']
        return motion, count

    def geometry(self, frames):
        result = super().geometry(frames)
        distance, _, valid = self.field.query(result['joints'])
        result.update(bone_sdf_distance=distance, bone_sdf_valid=valid)
        result.pop('vertices')
        return result

    def keep_ready(self):
        return None
