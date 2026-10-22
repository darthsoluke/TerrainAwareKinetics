"""Single native generation chain with always-on shared scene conditioning."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel
from .native_mb_online_conditions import VELOCITY_PROTOCOL
from .native_mb_online_live import NativeTerrainOnlineBackend, online_command_goal, predict_planar_terrain
from .native_preserving_terrain import PROFILE, history_arguments
from .terrain_decoder_pilot import training_targets
from .terrain_model_v3 import straight_through_codes
from .terrain_root_supervision import factual_history_root_targets
from .unified_scene_field import VolumetricEgoSampler, field_from_heightfield
from .unified_training_goals import validate_source_terminal_goal


SCHEMA = 'motionbricks.unified_terrain.v1'
IMPLEMENTATION = 'single-chain-always-conditioned-v1'


def sdf_observation_debug(points, distance, valid, root):
    """Serialize generation-time inputs, never re-query a newer observation."""
    return dict(version='unified-sdf-observation-v1', role='model_input',
                region_names=['body', 'requested_direction', 'actual_velocity'],
                points_per_region=points.shape[-2],
                world_xyz_m=points[0].reshape(-1, 3).detach().cpu().tolist(),
                signed_distance_m=distance[0].reshape(-1).detach().cpu().tolist(),
                valid=valid[0].reshape(-1).detach().cpu().tolist(),
                root_world_xyz_m=root[0, -1, :3].detach().cpu().tolist())


class SharedTerrainEncoder(nn.Module):
    def __init__(self, grid_size, width=128):
        super().__init__()
        self.grid_size = grid_size
        self.point = nn.Sequential(nn.Linear(15, 64), nn.GELU())
        self.output = nn.Sequential(nn.Linear(64+15+15+1, width), nn.GELU(), nn.Linear(width, width))

    def group(self, value):
        batch, regions, _, channels = value.shape
        side = self.grid_size//2
        value = value.reshape(batch, regions, 2, side, 2, side, 2, side, channels)
        return value.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).reshape(batch, regions*8, side**3, channels)

    def forward(self, observation):
        features = self.group(observation['features'])
        valid = self.group(observation['valid'][..., None]).squeeze(-1)
        distance = self.group(observation['distance'][..., None]).squeeze(-1)
        available = valid.any(-1)
        embedded = self.point(features).masked_fill(~valid[..., None], -torch.inf).amax(-2)
        embedded = torch.where(available[..., None], embedded, 0)

        def pick(score):
            index = score.masked_fill(~valid, torch.inf).argmin(-1)
            return features.gather(-2, index[..., None, None].expand(*index.shape, 1, 15)).squeeze(-2)

        # Preserve geometry together with its location, not an unlocated minimum.
        combined = torch.cat((embedded, pick(distance), pick(distance.abs()), valid.float().mean(-1, keepdim=True)), -1)
        tokens = self.output(combined)
        return torch.where(available[..., None], tokens, 0), available


class UnifiedSceneResidual(nn.Module):
    def __init__(self, width, time_query=False):
        super().__init__()
        self.query = nn.Linear(width, 128)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.output = nn.Linear(128, width, bias=False)
        self.time_query = nn.Linear(3, 128, bias=False) if time_query else None
        nn.init.zeros_(self.output.weight)

    def forward(self, value, memory, valid, time=None):
        if not valid.any(-1).all():
            raise ValueError('Attention requires known scene tokens in every row')
        query = self.query(value)
        if self.time_query is not None:
            if time is None or time.shape != value.shape[:2]:
                raise ValueError('Decoder queries need explicit padded-token times')
            query = query+self.time_query(torch.stack((time, torch.sin(time), torch.cos(time)), -1))
        delta, _ = self.attention(query, memory, memory, key_padding_mask=~valid, need_weights=False)
        return value+self.output(delta)


class UnifiedTerrainController(nn.Module):
    def __init__(self, inference, bundle, sampling=None, *, pose_condition_stage='input'):
        super().__init__()
        if pose_condition_stage not in ('input', 'output'):
            raise ValueError('Pose scene conditioning must use a declared input/output stage')
        self.pose_condition_stage = pose_condition_stage
        if inference._root_model.args.get('online_conditioning') != VELOCITY_PROTOCOL['version']:
            raise ValueError('Approved native velocity protocol required')
        if getattr(inference, '_terrain_execution', None) is not None:
            raise ValueError('A prior compiled graph must not bypass the new modules')
        if getattr(inference, '_unified_terrain_owner', None) is not None:
            raise ValueError('Inference already has a unified controller')
        object.__setattr__(self, 'inference', inference)
        object.__setattr__(inference, '_unified_terrain_owner', self)
        self.root = inference._root_model.backbone_net
        self.pose = inference._pose_model.backbone_net
        self.codec = inference._vqvae_pose_model
        inference.eval().requires_grad_(False)
        self.sampler = VolumetricEgoSampler(sampling)
        self.adaptation = nn.ModuleDict(dict(
            encoder=SharedTerrainEncoder(self.sampler.config.grid_size),
            root_stage1=UnifiedSceneResidual(self.root.args['n_embd']),
            root_stage2=UnifiedSceneResidual(self.root.args['n_embd']),
            pose=UnifiedSceneResidual(self.pose._args['n_embd']),
            decoder=UnifiedSceneResidual(self.codec.code_dim, time_query=True),
            height_encoder=nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128)),
        ))
        self.register_buffer('codebook', self.codec.get_codebook().detach().clone(), persistent=False)
        object.__setattr__(self, 'loss_model', MotionVQVAEModel(self.codec, None, inference.motion_rep,
            optimizer=None, scheduler=None, args=deepcopy(bundle['tokenizer_config']['model']['args'])))
        self._native_predict = inference.predict
        self._native_dequantize = self.codec.quantizer.dequantize
        self._context = None
        self._adapting = self._straight_through = False
        self._fields = OrderedDict()
        self._handles = []
        self.enabled = True
        self.native_calls = self.environment_calls = 0
        inference.predict = self._predict
        self.codec.quantizer.dequantize = self._dequantize
        for name, transformer in (('root_stage1', self.root._shared_transformer_model),
                                  ('root_stage2', self.root._root_token_transformer_model)):
            if transformer is not None:
                self._handles.append(transformer.register_forward_pre_hook(self._stage_hook(name)))
        self._handles.append(self.root.register_forward_hook(self._capture_root))
        self._handles.append(self.pose.register_forward_pre_hook(self._capture_pose_request, with_kwargs=True))
        self._handles.append(self.pose.register_forward_hook(self._capture_pose))
        pose_stage = self.pose._transformer_model if pose_condition_stage == 'input' else self.pose._proj_pose_output_logit
        self._handles.append(pose_stage.register_forward_pre_hook(self._stage_hook('pose')))
        self._handles.append(self.codec.decoder.register_forward_pre_hook(self._decoder_hook))
        self._handles.append(self.codec.decoder.register_forward_hook(self._capture_local))
        self.to(inference.device)
        self.eval()

    @contextmanager
    def context(self, history, scene, *, training=False, straight_through=False, volume_observation=None):
        if self._context is not None:
            raise ValueError('Controller contexts must not overlap')
        if volume_observation is not None and (training or not torch.is_inference_mode_enabled()):
            raise ValueError('Prepared volume observations are inference-only')
        self._context = dict(history=history, scene=scene, training=training, straight_through=straight_through,
                             volume_observation=volume_observation)
        try:
            yield
        finally:
            self._context = None
            self._adapting = self._straight_through = False

    def _stage_hook(self, name):
        def hook(_module, args):
            if not self._adapting:
                return args
            return (self.adaptation[name](args[0], self.memory, self.valid), *args[1:])
        return hook

    def _capture_root(self, _module, _args, output):
        self.last_root = output

    def _capture_pose_request(self, _module, args, kwargs):
        self.last_pose_request = (args, kwargs)

    def _capture_pose(self, _module, _args, output):
        self.last_logits = output['pose_logits']

    def _capture_local(self, _module, _args, output):
        self.last_local = output.transpose(1, 2)

    def _dequantize(self, *args, **kwargs):
        hard = self._native_dequantize(*args, **kwargs)
        if self._straight_through and torch.is_grad_enabled():
            return straight_through_codes(self.last_logits, hard, self.codebook)
        return hard

    def _decoder_hook(self, _module, args):
        if not self._adapting:
            return args
        root = self.inference.global_motion_rep.unnormalize(self.last_root['pred_global_root_values'])
        # Relative height cancels the native vertical origin; it is never a world-space query.
        height = root[:, :self.frames, 1]
        delta = height-height[:, 3:4]
        vertical_speed = torch.cat((torch.zeros_like(delta[:, :1]), torch.diff(delta, dim=1)*30), 1)
        time = (torch.arange(self.frames, device=height.device, dtype=height.dtype)-3)/30
        features = torch.stack((time[None].expand_as(delta), torch.asinh(delta), torch.asinh(vertical_speed)), -1)
        memory = torch.cat((self.memory, self.adaptation['height_encoder'](features)), 1)
        valid = torch.cat((self.valid, torch.ones_like(height, dtype=torch.bool)), 1)
        value = args[0].transpose(1, 2)
        token_time = (torch.arange(value.shape[1], device=value.device, dtype=value.dtype)*4+1.5-3)/30
        result = self.adaptation['decoder'](value, memory, valid, token_time[None].expand(value.shape[:2]))
        return (result.transpose(1, 2), *args[1:])

    def _predict(self, *args, **kwargs):
        if not self.enabled:
            return self._native_predict(*args, **kwargs)
        if self._context is None:
            raise ValueError('Factual history context required')
        history, scene = self._context['history'], self._context['scene']
        if len(history) != 1:
            raise ValueError('The native controller currently supports one character per call')
        tokens = int(args[6]) if isinstance(args[6], int) else int(args[6].item())
        self.frames = tokens*4
        self.observation = kwargs['terrain_observation']
        requested = args[2][:, 7, 1:3]
        self.volume_observation = self._context['volume_observation']
        if self.volume_observation is None:
            self.volume_observation = self.sampler.observe(history[..., :5], requested, self.scene_field(scene),
                                                           (self.frames-4)/30)
        with torch.set_grad_enabled(self._context['training']):
            self.memory, self.valid = self.adaptation['encoder'](self.volume_observation)
            self.environment_calls += 1
            modified = history_arguments(self.inference, args, history) if self._context['training'] and history.requires_grad else args
            self._adapting = True
            self._straight_through = self._context['straight_through']
            options = dict(kwargs, config=dict(kwargs.get('config', {}), cache_terrain_encoder=not self._context['training']))
            try:
                self.native_calls += 1
                return self._native_predict(*modified, **options)
            finally:
                self._adapting = self._straight_through = False

    def scene_field(self, scene):
        key = id(scene['heightfield_y_m'])
        if key not in self._fields:
            self._fields[key] = (scene, field_from_heightfield(scene))
        self._fields.move_to_end(key)
        while len(self._fields) > 16:
            self._fields.popitem(last=False)
        return self._fields[key][1]

    def forward(self, history, controls, scene, *, training=False, straight_through=False):
        if controls.shape != (1, 4):
            raise ValueError('One world VX/VZ and cos/sin heading command required')
        goal = online_command_goal(history[..., :5], controls[:, :2],
            torch.atan2(controls[:, 3], controls[:, 2]), 8, VELOCITY_PROTOCOL['version'])
        with self.context(history, scene, training=training, straight_through=straight_through):
            world, count = predict_planar_terrain(self.inference, history, goal, scene, num_tokens=8,
                config=dict(num_inference_step=4, pose_token_sampling_use_argmax=True, cache_terrain_encoder=not training),
                query_controls=history.new_tensor([[-1., .6, 1., .4]]))
        if int(count.item()) != 8:
            raise ValueError('Planning horizon changed')
        return dict(world=world[:, :32], local_prediction=self.last_local[:, :32], root_output=self.last_root,
                    pose_logits=self.last_logits, observation=self.observation,
                    input_controls=controls.detach().clone(), goal_protocol=VELOCITY_PROTOCOL['version'])

    def native_losses(self, result, source, history, *, legacy_proxy_control_arm=False):
        # This bounded fit accepts source history only. It does not invent recovery targets.
        if not torch.equal(history.detach(), source[:, :4].detach()):
            raise ValueError('Source-history objective cannot supervise off-source generated history')
        if result.get('goal_protocol') != VELOCITY_PROTOCOL['version'] or 'input_controls' not in result:
            raise ValueError('Training requires the actual forward velocity controls and protocol')
        # Only the explicitly recorded negative-control arm may retain the old labels.
        if not legacy_proxy_control_arm:
            validate_source_terminal_goal(self.inference.global_motion_rep.extract_root(source), result['input_controls'])
        target = training_targets(self.inference, source, history, result['observation'])
        local = result['local_prediction']
        motion = self.loss_model.loss(target, dict(recon_state=local, l_commit=local.sum()*0, perplexity=0.), None)['loss']
        tokens = self.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False).detach()
        teacher = self.inference._pose_model
        batch = dict(groundtruth_pose_tokens=tokens, local_motions=target['local_motions'])
        keys = ('focused_token_mask', 'masked_token_mask', 'incorrect_token_mask', 'correct_token_mask', 'mask_percentage', 'incorrect_percentage')
        batch.update(zip(keys, teacher._get_token_masks(batch)))
        corrupted = teacher._generate_tokens_for_training(batch)
        args, kwargs = self.last_pose_request
        padded = torch.full_like(args[0], self.pose.POSE_MASK_ID)
        padded[:, :tokens.shape[1]] = corrupted
        self._adapting = True
        try:
            logits = self.pose(padded, *args[1:], **kwargs)['pose_logits'][:, :tokens.shape[1]]
        finally:
            self._adapting = False
        ce = teacher.loss(batch, dict(pose_logits=logits))['loss'] if batch['focused_token_mask'].any() else logits.sum()*0
        root_target = factual_history_root_targets(self.inference, target, history, result['observation'])
        root = self.inference._root_model.loss(root_target, result['root_output'])['loss']
        return dict(native_total=motion+root+ce, native_motion=motion, native_root=root, token_ce=ce)

    def close(self):
        self.inference.predict = self._native_predict
        self.codec.quantizer.dequantize = self._native_dequantize
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if getattr(self.inference, '_unified_terrain_owner', None) is self:
            delattr(self.inference, '_unified_terrain_owner')


class UnifiedTerrainBackend(NativeTerrainOnlineBackend):
    def __init__(self, controller, initial_history, scene):
        super().__init__(controller.inference, initial_history, scene)
        self.controller = controller

    @torch.inference_mode()
    def predict_plan(self, velocity_xz, heading):
        execution = self.execution
        if execution is not None and hasattr(execution, 'supports') and not execution.supports(self.runtime_controls):
            self.execution = None
        try:
            with self.controller.context(self.history, self.scene):
                result = super().predict_plan(velocity_xz, heading)
            if self.last_generation.get('neural_generation') and 'unified_sdf' not in result['debug']:
                observation = self.controller.volume_observation
                result['debug']['unified_sdf'] = sdf_observation_debug(
                    observation['points'], observation['distance'], observation['valid'], self.history)
            return result
        finally:
            self.execution = execution
