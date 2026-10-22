"""Terrain residuals around the unchanged native velocity-conditioned pipeline."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel
from .joint_sdf_observation import HeightfieldPrismSDF, joint_sdf_features
from .native_mb_online_conditions import VELOCITY_PROTOCOL
from .native_mb_online_live import NativeTerrainOnlineBackend, online_command_goal, predict_planar_terrain
from .terrain_decoder_pilot import training_targets
from .terrain_model_v3 import straight_through_codes
from .terrain_root_supervision import factual_history_root_targets


SCHEMA = 'motionbricks.native_preserving_terrain.v1'
IMPLEMENTATION = 'native-preserving-v2-flat-history-grad'
PROFILE = dict(num_tokens=8, inference_steps=4, lookahead_m=-1., lateral_halfwidth_m=.6,
               refresh_frames=24, lookahead_gain=1., lookahead_base_m=.4, path_refinements=0)


def history_arguments(inference, args, history):
    """Restore only factual-history derivatives discarded by the native packer."""
    modified = list(args)
    local = inference.motion_rep.dual_rep.global_to_local(history, False, False,
        lengths=torch.full((len(history),), 4, device=history.device, dtype=torch.long))
    modified[0] = args[0].clone()
    modified[0][:, :4] = extract_feature_from_motion_rep(history, inference.global_motion_rep, 'root')
    modified[2] = args[2].clone()
    modified[2][:, :3] = extract_feature_from_motion_rep(local, inference.local_motion_rep, 'root')[:, :3]
    modified[4] = args[4].clone()
    modified[4][:, :4] = extract_feature_from_motion_rep(local, inference.local_motion_rep, inference.EXTERNAL_POSE_FEATURE_MODE)
    return modified


class SceneResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, 128)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.output = nn.Linear(128, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, value, actual, plane, valid):
        if not valid.any():
            return value
        available = valid.any(-1)
        safe = valid.clone(); safe[:, 0] |= ~available
        query = self.query(value)
        a, _ = self.attention(query, actual, actual, key_padding_mask=~safe, need_weights=False)
        b, _ = self.attention(query, plane, plane, key_padding_mask=~safe, need_weights=False)
        return value+self.output(a-b)*available[:, None, None]


def terrain_contrast(observation, actual, plane, valid):
    """No scene-name routing: only differences from the observed support plane."""
    points, known = observation['points'], observation['valid'].bool()
    level = points[:, 0, 4, 4, 1]
    if not known[:, 0, 4, 4].all():
        raise ValueError('Current support plane is unknown')
    difference = (points[..., 1]-level[:, None, None, None]).abs()
    height_change = torch.where(known, difference, torch.zeros_like(difference)).flatten(1).amax(1)
    sdf_change = torch.where(valid[..., None], (actual-plane).abs(), torch.zeros_like(actual)).flatten(1).amax(1)
    # Ignore only floating-point roundoff, not centimetre-scale geometry.
    return torch.maximum(height_change, sdf_change)>1e-6


class NativePreservingTerrain(nn.Module):
    def __init__(self, inference, bundle):
        super().__init__()
        if inference._root_model.args.get('online_conditioning') != VELOCITY_PROTOCOL['version']:
            raise ValueError('The preserved native velocity controller is required')
        if getattr(inference, '_terrain_execution', None) is not None:
            raise ValueError('An old compiled graph must not bypass new terrain modules')
        object.__setattr__(self, 'inference', inference)
        self.root = inference._root_model.backbone_net
        self.pose = inference._pose_model.backbone_net
        self.codec = inference._vqvae_pose_model
        inference.eval().requires_grad_(False)
        self.adaptation = nn.ModuleDict(dict(
            encoder=nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, 128)),
            joint_identity=nn.Embedding(15, 128),
            goal=nn.Sequential(nn.Linear(133, 128), nn.GELU(), nn.Linear(128, 2)),
            root=SceneResidual(self.root.args['n_embd']),
            pose=SceneResidual(self.pose._args['n_embd']),
            decoder=SceneResidual(self.codec.code_dim),
            height_encoder=nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128)),
        ))
        nn.init.zeros_(self.adaptation['goal'][-1].weight)
        nn.init.zeros_(self.adaptation['goal'][-1].bias)
        self.register_buffer('codebook', self.codec.get_codebook().detach().clone(), persistent=False)
        object.__setattr__(self, 'loss_model', MotionVQVAEModel(self.codec, None, inference.motion_rep,
            optimizer=None, scheduler=None, args=deepcopy(bundle['tokenizer_config']['model']['args'])))
        self.enabled = True
        self.body_enabled = True
        self._context = None
        self._adapting = False
        self._straight_through = False
        self._fields = OrderedDict()
        self._handles = []
        self.last_root = self.last_logits = self.last_local = None
        self._native_predict = inference.predict
        self._native_dequantize = self.codec.quantizer.dequantize
        self._native_decode = self.codec.forward_decoder
        inference.predict = self._predict
        self.codec.quantizer.dequantize = self._dequantize
        self.codec.forward_decoder = self._decode
        for transformer in (self.root._shared_transformer_model, self.root._root_token_transformer_model):
            if transformer is not None:
                self._handles.append(transformer.register_forward_pre_hook(self._root_hook))
        self._handles.append(self.root.register_forward_hook(self._capture_root))
        self._handles.append(self.pose.register_forward_pre_hook(self._capture_pose_request, with_kwargs=True))
        self._handles.append(self.pose.register_forward_hook(self._capture_pose))
        self._handles.append(self.pose._transformer_model.register_forward_pre_hook(self._pose_hook))
        self._handles.append(self.codec.decoder.register_forward_pre_hook(self._decoder_hook))
        self.to(inference.device); self.eval()

    @contextmanager
    def context(self, history, scene, *, training=False, straight_through=False):
        if self._context is not None:
            raise ValueError('Controller calls must not overlap')
        self._context = dict(history=history, scene=scene, training=training, straight_through=straight_through)
        try:
            yield
        finally:
            self._context = None
            self._adapting = self._straight_through = False

    def _capture_root(self, _module, _args, output):
        self.last_root = output

    def _capture_pose_request(self, _module, args, kwargs):
        self.last_pose_request = (args, kwargs)

    def _capture_pose(self, _module, _args, output):
        self.last_logits = output['pose_logits']

    def _decode(self, *args, **kwargs):
        result = self._native_decode(*args, **kwargs)
        self.last_local = result['recon_state']
        return result

    def _dequantize(self, *args, **kwargs):
        hard = self._native_dequantize(*args, **kwargs)
        if self._straight_through and torch.is_grad_enabled():
            return straight_through_codes(self.last_logits, hard, self.codebook)
        return hard

    def _root_hook(self, _module, args):
        if not self._adapting:
            return args
        return (self.adaptation['root'](args[0], self.memory, self.plane_memory, self.valid), *args[1:])

    def _pose_hook(self, _module, args):
        if not self._adapting or not self.body_enabled:
            return args
        return (self.adaptation['pose'](args[0], self.memory, self.plane_memory, self.valid), *args[1:])

    def _decoder_hook(self, _module, args):
        if not self._adapting or not self.body_enabled:
            return args
        root = self.inference.global_motion_rep.unnormalize(self.last_root['pred_global_root_values'])
        height = root[:, :self.frames, 1]
        delta = height-height[:, 3:4]
        velocity = torch.cat((delta[:, :1]*0, torch.diff(delta, dim=1)*30), 1)
        time = (torch.arange(self.frames, device=height.device, dtype=height.dtype)-3)/30
        features = torch.stack((time[None].expand_as(delta), torch.asinh(delta), torch.asinh(velocity)), -1)
        predicted_height = self.adaptation['height_encoder'](features)
        actual = torch.cat((self.memory, predicted_height), 1)
        plane = torch.cat((self.plane_memory, predicted_height), 1)
        valid = torch.cat((self.valid, torch.ones_like(height, dtype=torch.bool)), 1)
        value = self.adaptation['decoder'](args[0].transpose(1, 2), actual, plane, valid)
        return (value.transpose(1, 2), *args[1:])

    def _predict(self, *args, **kwargs):
        if not self.enabled:
            return self._native_predict(*args, **kwargs)
        if self._context is None:
            raise ValueError('The native backend must supply its factual history context')
        history, scene = self._context['history'], self._context['scene']
        if len(history) != 1:
            raise ValueError('This bounded controller uses the actual single-character runtime')
        tokens = args[6]
        tokens = int(tokens) if isinstance(tokens, int) else int(tokens.item())
        self.frames = tokens*4
        observation = kwargs['terrain_observation']
        self.observation = observation
        self._adapting = self._straight_through = False
        with torch.no_grad():
            proposal, count = self._native_predict(*args, **kwargs)
            key = id(scene['heightfield_y_m'])
            if key not in self._fields:
                self._fields[key] = (scene, HeightfieldPrismSDF(scene))
            self._fields.move_to_end(key)
            while len(self._fields)>16:
                self._fields.popitem(last=False)
            field = self._fields[key][1]
            ground = history[:, 3, 1]+observation['points'][:, 0, 4, 4, 1]
            values, valid, plane = joint_sdf_features(self.inference.global_motion_rep, history,
                proposal[:, :self.frames], [scene], fields=[field], ground_y=ground)
            active = terrain_contrast(observation, values, plane, valid)
        self.last_active = bool(active.any())
        self.last_requested_goal = args[2][:, 7, 1:3].detach().clone()
        self.last_effective_goal = self.last_requested_goal.clone()
        if not self.last_active:
            # Return the actual original forecast, not a reconstructed copy or
            # a blended skeleton. Flat equality holds even after fitting.
            if self._context['training'] and history.requires_grad:
                with torch.enable_grad():
                    modified = history_arguments(self.inference, args, history)
                    options = dict(kwargs, config=dict(kwargs.get('config', {}), cache_terrain_encoder=False))
                    self._straight_through = self._context['straight_through']
                    try:
                        return self._native_predict(*modified, **options)
                    finally:
                        self._straight_through = False
            return proposal, count
        with torch.set_grad_enabled(self._context['training']):
            identity = self.adaptation['joint_identity'](torch.arange(15, device=history.device))[None, None]
            self.memory = (self.adaptation['encoder'](values)+identity).flatten(1, 2)
            self.plane_memory = (self.adaptation['encoder'](plane)+identity).flatten(1, 2)
            self.valid = valid.flatten(1, 2)
            weight = self.valid[..., None].to(self.memory)
            pooled = (self.memory*weight).sum(1)/weight.sum(1).clamp_min(1)
            pooled_plane = (self.plane_memory*weight).sum(1)/weight.sum(1).clamp_min(1)
            requested = args[2][:, 7, 1:3]
            command = torch.cat((torch.asinh(requested), args[0][:, 7, 3:5], history.new_full((1,1),(self.frames-4)/30)), -1)
            delta = self.adaptation['goal'](torch.cat((pooled, command), -1))-self.adaptation['goal'](torch.cat((pooled_plane, command), -1))
            effective = requested+delta
            modified = history_arguments(self.inference,args,history) if self._context['training'] and history.requires_grad else list(args)
            modified[2] = modified[2].clone()
            modified[2][:, 7, 1:3] = effective
            self.last_effective_goal = effective
            self._adapting = True
            self._straight_through = self._context['straight_through']
            options = dict(kwargs)
            options['config'] = dict(kwargs.get('config', {}), cache_terrain_encoder=not self._context['training'])
            try:
                return self._native_predict(*modified, **options)
            finally:
                self._adapting = self._straight_through = False

    def forward(self, history, controls, scene, *, training=False, straight_through=False):
        if controls.shape != (1,4):
            raise ValueError('Use one native VX/VZ/facing command')
        goal = online_command_goal(history[..., :5], controls[:, :2],
            torch.atan2(controls[:, 3], controls[:, 2]), 8, VELOCITY_PROTOCOL['version'])
        query = history.new_tensor([[-1., .6, 1., .4]])
        with self.context(history, scene, training=training, straight_through=straight_through):
            world, count = predict_planar_terrain(self.inference, history, goal, scene, num_tokens=8,
                config=dict(num_inference_step=4, pose_token_sampling_use_argmax=True, cache_terrain_encoder=not training),
                query_controls=query)
        if int(count.item()) != 8:
            raise ValueError('Explicit native planning horizon changed')
        return dict(world=world[:, :32], local_prediction=self.last_local[:, :32], root_output=self.last_root,
            pose_logits=self.last_logits, observation=self.observation, terrain_active=self.last_active)

    def native_losses(self, result, source, history):
        target = training_targets(self.inference, source, history, result['observation'])
        local = result['local_prediction']
        motion = self.loss_model.loss(target, dict(recon_state=local, l_commit=local.sum()*0, perplexity=0.), None)['loss']
        tokens = self.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False).detach()
        teacher = self.inference._pose_model
        batch = dict(groundtruth_pose_tokens=tokens, local_motions=target['local_motions'])
        keys = ('focused_token_mask','masked_token_mask','incorrect_token_mask','correct_token_mask','mask_percentage','incorrect_percentage')
        batch.update(zip(keys, teacher._get_token_masks(batch)))
        corrupted = teacher._generate_tokens_for_training(batch)
        args, kwargs = self.last_pose_request
        padded = torch.full_like(args[0], self.pose.POSE_MASK_ID)
        padded[:, :tokens.shape[1]] = corrupted
        self._adapting = result['terrain_active']
        try:
            logits = self.pose(padded, *args[1:], **kwargs)['pose_logits'][:, :tokens.shape[1]]
        finally:
            self._adapting = False
        ce = teacher.loss(batch, dict(pose_logits=logits))['loss'] if batch['focused_token_mask'].any() else logits.sum()*0
        root_target = factual_history_root_targets(self.inference, target, history, result['observation'])
        root = self.inference._root_model.loss(root_target, result['root_output'])['loss']
        return dict(native_total=motion+root+ce, native_motion=motion, native_root=root, token_ce=ce)


class NativePreservingBackend(NativeTerrainOnlineBackend):
    def __init__(self, controller, initial_history, scene):
        super().__init__(controller.inference, initial_history, scene)
        if self.execution is not None:
            raise ValueError('This new terrain extension has not been compiled')
        self.controller = controller

    @torch.inference_mode()
    def predict_plan(self, velocity_xz, heading):
        with self.controller.context(self.history, self.scene):
            return super().predict_plan(velocity_xz, heading)
