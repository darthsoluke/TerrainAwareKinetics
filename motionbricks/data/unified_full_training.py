"""Batched native MB training with the same causal input preparation as runtime."""

import torch
from torch import nn

from .native_mb_online_conditions import sparse_velocity_conditions
from .terrain_decoder_pilot import training_targets
from .terrain_root_supervision import factual_history_root_targets
from .unified_pose_training import pose_memory, pose_teacher
from .unified_training_goals import source_terminal_goal
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.probe_unified_native_codec import source_codec_context
from scripts.probe_unified_native_stages import fresh_kwargs


class CapturedPoseRequest(Exception):
    pass


@torch.no_grad()
def prepare_native_batch(controller, world, observation):
    """Execute native input preparation, stopping before learned Pose inference.

    The temporary Root output is a source-Root training teacher. Root's captured
    INPUTS remain causal and are used unchanged for actual Root optimization.
    """
    inference = controller.inference
    history = world[:, :4]
    count = len(world)
    command = source_terminal_goal(world[..., :5])
    root = extract_feature_from_motion_rep(history, inference.global_motion_rep, 'root')
    local = inference.motion_rep.dual_rep.global_to_local(history, False, False,
        lengths=torch.full((count,), 4, device=world.device, dtype=torch.long))
    velocity = extract_feature_from_motion_rep(local, inference.local_motion_rep, 'root')
    poses = extract_feature_from_motion_rep(local, inference.local_motion_rep, inference.EXTERNAL_POSE_FEATURE_MODE)
    conditions = sparse_velocity_conditions(root, velocity, poses, command)
    target = training_targets(inference, world, history, observation)
    root_target = factual_history_root_targets(inference, target, history, observation)
    captured = {}
    previous = controller.root.__dict__.get('forward')

    def root_teacher(*args, **kwargs):
        captured['root_args'], captured['root_kwargs'] = args, fresh_kwargs(kwargs)
        trajectory = world.new_zeros(count, inference._args['max_tokens']*4, 5)
        trajectory[:, :32] = target['global_motions'][..., :5]
        return dict(pred_global_root_values=trajectory,
            pred_num_tokens=torch.full((count,), 8, device=world.device, dtype=torch.long))

    def capture_pose(_module, args, kwargs):
        captured['pose_args'], captured['pose_kwargs'] = args, fresh_kwargs(kwargs)
        raise CapturedPoseRequest()

    controller.root.forward = root_teacher
    handle = controller.pose.register_forward_pre_hook(capture_pose, with_kwargs=True)
    try:
        try:
            controller._native_predict(conditions['global_root_values'], conditions['has_global_root_values'],
                conditions['local_root_values'], conditions['has_local_root_values'],
                conditions['local_poses'], conditions['has_local_poses'], 8,
                terrain_observation=observation,
                global_root_component_mask=conditions['global_root_component_mask'],
                local_root_component_mask=conditions['local_root_component_mask'],
                config=dict(num_inference_step=4, pose_token_sampling_use_argmax=True,
                    cache_terrain_encoder=False, allow_pred_out_of_reach_num_tokens=False))
        except CapturedPoseRequest:
            pass
    finally:
        handle.remove()
        if previous is None:
            del controller.root.forward
        else:
            controller.root.forward = previous
    if set(captured) != {'root_args', 'root_kwargs', 'pose_args', 'pose_kwargs'}:
        raise ValueError('Native training preparation did not reach both modules')
    raw_args, raw_kwargs = captured['root_args'], captured['root_kwargs']
    if raw_kwargs['global_root_component_mask'][:, 4:, 1].any() \
            or raw_kwargs['local_root_component_mask'][:, 3:, 3].any() or raw_args[5][:, 4:].any():
        raise ValueError('Future source height or pose leaked into Root inputs')
    args = captured['pose_args']
    captured['pose_args'] = (args[0][:, :8], args[1][:, :32], args[2][:, :32], args[3][:, :32], *args[4:])
    if 'root_height_trajectory' in captured['pose_kwargs']:
        captured['pose_kwargs']['root_height_trajectory'] = captured['pose_kwargs']['root_height_trajectory'][:, :32]
    target = {key: value for key, value in target.items() if key not in ('motion_frame_mask', 'motion_interval_mask')}
    return dict(captured, codec_target=target, root_target=root_target, command=command)


def rotate_volume_observation(observation, quarters, *, field_is_plane=False):
    """Rotate plane observations; non-plane closest-normal ties need re-query."""
    if not field_is_plane:
        raise ValueError('Non-plane observations must re-query rotated geometry to preserve normal tie conventions')
    features = observation['features'].clone()
    quarters = torch.as_tensor(quarters, device=features.device).remainder(4)
    for turn in (1, 2, 3):
        rows = quarters == turn
        for x_index, z_index in ((0, 2), (7, 9)):
            x, z = features[rows, ..., x_index].clone(), features[rows, ..., z_index].clone()
            features[rows, ..., x_index] = z if turn == 1 else -x if turn == 2 else -z
            features[rows, ..., z_index] = -x if turn == 1 else -z if turn == 2 else x
        cosine, sine = features[rows, ..., 10].clone(), features[rows, ..., 11].clone()
        features[rows, ..., 10] = -sine if turn == 1 else -cosine if turn == 2 else sine
        features[rows, ..., 11] = cosine if turn == 1 else -sine if turn == 2 else -cosine
    return dict(observation, features=features)


class NativeTrainingStage(nn.Module):
    def __init__(self, controller, stage, *, cache_pose_tokens=False, coalesce_dilated_bias=False,
                 graph_dilated_convolutions=False, isolate_pose_teacher_rng=False):
        super().__init__()
        if stage not in ('root', 'codec', 'pose'):
            raise ValueError('Unknown full training stage')
        self.controller, self.stage = controller, stage
        if isolate_pose_teacher_rng and stage != 'pose':
            raise ValueError('Device-local teacher RNG is only available for Pose')
        self.isolate_pose_teacher_rng = isolate_pose_teacher_rng
        if graph_dilated_convolutions and (stage != 'codec' or not coalesce_dilated_bias):
            raise ValueError('Convolution graphs require the verified coalesced codec stage')
        self.graph_dilated_convolutions = graph_dilated_convolutions
        self._dilated_graph_execution = None
        if graph_dilated_convolutions:
            from .dilated_graph_convolution import GraphConvolutionExecution
            self._dilated_graph_execution = GraphConvolutionExecution()
        if coalesce_dilated_bias and stage == 'pose' and not cache_pose_tokens:
            raise ValueError('Pose bias coalescing requires frozen-token memoization')
        self.coalesce_dilated_bias = coalesce_dilated_bias
        if cache_pose_tokens and stage != 'pose':
            raise ValueError('Only the frozen-codec Pose stage may memoize source tokens')
        self.cache_pose_tokens = cache_pose_tokens
        self._pose_tokens = {}
        self._pose_token_layouts = {}
        self.token_cache_hits = self.token_cache_misses = 0
        if not hasattr(controller, 'memory'):
            controller.memory, controller.valid = None, None
        if not hasattr(controller, 'frames'):
            controller.frames = 32
        controller.eval().requires_grad_(False)
        if stage == 'root':
            modules = [controller.root, *[controller.adaptation[name] for name in ('encoder', 'root_stage1', 'root_stage2')]]
        elif stage == 'codec':
            modules = [controller.codec, controller.adaptation['decoder'], controller.adaptation['height_encoder']]
        else:
            modules = [controller.pose, controller.adaptation['pose']]
        for module in modules:
            module.requires_grad_(True)
        object.__setattr__(self, '_active_modules', modules)

    def train(self, mode=True):
        self.training = mode
        self.controller.eval()
        for module in self._active_modules:
            module.train(mode)
        return self

    def forward(self, batch, seed=2333):
        if self.graph_dilated_convolutions:
            from .dilated_graph_context import graphed_dilated_convolutions
            with graphed_dilated_convolutions(self.controller, self._dilated_graph_execution):
                return self._forward_native(batch, seed)
        if self.coalesce_dilated_bias:
            if self.stage == 'pose' and (self.controller.codec.training or
                    any(parameter.requires_grad for parameter in self.controller.codec.parameters())):
                raise ValueError('Pose bias coalescing requires an eval-mode frozen codec')
            from .dilated_bias_convolution import coalesced_dilated_bias
            with coalesced_dilated_bias(self.controller):
                return self._forward_native(batch, seed)
        return self._forward_native(batch, seed)

    def _forward_native(self, batch, seed=2333):
        controller = self.controller
        prepared = prepare_native_batch(controller, batch['world'], batch['observation'])
        memory, valid = controller.adaptation['encoder'](batch['volume'])
        context = dict(memory=memory, valid=valid)
        if self.stage == 'root':
            with pose_memory(controller, context):
                output = controller.root(*prepared['root_args'], **fresh_kwargs(prepared['root_kwargs']),
                    groundtruth_num_tokens=prepared['root_target']['groundtruth_num_tokens'])
            return controller.inference._root_model.loss(prepared['root_target'], output)['loss']
        if self.stage == 'codec':
            target = prepared['codec_target']
            local = target['local_motions']
            external = extract_feature_from_motion_rep(local, controller.codec.motion_rep,
                controller.codec.decoder_external_cond_feature_mode)
            mask = torch.zeros(local.shape[:2], device=local.device, dtype=torch.bool)
            mask[:, :4] = True
            with source_codec_context(controller, dict(context, target=target)):
                output = controller.codec(local, target_cond=local, has_target_cond=mask, external_cond=external)
            return controller.loss_model.loss(target, output, None)['loss']
        with torch.no_grad():
            tokens = self.pose_tokens(batch, prepared['codec_target']['local_motions'])
        teacher = dict(context, args=prepared['pose_args'], kwargs=prepared['pose_kwargs'],
            target=prepared['codec_target'], tokens=tokens.detach())
        loss, _, _, _ = pose_teacher(controller, teacher, seed,
            isolate_cuda_rng=self.isolate_pose_teacher_rng)
        return loss

    def pose_tokens(self, batch, local):
        if not self.cache_pose_tokens:
            return self.controller.codec.encode_into_idx(local, fetch_overall_indices=False)
        if self.controller.codec.training or any(p.requires_grad for p in self.controller.codec.parameters()):
            raise ValueError('Memoized labels require an eval-mode frozen codec')
        ids = batch.get('sample_ids')
        if not torch.is_tensor(ids) or ids.dtype != torch.long or ids.ndim != 1 or len(ids) != len(local):
            raise ValueError('Memoized labels require immutable primitive IDs')
        keys = ids.detach().cpu().tolist()
        if len(keys) in self._pose_token_layouts and all(key in self._pose_tokens for key in keys):
            self.token_cache_hits += len(keys)
            shape, strides = self._pose_token_layouts[len(keys)]
            # randint_like preserves this head-major layout. A contiguous copy
            # would assign the same random stream to different logical tokens.
            tokens = torch.empty_strided(shape, strides, dtype=torch.long, device=local.device)
            return tokens.copy_(torch.stack([self._pose_tokens[key] for key in keys]).to(local.device))
        # Preserve the original full microbatch encoding on misses. No partial
        # batch encoding or altered source motion is introduced by the cache.
        tokens = self.controller.codec.encode_into_idx(local, fetch_overall_indices=False)
        self._pose_token_layouts[len(keys)] = (tuple(tokens.shape), tuple(tokens.stride()))
        self.token_cache_misses += len(keys)
        for key, value in zip(keys, tokens.detach().cpu()):
            self._pose_tokens[key] = value.clone()
        return tokens
