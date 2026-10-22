"""Explicit bounded Root/Pose fitting scopes and causal masked-token teachers."""

from contextlib import contextmanager

import numpy as np
import torch

from .terrain_decoder_pilot import training_targets
from .terrain_root_supervision import factual_history_root_targets
from .unified_training_goals import source_terminal_goal
from motionbricks.helper.data_training_util import sample_keyframes


POSE_SCHEMA = 'motionbricks.unified_terrain.pose_capacity.v1'


def configure_pose_scope(controller, arm):
    if arm not in ('adapter', 'capacity', 'root_pose'):
        raise ValueError('Choose adapter, capacity or root_pose')
    controller.eval().requires_grad_(False)
    controller.adaptation['pose'].requires_grad_(True)
    if arm == 'root_pose':
        for name in ('root_stage1', 'root_stage2'):
            controller.adaptation[name].requires_grad_(True)
    if arm == 'capacity':
        if len(controller.pose._transformer_model.layers) != 16:
            raise ValueError('Expected the verified16-layer native Pose')
        for layer in controller.pose._transformer_model.layers[-2:]:
            layer.requires_grad_(True)
        for name in ('_proj_pose_output_logit', '_proj_local_root_values', '_terrain_root_height_projection'):
            getattr(controller.pose, name).requires_grad_(True)
        controller.pose._terrain_root_height_gate.requires_grad_(True)
    return [name for name, value in controller.named_parameters() if value.requires_grad]


@torch.no_grad()
def pose_example(controller, source, scene, *, actor_gradient=False):
    if source.shape[0] != 1 or source.shape[1] != 33:
        raise ValueError('One complete real33-frame source window is required')
    command = source_terminal_goal(source[..., :5])
    with torch.set_grad_enabled(actor_gradient):
        result = controller(source[:, :4], command, scene, training=actor_gradient, straight_through=actor_gradient)
    target = training_targets(controller.inference, source, source[:, :4], result['observation'])
    tokens = controller.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False).detach()
    args, kwargs = controller.last_pose_request
    return dict(result=result, target=target, tokens=tokens, args=args, kwargs=kwargs,
                memory=controller.memory.detach(), valid=controller.valid.detach(), command=command)


def decoded_native_losses(controller, example, source):
    """The existing native motion/Root objectives, evaluated on the causal actor."""
    local = example['result']['local_prediction']
    motion = controller.loss_model.loss(example['target'],
        dict(recon_state=local, l_commit=local.sum() * 0, perplexity=0.), None)['loss']
    root_target = factual_history_root_targets(controller.inference, example['target'],
        source[:, :4], example['result']['observation'])
    root = controller.inference._root_model.loss(root_target, example['result']['root_output'])['loss']
    return dict(native_motion=motion, native_root=root)


@contextmanager
def pose_memory(controller, example):
    adapting, memory, valid = controller._adapting, controller.memory, controller.valid
    controller.memory, controller.valid = example['memory'], example['valid']
    controller._adapting = True
    try:
        yield
    finally:
        controller._adapting, controller.memory, controller.valid = adapting, memory, valid


def teacher_history_args(args, seed, policy='full'):
    if policy == 'full':
        return args
    if policy != 'native-sparse-start-v1':
        raise ValueError('Unknown Pose teacher history policy')
    pose, known = args[2:4]
    if known.dtype != torch.bool or not known[:, :4].all() or known[:, 4:].any():
        raise ValueError('Sparse teacher requires exactly four factual starting poses and no future pose')
    devices = [pose.device.index] if pose.is_cuda else []
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            np.random.seed(seed ^ 0x53504152)
            torch.manual_seed(seed ^ 0x53504152)
            # Mature MB start-keyframe distribution; only the factual prefix is eligible.
            chosen, _ = sample_keyframes(pose[:, :4], 4, [0., .25, .25, .25, .25])
    finally:
        np.random.set_state(numpy_state)
    mask = known.clone()
    mask[:, :4] = chosen
    return (args[0], args[1], pose * mask[..., None], mask, *args[4:])


def pose_teacher(controller, example, seed, *, all_mask=False, history_policy='full',
                 isolate_cuda_rng=False):
    teacher = controller.inference._pose_model
    tokens = example['tokens']
    batch = dict(groundtruth_pose_tokens=tokens, local_motions=example['target']['local_motions'])
    devices = [tokens.device.index] if tokens.is_cuda else []
    if isolate_cuda_rng:
        if history_policy != 'full':
            raise ValueError('Isolated Pose sampling requires the full factual history policy')
        from .pose_teacher_rng import isolated_pose_seed
        scope = isolated_pose_seed(tokens, seed)
    else:
        scope = torch.random.fork_rng(devices=devices)
    with scope:
        if not isolate_cuda_rng:
            torch.manual_seed(seed)
        keys = ('focused_token_mask', 'masked_token_mask', 'incorrect_token_mask',
                'correct_token_mask', 'mask_percentage', 'incorrect_percentage')
        batch.update(zip(keys, teacher._get_token_masks(batch)))
        if all_mask:
            for key in ('focused_token_mask', 'masked_token_mask'):
                batch[key] = torch.ones_like(batch[key])
            for key in ('incorrect_token_mask', 'correct_token_mask'):
                batch[key] = torch.zeros_like(batch[key])
        corrupted = teacher._generate_tokens_for_training(batch)
    args, kwargs = teacher_history_args(example['args'], seed, history_policy), example['kwargs']
    batch['history_condition_mask'] = args[3].detach().clone()
    padded = torch.full_like(args[0], controller.pose.POSE_MASK_ID)
    padded[:, :tokens.shape[1]] = corrupted
    with pose_memory(controller, example):
        logits = controller.pose(padded, *args[1:], **kwargs)['pose_logits'][:, :tokens.shape[1]]
    loss = teacher.loss(batch, dict(pose_logits=logits))['loss'] if batch['focused_token_mask'].any() else logits.sum() * 0
    return loss, logits, batch, corrupted


def load_pose_checkpoint(controller, saved, bundle_hash):
    if saved.get('schema') != POSE_SCHEMA or saved['contract']['bundle_sha256'] != bundle_hash:
        raise ValueError('Pose checkpoint schema/baseline mismatch')
    if saved['contract'].get('pose_condition_stage', 'input') != getattr(controller, 'pose_condition_stage', 'input'):
        raise ValueError('Pose conditioning placement differs from the checkpoint')
    arm = saved['contract']['arm']
    names = configure_pose_scope(controller, arm)
    if names != saved['contract']['trainable_parameters']:
        raise ValueError('Pose checkpoint parameter scope changed')
    expected = {name.removeprefix('pose.') for name in names if name.startswith('pose.')}
    if set(saved['pose_delta']) != expected:
        raise ValueError('Checkpoint must carry every changed native Pose tensor')
    state = controller.pose.state_dict()
    for key, value in saved['pose_delta'].items():
        if value.shape != state[key].shape:
            raise ValueError('Native Pose tensor shape mismatch')
        state[key] = value
    controller.pose.load_state_dict(state, strict=True)
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    controller.eval().requires_grad_(False)
