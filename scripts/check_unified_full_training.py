"""Check batched training inputs, teacher identity, isolation and gradient scopes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config
from motionbricks.data.unified_full_training import NativeTrainingStage, prepare_native_batch
from motionbricks.data.unified_scene_field import VolumetricEgoSampler, field_from_heightfield
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_training_goals import source_terminal_goal
from motionbricks.data.terrain_decoder_pilot import training_targets
from scripts.audit_native_root_validation import source_root_intervention
from scripts.cache_full_terrain_primitives import observations,sources
from scripts.fit_native_root_fullbank import stack_calls
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_stages import fresh_kwargs
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def tensor_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree(item) for item in value)
    return value


def compare(left, right, path='input'):
    if torch.is_tensor(left):
        if left.dtype in (torch.bool, torch.int32, torch.int64):
            if not torch.equal(left, right):
                raise ValueError('Discrete native input differs at '+path+': '
                    +str(left.shape)+'/'+str(right.shape)+' '+str((left != right).nonzero()[:12].tolist()))
            return 0.
        torch.testing.assert_close(left, right, atol=5e-5, rtol=1e-5)
        return float((left-right).abs().max()) if left.numel() else 0.
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('Native input dictionary schema differs')
        return max([compare(left[key], right[key], path+'.'+key) for key in left]+[0.])
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError('Native input sequence lengths differ')
        return max([compare(a, b, path+'.'+str(index)) for index,(a,b) in enumerate(zip(left,right))]+[0.])
    if left != right:
        raise ValueError('Native input metadata differs: '+str(left)+'/'+str(right))
    return 0.


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    saved = torch.load(args.initial, map_location='cpu', weights_only=False)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.eval().requires_grad_(False)
    original = state_digest(controller, list(controller.state_dict()))
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    selected = [next(case for case in cases if case.get('entry') == entry) for entry in ('up', 'down')]
    selected.append(next(case for case in cases if case['name'] == 'idle'))
    if args.pfnn is not None:
        count=0
        for item in sources(SimpleNamespace(mixed=None,pfnn=args.pfnn,pilot=True),inference.global_motion_rep):
            for cursor in item['starts']:
                selected.append(dict(label=item['record']['source_id']+'-f'+str(cursor),name='pfnn',
                    world=item['world'][cursor:cursor+33],scene=item['scene']))
                count+=1
                if count>=3:
                    break
            if count>=3:
                break
        if count==0:
            raise ValueError('No corrected real PFNN primitive reached the input preflight')
    policy = terrain_observation_config(inference._root_model.args.get('observation_range_policy'))
    sampler = VolumetricEgoSampler()
    rows, singles, worlds, terrain, volumes = [], [], [], [], []
    with torch.no_grad():
        for case in selected:
            world = case['world'][None, :33].to(args.device)
            command = source_terminal_goal(world[..., :5])
            observation, volume = observations(world, case['scene'], policy, sampler, field_from_heightfield(case['scene']))
            calls = []
            handle = controller.root.register_forward_pre_hook(
                lambda _module, a, k: calls.append(tensor_tree((a, fresh_kwargs(k)))), with_kwargs=True)
            try:
                actual = controller(world[:, :4], command, case['scene'])
            finally:
                handle.remove()
            legacy_error = compare(observation, fresh_kwargs({'terrain_observation': actual['observation']})['terrain_observation'])
            volume_error = compare({key:volume[key] for key in ('features','distance','valid')},
                {key:controller.volume_observation[key] for key in ('features','distance','valid')})
            target = training_targets(inference, world, world[:, :4], observation)
            teacher_calls = []
            teacher_handle = controller.pose.register_forward_pre_hook(
                lambda _module, a, k: teacher_calls.append(tensor_tree((a, k))), with_kwargs=True)
            try:
                with source_root_intervention(controller, target['global_motions'][..., :5]):
                    controller(world[:, :4], command, case['scene'])
            finally:
                teacher_handle.remove()
            # Preparation captures the first all-masked call, before MaskGIT updates.
            teacher_args, teacher_kwargs = teacher_calls[0]
            if len(teacher_calls) != 4:
                raise ValueError('Expected the unchanged four-pass runtime Pose schedule')
            teacher_args = (teacher_args[0][:, :8], teacher_args[1][:, :32], teacher_args[2][:, :32], teacher_args[3][:, :32], *teacher_args[4:])
            teacher_kwargs = fresh_kwargs(teacher_kwargs)
            teacher_kwargs['root_height_trajectory'] = teacher_kwargs['root_height_trajectory'][:, :32]
            prepared = prepare_native_batch(controller, world, observation)
            root_error = compare(calls[0], (prepared['root_args'], prepared['root_kwargs']))
            teacher_error = compare((teacher_args, teacher_kwargs), (prepared['pose_args'], prepared['pose_kwargs']))
            changed = world.clone()
            changed[:, 4:, 1] += .37
            perturbed = prepare_native_batch(controller, changed, observation)
            isolation = compare((prepared['root_args'], prepared['root_kwargs']), (perturbed['root_args'], perturbed['root_kwargs']))
            if 'forward' in controller.root.__dict__:
                raise ValueError('Training teacher intervention was not restored')
            rows.append(dict(label=case['label'], legacy_observation_error=legacy_error, volume_error=volume_error,
                root_input_error=root_error, source_pose_teacher_error=teacher_error, future_height_root_input_error=isolation))
            singles.append(prepared)
            worlds.append(world)
            terrain.append(observation)
            volumes.append({key:volume[key] for key in ('features','distance','valid')})
        batch = dict(world=torch.cat(worlds), observation=stack_calls(terrain), volume=stack_calls(volumes))
        together = prepare_native_batch(controller, batch['world'], batch['observation'])
        batched_error = compare(together, stack_calls(singles))
    gradients = []
    for stage_name in ('root', 'codec', 'pose'):
        controller.load_state_dict(saved['model_state'], strict=True)
        controller.zero_grad(set_to_none=True)
        stage = NativeTrainingStage(controller, stage_name).train(True)
        loss = stage(batch, 2333)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite '+stage_name+' loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in controller.parameters() if parameter.requires_grad], .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid '+stage_name+' gradient')
        if any(parameter.grad is not None for parameter in controller.parameters() if not parameter.requires_grad):
            raise ValueError('Frozen parameter acquired gradient')
        gradients.append(dict(stage=stage_name, loss=float(loss.detach()), gradient_norm=float(norm),
            trainable_parameters=sum(parameter.numel() for parameter in controller.parameters() if parameter.requires_grad)))
        del loss, stage
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.eval().requires_grad_(False)
    if state_digest(controller, list(controller.state_dict())) != original:
        raise ValueError('Preflight did not restore initial model tensors')
    write_json(args.output/'complete.json', dict(passed=True, actual_optimizer_updates=0,
        rows=rows, batched_vs_single_preparation_error=batched_error, gradients=gradients,
        initial_model_restored=True, model_sha256=sha256(args.initial), bundle_sha256=sha256(args.bundle)))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','initial','cases','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pfnn',type=Path)
    options = parser.parse_args()
    try:
        main(options)
    except Exception as error:
        if options.output.is_dir():
            write_json(options.output/'failure.json', dict(error=repr(error)))
        raise
