"""Read-only factorial of previously learned Root/Pose weights on factual inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_native_root_validation import prediction_metrics
from scripts.audit_unified_stair_gait import difference, summarize
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def checked_component(base, learned, prefixes):
    if set(base) != set(learned):
        raise ValueError('Adaptation keys differ')
    for key in base:
        if not key.startswith(prefixes):
            torch.testing.assert_close(learned[key], base[key], atol=0, rtol=0)
    return {key: value for key, value in learned.items() if key.startswith(prefixes)}


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle).eval().requires_grad_(False)
    rep = inference.global_motion_rep
    paths = dict(initial=a.initial, learned_root=a.learned_root, learned_pose=a.learned_pose)
    saved = {key: torch.load(path, map_location='cpu', weights_only=False) for key, path in paths.items()}
    for checkpoint in saved.values():
        if checkpoint['contract']['bundle_sha256'] != sha256(a.bundle):
            raise ValueError('Bundle differs')
    if (saved['learned_root']['schema'] != 'motionbricks.unified_terrain.v1'
            or saved['learned_pose']['contract']['pose_condition_stage'] != 'input'
            or saved['learned_pose']['pose_delta']):
        raise ValueError('Require previously learned adapter-only components in the original input placement')
    initial = saved['initial']['adaptation']
    root = checked_component(initial, saved['learned_root']['adaptation'], ('root_stage1.', 'root_stage2.'))
    pose = checked_component(initial, saved['learned_pose']['adaptation'], ('pose.',))
    with np.load(a.recording, allow_pickle=False) as data:
        recorded = torch.from_numpy(data['global_features'].copy()).to(a.device)
    with np.load(a.scene, allow_pickle=False) as data:
        scene = {k: torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    contexts = []
    for frame in (48, 72, 96):
        history = reencode_played_history(recorded[None, frame:frame + 4], rep)
        contexts.append(dict(label=f'live-frame{frame}', history=history, source=None, scene=scene,
            command=history.new_tensor([[0., 1.6, 1., 0.]]), frame=frame))
    for case in cases:
        if case.get('entry') not in ('up', 'down'):
            continue
        first = case['interval']['first_riser'] - case['interval']['start']
        for start in sorted({max(0, first - 16), max(0, first - 4)}):
            if start + 33 > len(case['world']):
                raise ValueError('Incomplete real source entry')
            source = case['world'][None, start:start + 33].to(a.device)
            contexts.append(dict(label=f'{case["label"]}-start{start}', history=source[:, :4], source=source,
                scene=case['scene'], command=source_terminal_goal(source[..., :5]), frame=None))
    geometry = JointGeometryObjective(rep, a.device)
    rows = []
    for item in contexts:
        outputs = {}
        for arm, use_root, use_pose in [('before', False, False), ('root', True, False),
                                        ('pose', False, True), ('both', True, True)]:
            state = dict(initial)
            if use_root:
                state.update(root)
            if use_pose:
                state.update(pose)
            controller.adaptation.load_state_dict(state, strict=True)
            before_hash = state_digest(controller.adaptation, list(state))
            result = controller(item['history'], item['command'], item['scene'])
            future = result['world'][:, 4:28]
            outputs[arm] = dict(world=future.clone(), root=result['root_output']['pred_global_root_values'].clone())
            row = dict(label=item['label'], arm=arm, source_history=item['source'] is not None,
                future_skin=exact_skin_metrics(geometry, future, item['scene']),
                gait=summarize(rep, future, item['scene']), quality_approved=False)
            if item['source'] is not None:
                target = training_targets(inference, item['source'], item['history'], result['observation'])
                row['source_errors'] = prediction_metrics(controller, result, item['source'], target)
                row['leg_errors'] = difference(rep, future, item['source'][:, 4:28])
            elif arm == 'before':
                frame = item['frame']
                torch.testing.assert_close(future, recorded[None, frame + 4:frame + 28], atol=3e-5, rtol=1e-5)
            if state_digest(controller.adaptation, list(state)) != before_hash:
                raise ValueError('Read-only diagnosis changed weights')
            name = item['label'] + '-' + arm + '.npz'
            np.savez_compressed(a.output / name,
                global_features=torch.cat((item['history'], future), 1)[0].cpu().numpy(), fps=np.int32(30))
            row['recording'] = name
            rows.append(row)
        torch.testing.assert_close(outputs['before']['root'], outputs['pose']['root'], atol=0, rtol=0)
        torch.testing.assert_close(outputs['root']['root'], outputs['both']['root'], atol=0, rtol=0)
        write_json(a.output / 'progress.json', dict(rows=rows))
    write_json(a.output / 'complete.json', dict(rows=rows, checkpoints={k: sha256(v) for k, v in paths.items()},
        optimizer_updates=0, weights_unchanged=True, fixed_history_root_independence=True,
        future_truth_in_actor=False, runtime_root_replacement=False, runtime_changed=False,
        limitation='Previously learned module weights are combined offline, not output trajectories. '
            'One-primitive factual-history tests do not certify a closed-loop controller. '
            'Source future is used only for error measurement after normal actor generation.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'learned-root', 'learned-pose', 'cases', 'scene', 'recording', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
