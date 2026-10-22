"""Offline Pose-history intervention; Root and codec keep the actual full history."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import load_pose_checkpoint
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.audit_unified_stair_gait import summarize
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@contextmanager
def mask_pose_history(controller):
    def mask(_module, args):
        return (args[0], args[1], torch.zeros_like(args[2]), torch.zeros_like(args[3]), *args[4:])
    handle = controller.pose.register_forward_pre_hook(mask)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    checkpoint = torch.load(a.candidate, map_location='cpu', weights_only=False)
    load_pose_checkpoint(controller, checkpoint, sha256(a.bundle))
    rep = inference.global_motion_rep
    frozen = state_digest(controller, list(controller.state_dict()))
    with np.load(a.scene, allow_pickle=False) as data:
        scene = {k: torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    rows = []
    geometry = JointGeometryObjective(rep, a.device)
    for speed, frames in ((.8, (96, 120, 144, 168)), (1.6, (24, 48, 72, 96))):
        path = a.run / 'final/live_stairs' / f'forward-{speed:g}.npz'
        with np.load(path, allow_pickle=False) as data:
            recorded = torch.from_numpy(data['global_features'].copy()).to(a.device)
        for frame in frames:
            history = reencode_played_history(recorded[None, frame:frame + 4], rep)
            command = history.new_tensor([[0., speed, 1., 0.]])
            with capture_decoder_call(controller.codec) as actual_calls:
                actual = controller(history, command, scene)
            with mask_pose_history(controller), capture_decoder_call(controller.codec) as masked_calls:
                masked = controller(history, command, scene)
            if len(actual_calls) != 1 or len(masked_calls) != 1:
                raise ValueError('Expected one codec call per prediction')
            for key in ('target_cond', 'has_target_cond', 'external_cond'):
                torch.testing.assert_close(actual_calls[0]['kwargs'][key], masked_calls[0]['kwargs'][key], atol=0, rtol=0)
            torch.testing.assert_close(actual['root_output']['pred_global_root_values'],
                masked['root_output']['pred_global_root_values'], atol=0, rtol=0)
            torch.testing.assert_close(actual['world'][:, 4:28], recorded[None, frame + 4:frame + 28], atol=3e-5, rtol=1e-5)
            values = {}
            for name, result in (('actor', actual), ('offline_pose_history_mask', masked)):
                future = result['world'][:, 4:28]
                values[name] = dict(gait=summarize(rep, future, scene), skin=exact_skin_metrics(geometry, future, scene),
                    advance_m=float(future[0, -1, 2] - history[0, -1, 2]))
                np.savez_compressed(a.output / f'{speed:g}-frame{frame}-{name}.npz',
                    global_features=torch.cat((history, future), 1)[0].cpu().numpy(), fps=np.int32(30))
            rows.append(dict(speed=speed, frame=frame, history_gait=summarize(rep, history, scene), values=values,
                changed_tokens=int((actual['pose_logits'].argmax(-1)[:, :8] != masked['pose_logits'].argmax(-1)[:, :8]).sum()),
                root_difference=0., codec_conditions_difference=0.))
            write_json(a.output / 'progress.json', dict(rows=rows))
    if state_digest(controller, list(controller.state_dict())) != frozen:
        raise ValueError('Read-only probe changed weights')
    write_json(a.output / 'complete.json', dict(rows=rows, optimizer_updates=0, weights_unchanged=True,
        pose_training_config={k: inference._pose_model.args.get(k) for k in
            ('max_num_start_keyframes', 'no_start_keyframe_prob', 'keyframe_num_warmup_steps', 'online_conditioning')},
        runtime_changed=False, scope='OFFLINE only. Root and codec retain identical actual full-history conditions. '
        'Masking Pose history is not a deployed fix or an expert recovery label.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'candidate', 'run', 'scene', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
