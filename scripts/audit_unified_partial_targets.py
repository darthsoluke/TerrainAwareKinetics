"""Offline ablation of complete-latent replacement for heading-only Root targets."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.unified_terrain import UnifiedTerrainController, SCHEMA
from scripts.audit_unified_stair_anticipation import fresh_trace
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@contextmanager
def complete_root_targets_only(root_decoder):
    """Keep all Root/velocity/heading inputs elsewhere; ablate only this skip path."""
    audit = []

    def hook(_module, args, kwargs):
        mask = kwargs.get('target_component_mask')
        if mask is None or mask.ndim != 3 or mask.shape[-1] != 5:
            raise ValueError('Expected native five-component Root target mask')
        partial = mask.any(-1) & ~mask.all(-1)
        partial[:, :4] = False
        changed = mask & ~partial[..., None]
        updated = dict(kwargs, target_component_mask=changed, has_target_cond=changed.any(-1))
        audit.append(dict(partial_future_frames=int(partial.sum()),
                          history_mask_unchanged=bool(torch.equal(changed[:, :4], mask[:, :4])),
                          external_condition_same_object=updated['external_cond'] is kwargs['external_cond']))
        return args, updated

    handle = root_decoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield audit
    finally:
        handle.remove()


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    with np.load(args.scene, allow_pickle=False) as values:
        scene = {key: torch.from_numpy(values[key].copy()) for key in
                 ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    with np.load(args.played, allow_pickle=False) as values:
        played = torch.from_numpy(values['global_features'].copy()).to(args.device)[None]
    rows = []
    for checkpoint in args.candidates:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('schema') != SCHEMA:
            raise ValueError('This diagnostic accepts adaptation-only candidates')
        controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
        for frame in (48, 64, 72, 80, 88):
            history = reencode_played_history(played[:, frame - 4:frame], inference.global_motion_rep)
            command = history.new_tensor([[0., 1.6, 1., 0.]])
            before, before_root, before_metrics = fresh_trace(controller, history, command, scene)
            with complete_root_targets_only(controller.root._conv_output) as hooks:
                after, after_root, after_metrics = fresh_trace(controller, history, command, scene)
            replay, replay_root, _ = fresh_trace(controller, history, command, scene)
            torch.testing.assert_close(replay['world'], before['world'], atol=0, rtol=0)
            torch.testing.assert_close(replay_root, before_root, atol=0, rtol=0)
            rows.append(dict(checkpoint=checkpoint.parent.name, checkpoint_sha256=sha256(checkpoint), frame=frame,
                before=before_metrics, complete_targets_only=after_metrics, hooks=hooks,
                root_y_max_abs_difference_m=float((after_root[:, 4:28, 1] - before_root[:, 4:28, 1]).abs().max()),
                decoded_y_max_abs_difference_m=float((after['world'][:, 4:28, 1] - before['world'][:, 4:28, 1]).abs().max()),
                restored_forward_error=0.))
            write_json(args.output / 'progress.json', dict(rows=rows))
    write_json(args.output / 'complete.json', dict(rows=rows, optimizer_updates=0, runtime_changed=False,
        inputs='same factual history, explicit command and scene; no source future',
        scope='Condition-path ablation only, not an approved correction or a full locomotion review',
        quality_approved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'scene', 'played', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise
