"""Offline same-weight relocation of Root SDF input; not a runtime fix."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.unified_terrain import UnifiedTerrainController, SCHEMA
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_unified_entry_learning import inspect
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


class Bypass(torch.nn.Module):
    def forward(self, value, _memory, _valid):
        return value


@contextmanager
def before_decoder(controller):
    adapter = controller.adaptation['root_stage2']
    controller.adaptation['root_stage2'] = Bypass()

    def condition(_module, args):
        if not controller._adapting:
            return args
        value = adapter(args[0].transpose(1, 2), controller.memory, controller.valid)
        return (value.transpose(1, 2), *args[1:])

    handle = controller.root._conv_output.register_forward_pre_hook(condition)
    try:
        yield
    finally:
        handle.remove()
        controller.adaptation['root_stage2'] = adapter


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle).eval().requires_grad_(False)
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    with np.load(args.live_recording, allow_pickle=False) as archive:
        live = torch.from_numpy(archive['global_features'].copy()).to(args.device)
    with np.load(args.scene, allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(archive[key].copy()) for key in
                 ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    contexts = []
    for frame in (64, 72, 80, 88):
        history = reencode_played_history(live[None, frame:frame + 4], inference.global_motion_rep)
        contexts.append((f'live-{frame}', history,
                         live.new_tensor([[0., 1.6, 1., 0.]]), scene, None))
    for case in cases:
        if case['native_split'] == 'train':
            for start in (0, 8, 16):
                source = case['world'][start:start + 33].to(args.device)
                contexts.append((f'{case["label"]}-{start}', source[None, :4],
                                 source_terminal_goal(source[None, :, :5]), case['scene'], source))
    rows = []
    for candidate in args.candidates:
        saved = torch.load(candidate, map_location='cpu', weights_only=False)
        if saved['schema'] != SCHEMA or saved['contract']['bundle_sha256'] != sha256(args.bundle):
            raise ValueError('Only matched adaptation-only checkpoints are supported')
        controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
        before = state_digest(controller, list(controller.state_dict()))
        for label, history, command, field, reference in contexts:
            original = controller(history, command, field)['world'].clone()
            baseline = inspect(controller, history, command, field, reference)
            with before_decoder(controller):
                moved = inspect(controller, history, command, field, reference)
            replay = controller(history, command, field)['world']
            torch.testing.assert_close(replay, original, atol=0, rtol=0)
            rows.append(dict(checkpoint=candidate.parent.name, label=label,
                             before_transformer=baseline, before_decoder=moved, replay_error=0.))
        if state_digest(controller, list(controller.state_dict())) != before:
            raise ValueError('Placement audit changed weights')
        write_json(args.output / 'progress.json', dict(rows=rows))
    write_json(args.output / 'complete.json', dict(rows=rows, optimizer_updates=0,
        weights_unchanged=True, runtime_changed=False, quality_approved=False,
        live_history='official FK reencoded, matching publication',
        note='Moving a trained adapter changes its input distribution. Sensitivity alone does '
             'not prove a bottleneck or justify deployment; a learned, matched control is required.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'cases', 'scene', 'live-recording', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
