"""Package the visually accepted native model without changing any model tensor."""
import argparse
import configparser
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch

from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.package_approved_parc_live import state_hash
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


PARENT = '03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1'


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if sha256(a.checkpoint) != PARENT:
        raise ValueError('Visually accepted native model identity differs')
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if (saved['contract']['bundle_sha256'] != sha256(a.bundle)
            or saved['contract']['cases_sha256'] != sha256(a.cases)):
        raise ValueError('Source identity differs')
    compact = {key: saved[key] for key in ('schema', 'contract', 'model_state', 'actual_updates')}
    torch.save(compact, a.output / 'weights.pt')
    digest = state_hash(saved['model_state'])
    if state_hash(torch.load(a.output / 'weights.pt', map_location='cpu', mmap=True,
                             weights_only=False)['model_state']) != digest:
        raise ValueError('Packaging changed model tensors')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    by_label = {case['label']: case for case in cases}
    entries, references = [], []
    for window in saved['contract']['capacity_anchors']:
        case = by_label[window['label']]
        if case.get('entry') not in ('up', 'down'):
            continue
        name = case['entry']; history = case['world'][None, :4].to(a.device)
        command = source_terminal_goal(case['world'][None, window['start']:window['start'] + 33, :5].to(a.device))[0]
        heading = float(torch.atan2(command[3], command[2])); travel = float(torch.atan2(command[0], command[1]))
        metadata = dict(schema='parc.live_scene.v1', history_frames=4, future_motion_included=False,
            source_id=case['row']['source_id'], source_start_frame=case['start_frame'],
            initial_state='same factual four frames as orange source-start recording')
        scene_path = a.output / (name + '-scene.npz')
        np.savez_compressed(scene_path,
            **{k: torch.as_tensor(case['scene'][k]).numpy() for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')},
            initial_history=history.cpu().numpy(), metadata_json=np.array(json.dumps(metadata)))
        entry_command = dict(velocity_xz_mps=command[:2].tolist(), speed_mps=float(command[:2].norm()),
            heading_rad=heading, travel_heading_rad=travel,
            heading_offset_rad=float(np.arctan2(np.sin(heading - travel), np.cos(heading - travel))))
        write_json(scene_path.with_suffix('.json'), dict(metadata, scene_sha256=sha256(scene_path), entry_command=entry_command))
        entries.append(dict(entry=name, command=entry_command, scene=scene_path.name))
        backend = UnifiedTerrainBackend(controller, history, case['scene']); backend.configure_runtime(dict(PROFILE))
        expected = []
        schedule = [dict(velocity=command[:2].tolist(), heading=heading) for _ in range(9)]
        schedule += [dict(velocity=[0., 0.], heading=heading),
                     dict(velocity=[0., 1.6], heading=0.), dict(velocity=[-1.6, 0.], heading=-np.pi / 2)]
        for control in schedule:
            result = backend.predict_plan(control['velocity'], control['heading'])
            expected.append(result['planned_motion'].cpu().clone())
            backend.history = result['published_history'].clone()
        references.append(dict(entry=name, schedule=schedule, motion=torch.cat(expected, 1),
            final_history=backend.history.cpu().clone()))
    # This separate test artifact is never read by the live server.
    torch.save(dict(profile=dict(PROFILE), cases=references), a.output / 'offline-parity.pt')
    controller.close()
    write_json(a.output / 'export.json', dict(schema='motionbricks.native_bank_live.v1',
        original_checkpoint_sha256=PARENT, checkpoint_sha256=sha256(a.output / 'weights.pt'),
        source_bundle_sha256=sha256(a.bundle), model_state_sha256=digest, model_tensors_unchanged=True,
        runtime_profile=dict(PROFILE), entries=entries, user_visual_approval=True,
        live_verified=False, quality_approved=False,
        limitations='Constant-command terrain failures and flat regressions remain; user-authorized interactive trial.',
        files={p.name: sha256(p) for p in a.output.iterdir() if p.is_file()}))
    files = {p.name: sha256(p) for p in a.output.iterdir() if p.is_file()}
    write_json(a.output / 'receipt.json', dict(files=files))
    archive = a.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(a.output, arcname=a.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive), files=len(files)))
    settings = configparser.ConfigParser(interpolation=None); settings.read(a.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Owned private OAuth client required')
    env = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(a.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=env, check=True)
    print(json.dumps(dict(uploaded=True, archive_sha256=sha256(archive), model_tensors_unchanged=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'cases', 'output', 'drive-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
