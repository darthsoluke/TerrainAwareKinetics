"""Compare A100, local native eager, real playback packets and midpoint prefetch."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.unified_terrain import PROFILE
from motionbricks.data.unified_terrain import UnifiedTerrainBackend
from scripts.soma_native_bank_server import NativeBankRuntime
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.inference_mode()
def check_entry(a, reference):
    runtime = NativeBankRuntime(a.bundle, a.artifact, 'cuda:0', reference['entry'], a.execution)
    serial = UnifiedTerrainBackend(runtime.unified_controller, runtime.terrain_initial_history, runtime.terrain_scene)
    serial.configure_runtime(dict(PROFILE)); session = SomaLabSession(runtime, 'terrain')
    parts, times, parity, prefetch = [], [], [], []
    for index, control in enumerate(reference['schedule']):
        original_history = session.backend.history.clone()
        first = serial.predict_plan(control['velocity'], control['heading'])
        if index == 0:
            direct = runtime.unified_controller(serial.history,
                serial.history.new_tensor([[*control['velocity'], np.cos(control['heading']), np.sin(control['heading'])]]),
                runtime.terrain_scene)['world'][:, 4:12]
            direct_error = float((direct - first['planned_motion']).abs().max())
            torch.testing.assert_close(direct, first['planned_motion'],
                atol=3e-5 if a.execution == 'verified-cuda' else 1e-5, rtol=1e-5)
        request = dict(mode='terrain', request_id=session.request_id + 1,
            velocity_xz=control['velocity'], heading=control['heading'], runtime_controls=dict(PROFILE))
        started = time.perf_counter()
        metadata, payload = session.plan(request)
        times.append((time.perf_counter() - started) * 1000)
        actual = session.pending_playback['motion']
        torch.testing.assert_close(actual, first['planned_motion'], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(session.backend.history, original_history, atol=0, rtol=0)
        packet = np.frombuffer(payload, dtype='<f4').reshape(8, metadata['frame_stride_floats'])
        np.testing.assert_array_equal(packet[:, :3], actual[0, :, :3].cpu().numpy())
        if metadata.get('geometry_format') != 'soma77-rig-v1' or metadata['fps'] != 30:
            raise ValueError('Live packet is not unchanged official SOMA at30Hz')
        expected = reference['motion'][:, index * 8:(index + 1) * 8].to(actual)
        rep = runtime.inference.global_motion_rep
        posed = rep.inverse(actual, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        reference_pose = rep.inverse(expected, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        error = float((posed - reference_pose).norm(dim=-1).max())
        root_error = float((actual[..., :3] - expected[..., :3]).abs().max())
        if error > 2e-4 or root_error > 1e-4:
            raise ValueError(f'Local versus A100 motion differs: FK{error}, Root{root_error}')
        parity.append(dict(block=index, serial_vs_packet_max=float((actual - first['planned_motion']).abs().max()),
            a100_fk_max_m=error, a100_root_max_m=root_error, neural=metadata['generation']['neural_generation']))
        parts.append(actual.cpu().clone())
        session.acknowledge_played(dict(request_id=request['request_id'], frame_count=8))
        serial.history = first['published_history'].clone()
    # Exercise actual prefetch semantics in a separate session, including deferral.
    session = SomaLabSession(runtime, 'terrain')
    serial = UnifiedTerrainBackend(runtime.unified_controller, runtime.terrain_initial_history, runtime.terrain_scene)
    serial.configure_runtime(dict(PROFILE))
    control = reference['schedule'][0]
    request = dict(mode='terrain', velocity_xz=control['velocity'], heading=control['heading'], runtime_controls=dict(PROFILE))
    metadata, _ = session.plan(dict(request, request_id=1)); active_id = 1
    first = serial.predict_plan(control['velocity'], control['heading'])
    for index in range(5):
        control = reference['schedule'][0] if index < 2 else reference['schedule'][-3 + index - 2]
        request = dict(request, velocity_xz=control['velocity'], heading=control['heading'])
        serial.history = first['published_history'].clone()
        first = serial.predict_plan(control['velocity'], control['heading'])
        next_id = session.request_id + 1
        prior_history = session.backend.history.clone()
        prior_buffer = session.backend._future_buffer.clone()
        played = 0 if getattr(session.backend, 'committed_history_prefetch', False) else 2
        observed = prior_history if played == 0 else session._observed_history(torch.cat((
            session.pending_playback['prior_history'], session.pending_playback['motion'][:, :played]), 1)[:, -4:])
        prepared, _ = session.prefetch(dict(request, type='prefetch', request_id=next_id,
            source_request_id=active_id, played_frame_count=played))
        torch.testing.assert_close(session.backend.history, observed, atol=0, rtol=0)
        torch.testing.assert_close(session.backend._future_buffer, prior_buffer, atol=0, rtol=0)
        if prepared['type'] == 'prefetch_deferred':
            session.acknowledge_played(dict(request_id=active_id, frame_count=8))
            next_id = session.request_id + 1
            session.plan(dict(request, request_id=next_id))
            actual = session.pending_playback['motion']
        else:
            actual = session.pending_playback['prepared']['motion']
            session.acknowledge_played(dict(request_id=active_id, frame_count=8))
        torch.testing.assert_close(actual, first['planned_motion'], atol=1e-5, rtol=1e-5)
        prefetch.append(dict(block=index + 1, deferred=prepared['type'] == 'prefetch_deferred',
            max_feature_error=float((actual - first['planned_motion']).abs().max()), command=control,
            played_frame_count=played, history_matches_declared_played_frames=True, buffer_unchanged_before_ack=True))
        active_id = next_id
    session.acknowledge_played(dict(request_id=active_id, frame_count=8))
    row = dict(entry=reference['entry'], execution=a.execution, frames=len(parts) * 8,
        direct_max_feature_error=direct_error,
        parity=parity, prefetch=prefetch, packet_ms=times, median_packet_ms=float(np.median(times)),
        p95_packet_ms=float(np.quantile(times, .95)), config=runtime.config()['live_model'],
        rig_vertex_error_m=runtime.terrain_quality['rig_transport_max_vertex_error_m'],
        compiled_parity=runtime.terrain_quality.get('execution_parity'), source_future_used=False)
    torch.save(torch.cat(parts, 1), a.output / (reference['entry'] + '-actual.pt'))
    runtime.unified_controller.close()
    return row


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.artifact / 'export.json').read_text())
    for name, digest in receipt['files'].items():
        if sha256(a.artifact / name) != digest:
            raise ValueError('Private artifact hash mismatch: ' + name)
    references = torch.load(a.artifact / 'offline-parity.pt', map_location='cpu', weights_only=False)
    if references['profile'] != PROFILE:
        raise ValueError('A100 and local playback profiles differ')
    rows = []
    for reference in references['cases']:
        if a.entry and reference['entry'] != a.entry:
            continue
        rows.append(check_entry(a, reference))
        write_json(a.output / 'progress.json', dict(rows=rows))
        gc.collect(); torch.cuda.empty_cache()
    write_json(a.output / 'complete.json', dict(rows=rows, passed=True, actual_browser_wasd_verified=False,
        checkpoint_sha256=receipt['checkpoint_sha256'], model_weights_changed=False, postprocessing=None))
    print(json.dumps(dict(passed=True, entries=[row['entry'] for row in rows])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'bundle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execution', choices=('eager', 'verified-cuda'), default='eager')
    parser.add_argument('--entry', choices=('up', 'down'))
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
