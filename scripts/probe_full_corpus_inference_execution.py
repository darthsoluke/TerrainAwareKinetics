"""Check existing CUDA graph execution with the fully loaded mechanics fixture."""

import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from scripts.check_full_corpus_acceptance import check_idle, save


def main(args):
    snapshot = check_idle(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    save(args.output / 'state.json', dict(status='loading_fixture', gpu=args.gpu,
        gpu_snapshot=snapshot, actual_optimizer_updates=0))
    import numpy as np
    import torch
    from motionbricks.data.full_corpus_inference import tensor_digest
    from motionbricks.data.soma_lab import SomaLabSession
    from motionbricks.data.soma_terrain_execution import TerrainExecution
    from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
    from scripts.train_native_mb import sha256

    check_idle(args.gpu)
    runtime = FullCorpusAcceptanceRuntime(args.bundle, args.artifact, 'cuda:0',
        acceptance_test=True, allow_mechanics=True)
    if runtime.acceptance_receipt['mechanics_only'] is not True:
        raise ValueError('This diagnostic requires the explicit mechanics fixture')
    controller = runtime.unified_controller
    original_digest = tensor_digest(controller.state_dict())
    commands = [([0., 0.], 0.), ([0., 1.6], 0.), ([1.6, 0.], float(np.pi / 2)),
                ([0., -1.6], float(np.pi)), ([-1.6, 0.], float(-np.pi / 2))]

    def packet(command):
        velocity, heading = command
        session = SomaLabSession(runtime, 'terrain')
        expected_execution = getattr(runtime.inference, '_terrain_execution', None)
        if session.backend.execution is not expected_execution:
            raise ValueError('Session did not bind the selected execution engine')
        metadata, payload = session.plan(dict(type='plan', mode='terrain', request_id=1,
            velocity_xz=velocity, heading=heading))
        return metadata, payload, session.pending_playback['motion'].clone()

    def measure():
        durations = []
        for index in range(12):
            torch.cuda.synchronize()
            started = time.perf_counter()
            packet(commands[index % len(commands)])
            torch.cuda.synchronize()
            if index >= 2:
                durations.append((time.perf_counter() - started) * 1000)
        return dict(samples_ms=durations, median_ms=float(np.median(durations)),
                    p95_ms=float(np.percentile(durations, 95)))

    try:
        with torch.inference_mode():
            reference = [packet(command) for command in commands]
            eager_latency = measure()
            save(args.output / 'eager_latency.json', eager_latency)
            reference_path = args.output / 'eager_reference.pt'
            torch.save(dict(model_state_sha256=original_digest,
                acceptance_receipt_sha256=sha256(args.artifact / 'acceptance.json'),
                commands=commands, packets=reference), reference_path)
            previous_reference_sha256 = None
            if args.eager_reference is not None:
                previous = torch.load(args.eager_reference, map_location='cuda:0', weights_only=False)
                if (previous['model_state_sha256'] != original_digest or previous['commands'] != commands
                        or previous['acceptance_receipt_sha256'] != sha256(args.artifact / 'acceptance.json')):
                    raise ValueError('Reference model, scene or commands differ')
                for old, new in zip(previous['packets'], reference):
                    if old[1] != new[1] or not torch.equal(old[2], new[2]):
                        raise ValueError('Viewer CUDA prerequisites changed the original eager output')
                previous_reference_sha256 = sha256(args.eager_reference)
            if args.eager_only:
                if tensor_digest(controller.state_dict()) != original_digest:
                    raise ValueError('Reference inference changed model tensors')
                save(args.output / 'complete.json', dict(passed=True, reference_only=True,
                    actual_optimizer_updates=0, weights_unchanged=True,
                    model_state_sha256=original_digest, eager_packet_latency=eager_latency,
                    reference_sha256=sha256(reference_path), production_approved=False))
                save(args.output / 'state.json', dict(status='complete', actual_optimizer_updates=0))
                return
            save(args.output / 'state.json', dict(status='capturing_and_validating',
                eager_packet_latency=eager_latency, actual_optimizer_updates=0))
            started = time.perf_counter()
            if args.isolate_shape_constants:
                from motionbricks.data.full_corpus_static_trace import trace_fixed_call
                context = patch('motionbricks.data.soma_terrain_execution.trace_fixed_call', trace_fixed_call)
            else:
                from contextlib import nullcontext
                context = nullcontext()
            with context:
                runner = TerrainExecution(runtime.inference, runtime.terrain_scene, runtime.skinner,
                    runtime.terrain_initial_history, controller=controller,
                    independent_validation_seeds=True, fk_tolerance_m=3e-5)
            capture_seconds = time.perf_counter() - started
            object.__setattr__(runtime.inference, '_terrain_execution', runner)
            rows = []
            for command, (expected_meta, expected_payload, expected_motion) in zip(commands, reference):
                actual_meta, actual_payload, actual_motion = packet(command)
                for key in ('fps', 'frame_count', 'vertex_count', 'geometry_format', 'frame_stride_floats'):
                    if actual_meta[key] != expected_meta[key]:
                        raise ValueError('Compiled packet contract differs: ' + key)
                if actual_meta['frame_count'] != 8 or actual_meta['fps'] != 30:
                    raise ValueError('Fixed playback profile differs')
                actual_pose = runtime.inference.global_motion_rep.inverse(
                    actual_motion, False, return_all=False, joint_positions_from='global_rot_data')
                expected_pose = runtime.inference.global_motion_rep.inverse(
                    expected_motion, False, return_all=False, joint_positions_from='global_rot_data')
                fk_error = float((actual_pose['posed_joints'] - expected_pose['posed_joints']).norm(dim=-1).max())
                if fk_error > 3e-5:
                    raise ValueError('Compiled published motion exceeds existing native-bank FK gate')
                packed = np.frombuffer(actual_payload, dtype='<f4').reshape(8, 3 + 77 * 16)
                rig = torch.from_numpy(packed[:, 3:].reshape(8, 77, 4, 4).copy()).cuda()
                vertices = runtime.skinner.lbs(rig)
                _, native_vertices, _ = runtime.skin_features_with_joints(actual_motion, runtime.inference)
                native_vertices = torch.as_tensor(native_vertices, device=vertices.device)
                torch.testing.assert_close(vertices, native_vertices, atol=1e-5, rtol=1e-5)
                rows.append(dict(velocity_xz=command[0], heading=command[1], fk_max_m=fk_error,
                    payload_bitwise_equal=actual_payload == expected_payload,
                    official_skin_max_abs_m=float((vertices - native_vertices).abs().max())))
            compiled_latency = measure()
            if tensor_digest(controller.state_dict()) != original_digest:
                raise ValueError('Compilation or inference changed model tensors')
            report = dict(passed=True, mechanics_only=True, actual_optimizer_updates=0,
                weights_unchanged=True, model_state_sha256=original_digest,
                acceptance_receipt_sha256=sha256(args.artifact / 'acceptance.json'),
                script_sha256=sha256(Path(__file__)), compile_seconds=capture_seconds,
                isolate_shape_constants=args.isolate_shape_constants,
                previous_eager_reference_sha256=previous_reference_sha256,
                previous_eager_output_bitwise_equal=args.eager_reference is not None,
                native_execution_validation=runner.validation, published_packets=rows,
                eager_packet_latency=eager_latency, compiled_packet_latency=compiled_latency,
                fk_tolerance_m=3e-5, browser_validated=False, local_device_validated=False,
                realtime_30fps_validated=False, production_approved=False, delivery_approved=False,
                scope='Remote full-model CUDA graph and fresh-session packet parity/latency only.')
            save(args.output / 'complete.json', report)
            save(args.output / 'state.json', dict(status='complete', actual_optimizer_updates=0))
    except Exception as error:
        save(args.output / 'failure.json', dict(error=repr(error), actual_optimizer_updates=0,
            weights_unchanged=tensor_digest(controller.state_dict()) == original_digest,
            production_approved=False))
        raise
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--isolate-shape-constants', action='store_true')
    parser.add_argument('--eager-only', action='store_true')
    parser.add_argument('--eager-reference', type=Path)
    for name in ('bundle', 'artifact', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        if not (args.output / 'failure.json').exists():
            save(args.output / 'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        save(args.output / 'state.json', dict(status='failed', error=repr(error)))
        raise
