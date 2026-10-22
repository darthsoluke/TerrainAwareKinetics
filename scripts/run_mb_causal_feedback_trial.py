"""Real-time zero-training direction changes with a fixed state observer."""

import argparse
import json
import shutil
import subprocess
from threading import Event, Thread
from pathlib import Path
from time import perf_counter, perf_counter_ns

from motionbricks.mb_sonic.reference import file_sha256
from motionbricks.mb_sonic.runtime import create_lab_runtime
from scripts.isolate_mb_continuous_reference import save_json


def build_config(base, case, output, args):
    config = json.loads(base.read_text())
    seed = case/'scene/source_scene.npz'
    scene = json.loads((case/'scene/scene_manifest.json').read_text())['scene_id']
    config.update(scene_root=str(case.resolve()), initial_scene_id=scene, enabled_scene_ids=[scene],
        scene_manifest_sha256={scene:file_sha256(case/'scene/scene_manifest.json')},
        scene_seeds={scene:dict(path=str(seed.resolve()), sha256=file_sha256(seed))},
        scene_query_backend='native', scene_query_proofs={},
        recording_dir=str((output/'recording').resolve()), duration_s=28., max_run_wall_s=60.,
        mb_replan_interval_s=args.replan_interval, mb_adopt_interval_s=0., inference_rng_seed=20261020,
        mb_conditioning_mode=args.conditioning_mode,
        state_observer_limb_source='reference', retarget_limb_rotation_source=args.limb_rotation_source)
    for key in ('scene_catalog','state_feedback_segments','state_feedback_stop_segments','state_feedback_robot_prior'):
        config.pop(key, None)
    config['reference_constraints']['pose_task_frame'] = args.pose_task_frame
    if args.reference_stream:
        config.update(reference_stream_policy='continuous_native_AR_v3',mb_conditioning_mode='B1',
            mb_replan_interval_s=.02,consumer_horizon_s=args.consumer_horizon)
    if args.sequence == 'forward':
        config.update(duration_s=16.)
    if args.fixed_source_case is not None:
        config.update(mb_conditioning_mode='B1', mb_device='cpu', mb_execution='native', mb_planner_backend='thread')
    save_json(output/'config.json', config)
    return output/'config.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditioning-mode', choices=('B1','B2'), default='B2')
    parser.add_argument('--replan-interval', type=float, default=.1)
    parser.add_argument('--limb-rotation-source', choices=('markers','native_material_axes'), default='native_material_axes')
    parser.add_argument('--stable-gpu-profile', action='store_true')
    parser.add_argument('--fixed-source-case', type=Path)
    parser.add_argument('--sequence', choices=('directions','forward'), default='directions')
    parser.add_argument('--pose-task-frame', choices=('world','root'), default='world')
    parser.add_argument('--reference-stream', action='store_true')
    parser.add_argument('--consumer-horizon',type=float,default=.6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from motionbricks.mb_sonic.planner_process import source_identity
    identities = source_identity()
    source_dir = args.output/'source'
    source_dir.mkdir()
    for name in identities:
        shutil.copy2(Path(__file__).resolve().parents[1]/'motionbricks/mb_sonic'/name, source_dir/name)
    shutil.copy2(__file__, source_dir/Path(__file__).name)
    save_json(args.output/'source_manifest.json', identities)
    path = build_config(args.base, args.case, args.output, args)
    profile = None
    if args.stable_gpu_profile:
        from scripts.windows_gpu_profile import StableGpuProfile

        profile = StableGpuProfile()
        profile.__enter__()
    gpu_stop, gpu_rows = Event(), []
    def sample_gpu():
        while not gpu_stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,clocks.current.graphics,clocks.current.memory,pstate,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, creationflags=0x08000000)
            gpu_rows.append(dict(monotonic_ns=perf_counter_ns(), returncode=result.returncode,
                value=result.stdout.strip(), error=result.stderr.strip()))
            gpu_stop.wait(.5)
    telemetry = Thread(target=sample_gpu, daemon=True)
    telemetry.start()
    try:
        if args.fixed_source_case is not None:
            from unittest.mock import patch
            from scripts.fixed_mb_source_provider import load_provider

            with patch('motionbricks.mb_sonic.runtime.FrozenMBProvider.load',
                    side_effect=lambda settings, canonical_scene: load_provider(settings,
                        canonical_scene=canonical_scene, source_path=args.fixed_source_case/'generation/proposal.npz')):
                lab = create_lab_runtime(path, allocation_id=args.output.name)
        else:
            lab = create_lab_runtime(path, allocation_id=args.output.name)
    except BaseException:
        gpu_stop.set()
        telemetry.join()
        save_json(args.output/'gpu_telemetry.json', gpu_rows)
        if profile is not None:
            profile.close()
        raise
    config = lab.config()
    sequence = [(0., [0.,-1.2], 0.), (4., [1.2,0.], 0.), (8., [0.,1.2], 0.),
        (12., [-1.2,0.], 0.), (16., [0.,-1.2], .6), (20., [0.,0.], 0.)]
    if args.sequence == 'forward':
        sequence = sequence[:1]
    sent, started, last_print = [], perf_counter(), -1.
    lab.start()
    try:
        while True:
            now = lab.physics.time_s
            if len(sent) < len(sequence) and now >= sequence[len(sent)][0]-1e-8:
                time, velocity, yaw = sequence[len(sent)]
                command = dict(scene_id=config['scene']['manifest_id'],
                    scene_manifest_sha256=config['scene']['manifest_sha256'], episode_id=config['episode_id'],
                    session_id='controlled-causal-feedback-trial', command_id=len(sent)+1, command_seq=len(sent)+1,
                    client_input_ms=(perf_counter()-started)*1000, velocity_xy_mps=velocity,
                    yaw_rate_radps=yaw, input_frame='world_z_up', effective_intent=None,
                    server_received_ns=perf_counter_ns())
                lab.submit_intent(command)
                sent.append(dict(sim_time_s=now, command=command))
                if len(sent) == 1:
                    lab.begin_episode()
            if now-last_print >= 4:
                print(json.dumps(dict(time_s=now, status=lab._status, root=lab.physics.data.qpos[:3].tolist())), flush=True)
                last_print = now
            if lab._status.startswith(('completed','terminal','runtime_failure')):
                break
            if perf_counter()-started > 75:
                raise TimeoutError('Controlled trial exceeded75s wall cap')
            lab._stop_event.wait(.005)
    finally:
        lab.shutdown()
        if profile is not None:
            profile.close()
        gpu_stop.set()
        telemetry.join()
        save_json(args.output/'gpu_telemetry.json', gpu_rows)
    report = dict(status=lab._status, failure=lab._failure, errors=lab._errors,
        time_s=lab.physics.time_s, wall_s=perf_counter()-started, sent=sent,
        state_observer=lab.config()['reference']['state_observer'], optimizer_updates=0,
        stable_gpu_profile=args.stable_gpu_profile, profiling_scope_restored=profile is None or not profile.enabled,
        training=False, filtering=False, autonomous_retry=False)
    save_json(args.output/'result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
