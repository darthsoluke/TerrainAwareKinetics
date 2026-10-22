"""Frozen final-model train/validation diagnostic on distinct cached windows."""

import argparse
import os
from pathlib import Path
import subprocess

from scripts.run_pfnn_bias_workflow import digest, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def select_windows(dataset, domain, count):
    import numpy as np

    pool = [i for i, actual in enumerate(dataset.ids)
            if dataset.records[actual]['domain'] == domain]

    def key(index):
        row = dataset.records[dataset.ids[index]]
        return row['source_sha256'], row['start']

    # Exclude whole source windows from the already completed periodic validation.
    excluded = {key(i) for i in np.random.default_rng(123).permutation(pool)[:128]}
    selected, seen = [], set()
    for i in np.random.default_rng(281016).permutation(pool).tolist():
        row = dataset.records[dataset.ids[i]]
        identity = key(i)
        if row['quarter'] != 0 or identity in seen:
            continue
        if row['split'] == 'validation' and identity in excluded:
            continue
        selected.append(i)
        seen.add(identity)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Insufficient distinct windows for the declared sample')
    return selected


def main(args):
    snapshot, available = gpu_snapshot()
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv'], text=True)
    if args.gpu not in available:
        raise ValueError('Selected GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
    from motionbricks.data.native_mb_vertical_frame import shift_global_motion
    from motionbricks.data.terrain_decoder_pilot import vertical_origin
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.audit_unified_stair_gait import difference
    from scripts.portable_parc_online_soma import load_terrain_bundle

    torch.set_num_threads(4)
    torch.manual_seed(281016)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.9)
    state = dict(pid=os.getpid(), gpu=args.gpu, domain=args.domain,
        gpu_snapshot_before=snapshot, compute_processes_before=processes,
        actual_optimizer_updates=0, quality_approved=False, delivery_approved=False)
    save(args.output/'state.json', dict(state, status='loading_final_model'))
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    checkpoint_hash, bundle_hash = digest(args.checkpoint), digest(args.bundle)
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    load_candidate(controller, saved,
        inference_receipt(saved, checkpoint_hash, checkpoint_hash, bundle_hash), bundle_hash)
    before = tensor_digest(controller.state_dict())

    def to_device(value):
        if torch.is_tensor(value):
            return value.to('cuda:0')
        if isinstance(value, dict):
            return {key: to_device(item) for key, item in value.items()}
        return value

    stages = {name: NativeTrainingStage(controller, name).eval()
              for name in ('root', 'codec', 'pose')}
    controller.eval().requires_grad_(False)
    rows = []
    try:
        with torch.no_grad():
            for split in ('train', 'validation'):
                dataset = PrimitiveDataset([path/'data' for path in args.cache], split)
                selected = select_windows(dataset, args.domain, args.count)
                batch = to_device(collate_primitives([dataset[index] for index in selected]))
                captured = {}
                root_handle = controller.root.register_forward_hook(
                    lambda _module, _inputs, output: captured.update(root=output))
                try:
                    losses = dict(root=float(stages['root'](batch, seed=281016)))
                finally:
                    root_handle.remove()
                codec_handle = controller.codec.register_forward_hook(
                    lambda _module, _inputs, output: captured.update(codec=output))
                try:
                    losses['codec'] = float(stages['codec'](batch, seed=281016))
                finally:
                    codec_handle.remove()
                losses['pose'] = float(stages['pose'](batch, seed=281016))
                source, observation = batch['world'], batch['observation']
                origin = vertical_origin(inference, source[:, :4], observation)
                root = inference.global_motion_rep.unnormalize(
                    captured['root']['pred_global_root_values'][:, :32])
                root_y = root[:, 4:28, 1] + origin[:, None]
                local = captured['codec']['recon_state']
                world = inference.motion_rep.dual_rep.local_to_global(local, True, False,
                    lengths=torch.full((len(local),), 32, device=local.device, dtype=torch.long))
                offsets, heading, _ = inference._extract_initial_root_info(source[:, :4, :5])
                world = inference._reapply_initial_root_info(dict(pred_global_poses=world,
                    reference_start_root_global_offsets=offsets, reference_start_root_global_heading=heading))
                world = shift_global_motion(world, inference.global_motion_rep, origin)
                windows = []
                for offset, index in enumerate(selected):
                    record = dataset.records[dataset.ids[index]]
                    windows.append(dict(sample_id=dataset.ids[index], record=record,
                        source_codec_errors=difference(inference.global_motion_rep,
                            world[offset:offset+1, 4:28], source[offset:offset+1, 4:28]),
                        source_codec_root_xyz_rms_m=float((world[offset, 4:28, :3] -
                            source[offset, 4:28, :3]).square().mean().sqrt()),
                        root_y_mae_m=float((root_y[offset]-source[offset, 4:28, 1]).abs().mean()),
                        root_y_max_m=float((root_y[offset]-source[offset, 4:28, 1]).abs().max())))
                rows.append(dict(split=split, domain=args.domain, examples=len(selected),
                    native_losses=losses, windows=windows))
                save(args.output/'progress.json', dict(state, rows=rows))
        if tensor_digest(controller.state_dict()) != before:
            raise ValueError('Frozen diagnostic modified model tensors')
        save(args.output/'complete.json', dict(state, completed=True, rows=rows,
            weights_unchanged=True, checkpoint_sha256=checkpoint_hash,
            selection_seed=281016, distinct_source_windows=True, quarter=0,
            periodic_validation_source_windows_excluded=True,
            source_codec_is_offline_diagnostic_not_actor=True,
            runtime_changed=False, geometry_evaluated=False,
            scope='64 distinct windows per train/validation split and domain. '
                  'Cached observations and original native targets; no new data preparation. '
                  'Descriptive sample only, not a full-corpus or closed-loop quality gate.'))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, nargs='+', required=True)
    parser.add_argument('--domain', choices=('pfnn', 'parc', 'bones'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--count', type=int, default=64, choices=(64,))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        raise
