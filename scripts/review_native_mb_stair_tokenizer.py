"""Compare native tokenizer geometry with source tokens, not generated skill."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_native_mb_stairs import (
    SIDES, START, LAST, _skin, digest, read_shard, render_example, root_scores,
    select_windows, sole_scores, support_report, transform_support,
)


def reconstruct(model, raw, context_frames=(3,)):
    """Use the original index encoder/decoder and declared source information."""
    import torch
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    if raw.shape[1] != 65 or not context_frames or any(i not in range(4) for i in context_frames):
        raise ValueError("Need 65 source frames and observed pose indices in 0..3")
    rep, local_rep, network = model.motion_rep, model.local_motion_rep, model.pose_net
    with full_precision_evaluation(), torch.inference_mode():
        canonical = rep.change_first_heading(raw, 0., False, False)
        local = rep.dual_rep.global_to_local(rep.normalize(canonical), True, True,
            lengths=torch.tensor([65], device=raw.device))[:, :64]
        mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
        mask[:, list(context_frames)] = True
        external = extract_feature_from_motion_rep(local, local_rep,
            network.decoder_external_cond_feature_mode)
        codes = network.encode_into_idx(local)
        prediction = network.forward_decoder(codes, target_cond=local * mask[..., None],
            has_target_cond=mask, external_cond=external)['recon_state']
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite original tokenizer reconstruction")
        return local_rep.inverse(local, True, return_all=True), local_rep.inverse(
            prediction, True, return_all=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True, help='Unique name=run-directory entries')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a fresh output directory')
    runs = {}
    for entry in args.runs:
        name, separator, path = entry.partition('=')
        if not separator or not name or name == 'source' or name in runs:
            parser.error('Need unique non-source names and name=path entries')
        runs[name] = Path(path)

    from scripts.train_native_mb import select_gpu, verify_dataset
    gpu = select_gpu(args.gpu)
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    verify_dataset(args.dataset, manifest)
    import torch
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix

    torch.set_num_threads(4)
    models, metadata = {}, {}
    for name, path in runs.items():
        model, identity, binding = load_native_checkpoint(path, device='cuda')
        for filename in ('mean.npy', 'std.npy'):
            if digest(binding['dataset_dir'] / 'stats' / filename) != digest(args.dataset / 'stats' / filename):
                raise ValueError('Tokenizer normalization must match the review data')
        models[name], metadata[name] = model, identity
    skeleton = build_native_representation().skeleton
    surfaces, _, _, mesh_hash = official_sole_surfaces(skeleton)
    skinner = load_official_skinner(skeleton)
    report = {'task': 'source-encoded tokenizer stair geometry, not skill',
        'models': metadata, 'gpu_at_launch': gpu, 'dataset_manifest_sha256': digest(args.dataset / 'native_manifest.json'),
        'mesh_sha256': mesh_hash, 'source_sha256': digest(__file__),
        'conditions': {'pose_frames': [3], 'source_interior_tokens': True,
            'source_dense_external_root': True, 'future_endpoint_pose': False},
        'postprocessing': 'none', 'quality_approved': False,
        'scoring_frames_inclusive': [START, LAST],
        'interpretation': 'Same source masks for every model. Source-timed material speed measures reconstruction agreement, not measured load-bearing slip. Weak patches do not reveal full stairs. Source encoding is not inference capability.',
        'windows': []}
    args.output.mkdir(parents=True)
    for index, row in enumerate(select_windows(args.dataset)):
        data = read_shard(args.dataset / row['shard'])
        start = row['start']
        raw = torch.as_tensor(data['global_features'][start:start + 65], device='cuda')[None]
        motions = {}
        for name, model in models.items():
            source, motions[name] = reconstruct(model, raw)
            if 'source' in motions and not torch.allclose(source['posed_joints'], motions['source']['posed_joints'], atol=1e-6, rtol=0):
                raise ValueError('Model source reconstructions have inconsistent coordinates')
            motions['source'] = source
        rep = next(iter(models.values())).motion_rep
        _, angle = rep.compute_root_pos_and_rot(raw, return_angle=True, return_quat=False)
        rotation = angle_to_Y_rotation_matrix(-angle[:, 0])[0].cpu().numpy()
        origin = raw[0, 0, :3].cpu().numpy().copy()
        origin[1] = 0
        patches = transform_support(support_report(args.dataset, row)['patches'], rotation, origin)
        soles = {name: _skin(motion, skeleton, surfaces, skinner) for name, motion in motions.items()}
        parity = {}
        for side in SIDES:
            expected = (data[f'sole_vertices_{side}'][start:start + 64] - origin) @ rotation.T
            parity[side] = float(abs(expected - soles['source'][side]).max())
            if parity[side] > .003:
                raise ValueError(f'Source official sole geometry mismatch: {parity}')
        joints = {name: motion['posed_joints'][0].cpu().numpy() for name, motion in motions.items()}
        name = f"{index:02d}_{row['direction']}_start{start:06d}"
        result = {'source': row, 'artifact': name, 'source_geometry_max_error_m': parity,
            'sole_scores': {key: sole_scores(points, data, patches, start) for key, points in soles.items()},
            'root_scores': {key: root_scores(value, joints['source'], skeleton.root_idx) for key, value in joints.items()}}
        report['windows'].append(result)
        np.savez_compressed(args.output / (name + '.npz'), **joints, fps=np.float32(30))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'artifact': name, 'root_scores': result['root_scores']}), flush=True)
        if args.render:
            render_example(args.output / name, {'source': joints['source'], **{key: joints[key] for key in runs}},
                patches, skeleton.joint_parents.cpu().numpy(), row['direction'],
                condition_caption='source tokens/root; pose context frame 3; reconstruction only')
    report['status'] = 'completed_diagnostic_only'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
