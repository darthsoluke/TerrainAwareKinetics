"""Bounded 2x2 native MB decoder experiment, with fixed held-out rollouts."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from motionbricks.data.terrain_decoder_pilot import DecoderExperiment, training_targets
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_feedback_ablation import evaluation_goal, terrain_gap, mean_or_none
from scripts.train_native_mb import select_gpu, sha256
from scripts.train_native_mb_control import state_digest


ARMS = {'baseline': (False, False), 'feedback': (False, True),
        'terrain': (True, False), 'both': (True, True)}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def frozen_state(pilot):
    return {name: state_digest(module) for name, module in {
        'root': pilot.inference._root_model, 'pose_backbone': pilot.inference._pose_model.backbone_net,
        'encoder': pilot.codec.encoder, 'quantizer': pilot.codec.quantizer}.items()}


def geometry(pilot, world, skin, indices):
    pose = pilot.inference.global_motion_rep.inverse(world, False, return_all=True,
                                                   joint_positions_from='global_rot_data')
    y, xz = region_geometry_from_motion(pose['local_joint_rots'][0], world[0, :, :3], skin, indices)
    return pose, y, xz


@torch.no_grad()
def evaluate(pilot, bundle, data, output, *, oracle=False):
    output.mkdir(parents=True, exist_ok=False)
    rep = pilot.inference.global_motion_rep
    skin = move_skinner(load_official_skinner(rep.skeleton), torch.device(pilot.inference.device))
    skin.skeleton_skin.float().to(pilot.inference.device)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    left = sole_region_indices(surfaces, 'Left')
    right = sole_region_indices(surfaces, 'Right')
    indices = np.concatenate((left, right))
    rows, oracle_rows, rest_rows = [], [], []
    validation = [w for w in data['windows'] if w['split'] == 'validation']
    for wi, item in enumerate(validation):
        world = item['world'].to(pilot.inference.device)
        scene = data['scenes'][item['scene']]
        source = world[:, 4:36]
        sp, sy, sxz = geometry(pilot, source, skin, indices)
        sgap, valid = terrain_gap(sy, sxz, scene)
        slow = ((sgap[:-1].abs() <= .02) & (sgap[1:].abs() <= .02) & valid[:-1] & valid[1:]
                & ((sxz[1:]-sxz[:-1]).norm(dim=-1)*30 <= .1))
        for mode in ('teacher_history', 'played_history'):
            history = world[:, :4]
            predictions = []
            for block in range(4):
                offset = block*8
                if mode == 'teacher_history':
                    history = world[:, offset:offset+4]
                goal = evaluation_goal(pilot.inference, world, offset, bundle['protocol']['version'])
                prediction, call = pilot.request(history, goal, scene)
                predictions.append(prediction[:, 4:12])
                if oracle and mode == 'teacher_history' and block == 0:
                    target = training_targets(pilot.inference, world[:, :33], history, call['observation'])
                    encoded = pilot.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False)
                    for token_source in ('model', 'source'):
                        for root_source in ('model', 'source'):
                            replay = dict(call, kwargs=dict(call['kwargs']))
                            if token_source == 'source':
                                tokens = call['args'][0].clone()
                                tokens[:, :8] = encoded
                                replay['args'] = (tokens,)
                            if root_source == 'source':
                                external = replay['kwargs']['external_cond'].clone()
                                external[:, :32] = extract_feature_from_motion_rep(target['local_motions'],
                                    pilot.inference.local_motion_rep, pilot.codec.decoder_external_cond_feature_mode)
                                replay['kwargs']['external_cond'] = external
                            decoded = pilot.restore_world(pilot.replay(replay), history, call['observation'])
                            if token_source == root_source == 'model':
                                torch.testing.assert_close(decoded[:, :32], prediction[:, :32], atol=2e-5, rtol=0)
                            dp, _, _ = geometry(pilot, decoded[:, 4:12], skin, indices)
                            oracle_rows.append({'window': wi, 'tokens': token_source, 'root': root_source,
                                'root_y_mae_m': float((decoded[:, 4:12, 1]-world[:, 4:12, 1]).abs().mean()),
                                'fk_mpjpe_m': float((dp['posed_joints']-sp['posed_joints'][:, :8]).norm(dim=-1).mean()),
                                'offline_oracle': token_source == 'source' or root_source == 'source'})
                history = reencode_played_history(prediction[:, 8:12], rep)
            generated = torch.cat(predictions, 1)
            pp, py, pxz = geometry(pilot, generated, skin, indices)
            pgap, known = terrain_gap(py, pxz, scene)
            selected = slow & known[:-1] & known[1:]
            row = {'window': wi, 'origin': item['origin'], 'category': item['category'], 'history': mode,
                'source_id': item['source_id'], 'start': item['start'],
                'root_y_mae_m': float((generated[..., 1]-source[..., 1]).abs().mean()),
                'fk_mpjpe_m': float((pp['posed_joints']-sp['posed_joints']).norm(dim=-1).mean()),
                'penetration_mean_m': mean_or_none(torch.relu(-pgap), known),
                'source_penetration_mean_m': mean_or_none(torch.relu(-sgap), valid),
                'slow_vertex_speed_mps': mean_or_none((pxz[1:]-pxz[:-1]).norm(dim=-1)*30, selected),
                'source_slow_vertex_speed_mps': mean_or_none((sxz[1:]-sxz[:-1]).norm(dim=-1)*30, slow),
                'source_slow_intervals': int(slow.sum()), 'assessed_slow_intervals': int(selected.sum()),
                'invalid_predicted_points': int((~known).sum())}
            rows.append(row)
            if wi < 6 or item['origin'] == 'parc_generated':
                np.savez_compressed(output/f'window-{wi:02d}-{mode}.npz',
                    source=sp['posed_joints'][0].cpu().numpy(), prediction=pp['posed_joints'][0].cpu().numpy(),
                    roots_source=source[0, :, :3].cpu().numpy(), roots_prediction=generated[0, :, :3].cpu().numpy(),
                    local_source=sp['local_joint_rots'][0].cpu().numpy(), local_prediction=pp['local_joint_rots'][0].cpu().numpy(),
                    hf=torch.as_tensor(scene['heightfield_y_m']).cpu().numpy(),
                    min_point=torch.as_tensor(scene['min_point_world_xyz_m']).cpu().numpy(), dx=float(scene['dx_m']), fps=30)
    # One unchanged held-out quiet seed; source motion never resets this rollout.
    for item in validation:
        if item['origin'] != 'bones_seed' or item['category'] != 'idle':
            continue
        world = item['world'].to(pilot.inference.device)
        if float((world[:, 1:, :3]-world[:, :-1, :3]).norm(dim=-1).max()*30) > .08:
            continue
        original_scene = data['scenes'][item['scene']]
        scene = dict(heightfield_y_m=torch.zeros(401, 401, device=world.device),
            min_point_world_xyz_m=torch.tensor([float(world[0, 3, 0])-40, 0., float(world[0, 3, 2])-40], device=world.device), dx_m=.2)
        if torch.count_nonzero(torch.as_tensor(original_scene['heightfield_y_m'])):
            raise ValueError('Declared flat rest seed is not flat')
        for behavior in ('rest', 'start_stop'):
            history = world[:, :4]
            parts = []
            # Four seconds of travel for start/stop, then 30 seconds at rest.
            travel = 15 if behavior == 'start_stop' else 0
            for block in range(travel+113):
                heading = world[0, 3, 3:5]
                velocity = 1.6 if block < travel else 0.
                goal = torch.stack((heading[1]*velocity, heading[0]*velocity, heading[0], heading[1]))[None]
                prediction, _ = pilot.request(history, goal, scene)
                parts.append(prediction[:, 4:12])
                history = reencode_played_history(prediction[:, 8:12], rep)
            motion = torch.cat(parts, 1)
            parts_y, parts_xz = [], []
            for start_frame in range(0, motion.shape[1], 64):
                _, y, xz = geometry(pilot, motion[:, start_frame:start_frame+64], skin, indices)
                parts_y.append(y); parts_xz.append(xz)
            y, xz = torch.cat(parts_y), torch.cat(parts_xz)
            gap, known = terrain_gap(y, xz, scene)
            start = travel*8 + 60
            roots = motion[0, start:, :3]
            foot_gap = torch.stack((gap[:, :len(left)].amin(-1), gap[:, len(left):].amin(-1)), -1)[start:]
            rest_rows.append({'behavior': behavior, 'seconds_after_stop': 113*8/30,
                'source_id': item['source_id'], 'source_start': item['start'],
                'drift_max_m': float((roots[:, [0, 2]]-roots[:1, [0, 2]]).norm(dim=-1).max()),
                'root_y_range_m': float(roots[:, 1].max()-roots[:, 1].min()),
                'penetration_p99_m': float(torch.quantile(torch.relu(-foot_gap).flatten(), .99)),
                'higher_foot_gap_median_m': float(foot_gap.max(-1).values.median())})
            np.savez_compressed(output/(behavior+'.npz'), roots=motion[0, :, :3].cpu().numpy(),
                                sole_gaps_m=foot_gap.cpu().numpy(), fps=30)
        break
    keys = ('root_y_mae_m', 'fk_mpjpe_m', 'penetration_mean_m', 'slow_vertex_speed_mps')
    summary = {mode: {key: float(np.mean([r[key] for r in rows if r['history'] == mode and r[key] is not None]))
        if any(r[key] is not None for r in rows if r['history'] == mode) else None for key in keys}
        for mode in ('teacher_history', 'played_history')}
    result = {'summary': summary, 'rows': rows, 'rest': rest_rows, 'oracle': oracle_rows,
        'quality_approved': False, 'scope': 'small held-out decoder ablation; not parkour certification',
        'rest_scope': '30-second stop test, not the A100 60-second benchmark',
        'foot_evidence': 'source geometry and same-vertex speed, not measured contact forces',
        'no_valid_rest_seed': not bool(rest_rows), 'postprocessing': None}
    save(output/'metrics.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=2333)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 1000:
        p.error('Fresh bounded pilot required, at most1000 updates')
    gpu = select_gpu(args.gpu)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    receipt = json.loads((args.data/'manifest.json').read_text())
    if sha256(args.source) != receipt['source_bundle_sha256'] or sha256(args.data/'windows.pt') != receipt['windows_sha256']:
        raise ValueError('Pilot input identity changed')
    inference, bundle = load_terrain_bundle(args.source, 'cuda:0')
    data = torch.load(args.data/'windows.pt', map_location='cpu', weights_only=False)
    terrain, feedback = ARMS[args.arm]
    pilot = DecoderExperiment(inference, bundle, terrain=terrain)
    frozen = frozen_state(pilot)
    args.output.mkdir(parents=True)
    contract = {'arm': args.arm, 'source_bundle_sha256': receipt['source_bundle_sha256'],
        'data_manifest_sha256': sha256(args.data/'manifest.json'), 'steps': args.steps, 'seed': args.seed,
        'gpu_at_launch': gpu, 'batch_windows': 4, 'primitives_per_update': 2, 'committed_frames_per_primitive': 8,
        'optimizer': 'fresh Adam5e-5, clip0.5; bounded decoder adaptation, not the paper schedule',
        'trainable': 'native decoder and optional terrain latent adapter', 'frozen': frozen,
        'loss': 'native MB reconstruction/contact-velocity/joint-velocity; commit zero for frozen codes',
        'feedback_schedule': '100 teacher updates, linear ramp100..400, then generated second-primitive history',
        'history_gradients': 'detached played FK, no cross-primitive gradient',
        'teacher_input_cache': 'exact frozen Root/Pose requests cached; decoder output always recomputed',
        'future_targets': 'source world and clock retained; history never replaces target motion',
        'postprocessing': None, 'quality_approved': False}
    save(args.output/'contract.json', contract)
    if args.arm == 'baseline':
        evaluate(pilot, bundle, data, args.output/'initial_review', oracle=True)
    training = [w for w in data['windows'] if w['split'] == 'train']
    rng = np.random.default_rng(args.seed)
    digest = hashlib.sha256()
    optimizer = torch.optim.Adam(pilot.parameters(), lr=5e-5)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    counts = {'teacher_second': 0, 'generated_second': 0}
    request_cache = {}
    try:
        for step in range(args.steps):
            chosen = rng.integers(len(training), size=4)
            draws = rng.random(4)
            digest.update(chosen.tobytes()); digest.update(draws.tobytes())
            probability = min(1., max(0., (step-100)/300)) if feedback else 0.
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for index, draw in zip(chosen, draws):
                item = training[index]
                world = item['world'].to(inference.device)
                scene = data['scenes'][item['scene']]
                history = world[:, :4]
                generated_history = False
                for primitive, offset in enumerate((0, 8)):
                    goal = evaluation_goal(inference, world, offset, bundle['protocol']['version'])
                    key = (int(index), offset)
                    if generated_history or key not in request_cache:
                        _, call = pilot.request(history, goal, scene)
                        if not generated_history:
                            request_cache[key] = call
                    else:
                        call = request_cache[key]
                    losses, decoded = pilot.loss(world[:, offset:offset+33], history, call)
                    loss = losses['loss']/8
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite native decoder objective')
                    loss.backward()
                    total += float(loss.detach())
                    if primitive == 0:
                        use_generated = draw < probability
                        if use_generated:
                            with torch.no_grad():
                                generated = pilot.restore_world(decoded.detach(), history, call['observation'])
                                history = reencode_played_history(generated[:, 8:12], inference.global_motion_rep)
                        else:
                            history = world[:, 8:12]
                        generated_history = use_generated
                        counts['generated_second' if use_generated else 'teacher_second'] += 1
            norm = torch.nn.utils.clip_grad_norm_(pilot.parameters(), .5)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite decoder gradients')
            optimizer.step()
            if step == 0 or (step+1) % 50 == 0:
                record = {'status': 'training', 'step': step+1, 'loss': total,
                    'elapsed_seconds': time.perf_counter()-started, 'feedback_probability': probability,
                    'terrain_gate': float(pilot.adapter.attention.gate), 'counts': dict(counts)}
                save(args.output/'progress.json', record);print(json.dumps(record), flush=True)
        if frozen_state(pilot) != frozen:
            raise ValueError('Frozen Root/Pose or token vocabulary changed')
        checkpoint = args.output/'decoder.pt'
        torch.save({'decoder': pilot.codec.decoder.state_dict(), 'adapter': pilot.adapter.state_dict(),
                    'contract': contract, 'updates': args.steps}, checkpoint)
        save(args.output/'training_result.json', {'updates': args.steps, 'counts': counts,
            'sampling_sha256': digest.hexdigest(), 'checkpoint_sha256': sha256(checkpoint),
            'elapsed_seconds': time.perf_counter()-started,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2})
        evaluate(pilot, bundle, data, args.output/'review')
        save(args.output/'state.json', {'status': 'completed_review_required', 'quality_approved': False})
    except Exception as error:
        save(args.output/'state.json', {'status': 'failed', 'error': type(error).__name__+': '+str(error)})
        raise
    finally:
        pilot.close()


if __name__ == '__main__':
    main()
