"""Trace stair anticipation on the unchanged live model, without wall-clock UI.

Shadow replans and flat counterfactuals never enter played history. The scene
origin is logged separately from the learned relative Root output.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE
from motionbricks.data.unified_scene_field import field_from_heightfield
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def first_riser(scene, x, z, threshold=.08):
    height = torch.as_tensor(scene['heightfield_y_m']).cpu().numpy()
    origin = torch.as_tensor(scene['min_point_world_xyz_m']).cpu().numpy()
    dx = float(scene['dx_m'])
    ix, iz = np.floor((np.asarray([x, z]) - origin[[0, 2]]) / dx + .5).astype(int)
    candidates = np.flatnonzero(height[iz:, ix] > height[iz, ix] + threshold)
    if not len(candidates):
        raise ValueError('No forward riser on the default straight approach')
    row = iz + int(candidates[0])
    return dict(z_m=float(origin[2] + (row - .5) * dx),
                floor_y_m=float(height[iz, ix]), first_tread_y_m=float(height[row, ix]))


@torch.no_grad()
def fresh_trace(controller, history, command, scene):
    result = controller(history, command, scene)
    origin = vertical_origin(controller.inference, history, result['observation'])
    relative = controller.inference.global_motion_rep.unnormalize(
        result['root_output']['pred_global_root_values'])[:, :32]
    root = relative[..., :3].clone()
    root[..., 1] += origin[:, None]
    root[..., [0, 2]] += history[:, :1, [0, 2]]
    volume = controller.volume_observation
    distance, points, valid = volume['distance'], volume['points'], volume['valid']
    plane_distance = points[..., 1]
    row = dict(origin_y_m=float(origin.item()),
        root_future_max_world_y_m=float(root[:, 4:28, 1].max()),
        root_future_max_relative_y_m=float(relative[:, 4:28, 1].max()),
        root_future_end_world_y_m=float(root[0, 27, 1]),
        root_forecast_end_z_m=float(root[0, 27, 2]),
        decoded_future_max_y_m=float(result['world'][:, 4:28, 1].max()),
        sdf_difference_from_y0_plane_m=float((distance[valid] - plane_distance[valid]).abs().max()),
        sdf_known_points=int(valid.sum()))
    return result, root, row


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(14292)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hashes = {key: sha256(getattr(args, key)) for key in ('bundle', 'candidate', 'scene')}
    expected = dict(bundle='0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d',
        candidate='ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979',
        scene='e19876a349c54ba4a2c663b358f9e33bc0bab2e115f302e91ff3f63b7f819ffa')
    if hashes != expected:
        raise ValueError('Must reproduce the current live bundle, candidate and world')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    checkpoint = torch.load(args.candidate, map_location='cpu', weights_only=False)
    with np.load(args.scene, allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(archive[key].copy())
                 for key in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
        history = torch.from_numpy(archive['initial_history'].copy()).to(args.device)
    receipt = json.loads(args.scene_receipt.read_text())
    if receipt.get('scene_sha256') != hashes['scene']:
        raise ValueError('Spawn receipt belongs to another scene')
    spawn = next(s for s in receipt['spawns'] if s['id'] == 'zone-03')
    if abs(float(spawn['heading'])) > 1e-8:
        raise ValueError('This bounded audit is for the known +Z approach')
    rep = inference.global_motion_rep
    history = rep.change_first_heading(history, float(spawn['heading']), False, False)
    history[..., [0, 2]] += history.new_tensor(spawn['spawn_xz']) - history[:, -1:, [0, 2]].clone()
    riser = first_riser(scene, *spawn['spawn_xz'])
    controller = UnifiedTerrainController(inference, bundle)
    controller.adaptation.load_state_dict(checkpoint['adaptation'], strict=True)
    controller.eval().requires_grad_(False)
    keys = list(controller.state_dict())
    before = state_digest(controller, keys)
    geometry = JointGeometryObjective(rep, args.device)
    field = field_from_heightfield(scene)
    flat = dict(scene, heightfield_y_m=torch.full_like(scene['heightfield_y_m'], riser['floor_y_m']))
    actor = UnifiedTerrainBackend(controller, history, scene)
    actor.configure_runtime(dict(PROFILE))
    command = history.new_tensor([[0., 1.6, 1., 0.]])
    pieces, rows = [], []
    for block in range(args.blocks):
        observed = actor.history.clone()
        output = actor.predict_plan(command[0, :2], 0.)
        played = output['planned_motion'].clone()
        neural = bool(actor.last_generation['neural_generation'])
        primary_root = torch.tensor(output['debug']['root_stage_world_xyz_m'], device=args.device)
        shadow, root, trace = fresh_trace(controller, observed, command, scene)
        if neural:
            torch.testing.assert_close(played, shadow['world'][:, 4:12], atol=3e-5, rtol=1e-5)
            torch.testing.assert_close(primary_root, root[0, 4:28], atol=3e-5, rtol=1e-5)
        counterfactual = None
        if block * 8 >= 40 and float(observed[0, -1, 2]) < riser['z_m'] - .2:
            # This is a scene counterfactual, not a labelled flat demonstration.
            _, flat_root, flat_trace = fresh_trace(controller, observed, command, flat)
            replay, replay_root, _ = fresh_trace(controller, observed, command, scene)
            torch.testing.assert_close(replay['world'], shadow['world'], atol=0, rtol=0)
            torch.testing.assert_close(replay_root, root, atol=0, rtol=0)
            counterfactual = dict(flat_root_max_y_m=flat_trace['root_future_max_world_y_m'],
                real_minus_flat_mean_root_y_m=float((root[:, 4:28, 1] - flat_root[:, 4:28, 1]).mean()),
                real_minus_flat_max_abs_root_y_m=float((root[:, 4:28, 1] - flat_root[:, 4:28, 1]).abs().max()),
                origin_difference_m=trace['origin_y_m'] - flat_trace['origin_y_m'],
                not_published=True, after_counterfactual_replay_error=0.)
        _, surface = geometry.geometry(played)
        distance, _, valid = field.query(surface)
        penetration = torch.where(valid, (-distance).clamp_min(0), 0)
        sole = penetration[:, :, :geometry.sole_count]
        local_hit = (sole.amax(-1)[0] > .02).nonzero().flatten()
        row = dict(start_frame=block * 8, neural_generation=neural,
            observed_root_y_m=float(observed[0, -1, 1]), observed_root_z_m=float(observed[0, -1, 2]),
            distance_to_first_riser_m=riser['z_m'] - float(observed[0, -1, 2]),
            primary_plan_max_y_m=float(primary_root[:, 1].max()),
            primary_next8_max_y_m=float(primary_root[:8, 1].max()),
            fresh_shadow=trace, counterfactual=counterfactual,
            shadow_next8_max_y_m=float(root[0, 4:12, 1].max()),
            shadow_minus_cached_first_root_y_m=float(root[0, 4, 1] - primary_root[0, 1]),
            played_max_y_m=float(played[..., 1].max()),
            sole_penetration_peak_m=float(sole.max()), sampled_body_penetration_peak_m=float(penetration.max()),
            first_sole_hit_2cm_frame=block * 8 + int(local_hit[0]) if len(local_hit) else None,
            unknown_surface_points=int((~valid).sum()))
        rows.append(row)
        pieces.append(played)
        actor.history = output['published_history'].clone()
        write_json(args.output / 'progress.json', dict(rows=rows, first_riser=riser))
        print(f"frame={block*8} z={row['observed_root_z_m']:.3f} origin={trace['origin_y_m']:.3f} "
              f"root_max={trace['root_future_max_world_y_m']:.3f} relative={trace['root_future_max_relative_y_m']:.3f} "
              f"sole_peak={row['sole_penetration_peak_m']:.3f}", flush=True)
    world = torch.cat(pieces, 1)
    if state_digest(controller, keys) != before:
        raise ValueError('Diagnostic changed weights')
    np.savez_compressed(args.output / 'played.npz', global_features=world[0].cpu().numpy(), fps=np.int32(30))
    hits = [r['first_sole_hit_2cm_frame'] for r in rows if r['first_sole_hit_2cm_frame'] is not None]
    rise = (world[0, :, 1] > history[0, -1, 1] + .1).nonzero().flatten()
    write_json(args.output / 'complete.json', dict(status='complete', hashes=hashes, profile=PROFILE,
        first_riser=riser, rows=rows, frames=world.shape[1], first_sole_hit_2cm_frame=min(hits) if hits else None,
        first_root_rise_10cm_frame=int(rise[0]) if len(rise) else None,
        first_root_rise_10cm_z_m=float(world[0, rise[0], 2]) if len(rise) else None,
        root_at_frame191=world[0, 191, :3].tolist() if world.shape[1] >= 192 else None,
        geometry_scope=dict(sole_vertices_per_frame=geometry.sole_count,
            total_sampled_vertices_per_frame=len(geometry.indices), temporal_substeps=1,
            full_skin_collision_certificate=False),
        timing_scope='Logical playback frames, no browser or network waiting. Fresh shadows are never published.',
        optimizer_updates=0, weights_unchanged=True, runtime_modified=False, quality_approved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'candidate', 'scene', 'scene-receipt', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--blocks', type=int, default=30)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise
