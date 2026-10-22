"""Offline factor interventions, not a deployable character controller."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_native_mb import sha256


def world_from_decoder(infer, local, history, observation):
    world = infer.motion_rep.dual_rep.local_to_global(local[:, :32], True, False,
        lengths=torch.full((1,), 32, device=local.device, dtype=torch.long))
    offsets, heading, _ = infer._extract_initial_root_info(history[..., :5])
    world = infer._reapply_initial_root_info(dict(pred_global_poses=world,
        reference_start_root_global_offsets=offsets, reference_start_root_global_heading=heading))
    return shift_global_motion(world, infer.global_motion_rep, vertical_origin(infer, history, observation))


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.checkpoint) != 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf':
        raise ValueError('Not the user-approved moving weights')
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(saved)
    infer, rep = model.inference, model.inference.global_motion_rep
    dataset = Path(saved['contract']['dataset'])
    data = TerrainLearningDataset(dataset, build_native_representation(dataset/'stats'), 'train')
    indexed = {row['source_id']:i for i,row in enumerate(data.records)}
    contract = json.loads((args.run/'contract.json').read_text())
    prototypes = []
    for row in contract['source_cases']:
        if row['phase'] != 'rest':
            continue
        item = data.full_reference_item(indexed[row['source_id']])
        source = item['world'][None, row['source_start']:row['source_end']].cuda()
        history = source[:, :4]
        yaw = torch.atan2(history[:, 3, 4], history[:, 3, 3])
        task = SoftCommandState().intent(history, [0.,0.], yaw, 8)
        result = model(history, task, [item['scene']], source=source)
        tokens = model.codec.encode_into_idx(result['target']['local_motions'], fetch_overall_indices=False)
        prototypes.append(dict(tokens=tokens, source_id=row['source_id'], source_start=row['source_start']))
    receipt = json.loads((args.artifact/'export.json').read_text())
    geometry = JointGeometryObjective(rep, 'cuda:0')
    external_indices = extract_feature_from_motion_rep(None, model.codec.motion_rep,
        model.codec.decoder_external_cond_feature_mode, fetch_feat_idx=True)
    velocity_indices = np.asarray(model.codec.motion_rep.indices['local_root_vel']).reshape(-1)
    channels = [list(external_indices).index(int(index)) for index in velocity_indices]
    rows = []
    for entry in receipt['entries']:
        with np.load(args.artifact/(entry['entry']+'-scene.npz'), allow_pickle=False) as archive:
            initial = torch.from_numpy(archive['initial_history'].copy()).cuda()
            scene = {key:torch.from_numpy(archive[key].copy()).cuda()
                     for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        with np.load(args.run/'baseline'/(entry['entry']+'-release_resume.npz'), allow_pickle=False) as archive:
            played = torch.from_numpy(archive['world'].copy())[None].cuda()
        driver = SoftCommandState()
        for block in range(13):
            history = initial if block == 0 else reencode_played_history(played[:, block*8-4:block*8], rep)
            task = driver.intent(history, entry['command_velocity_mps'] if block < 3 else [0.,0.],
                entry['command_heading'], 8)
            if block not in (3,7,12):
                continue
            with capture_decoder_call(model.codec) as calls:
                result = model(history, task, [scene])
            torch.testing.assert_close(result['world'][:,4:12], played[:,block*8:block*8+8], atol=3e-5, rtol=1e-5)
            call = calls[-1]
            if call['kwargs']['has_target_cond'][:,4:].any():
                raise ValueError('An unknown future body pose entered the actor')
            external = call['kwargs']['external_cond']
            external_raw = model.codec.motion_rep.stats.unnormalize(external, index=external_indices)
            observed_local = infer.motion_rep.dual_rep.global_to_local(history, False, False,
                lengths=torch.full((1,),4,device=history.device,dtype=torch.long))
            observed_external = extract_feature_from_motion_rep(observed_local, model.codec.motion_rep,
                model.codec.decoder_external_cond_feature_mode).clone()
            height_channel = list(external_indices).index(int(model.codec.motion_rep.indices['global_root_y'][0]))
            observed_external[...,height_channel] -= vertical_origin(infer,history,result['observation'])[:,None]
            past_velocity_error = (observed_external[:,:3,channels]-external_raw[:,:3,channels]).abs().max()
            root = rep.unnormalize(result['root_output']['pred_global_root_values'])
            expected_velocity = torch.diff(root[:,:32,[0,2]], dim=1)*30
            difference = (external_raw[:,:31,channels]-expected_velocity).abs().max()
            if difference > 3e-4:
                raise ValueError('Decoder velocity fields disagree with Root prediction')
            modes = [('replay', None, False), ('zero_velocity_condition', None, True)]
            modes += [(f'rest_tokens_{i}', prototype, False) for i,prototype in enumerate(prototypes)]
            modes += [(f'rest_tokens_zero_velocity_{i}', prototype, True) for i,prototype in enumerate(prototypes)]
            if args.factual_past:
                modes = [('replay',None,False), ('factual_past',None,False),
                    ('factual_past_zero_velocity',None,True),
                    ('rest_tokens_factual_past',prototypes[0],False),
                    ('rest_tokens_factual_past_zero_velocity',prototypes[0],True)]
            for mode, prototype, zero in modes:
                tokens = call['args'][0].clone()
                kwargs = dict(call['kwargs'])
                if prototype is not None:
                    tokens[:,:prototype['tokens'].shape[1]] = prototype['tokens']
                if 'factual_past' in mode:
                    changed_past = external_raw.clone()
                    changed_past[:,:3,channels] = observed_external[:,:3,channels]
                    changed_past[:,:4,height_channel] = observed_external[:,:4,height_channel]
                    normalized_past = model.codec.motion_rep.stats.normalize(changed_past,index=external_indices)
                    kwargs['external_cond'] = external.clone()
                    kwargs['external_cond'][:,:3,channels] = normalized_past[:,:3,channels]
                    kwargs['external_cond'][:,:4,height_channel] = normalized_past[:,:4,height_channel]
                if zero:
                    changed = model.codec.motion_rep.stats.unnormalize(kwargs['external_cond'],index=external_indices).clone()
                    # Interval3 is future: observed history contains only intervals0..2.
                    changed[:,3:32,channels] = 0
                    normalized = model.codec.motion_rep.stats.normalize(changed, index=external_indices)
                    previous_external = kwargs['external_cond'].clone()
                    kwargs['external_cond'] = previous_external.clone()
                    kwargs['external_cond'][:,3:32,channels] = normalized[:,3:32,channels]
                    untouched = [i for i in range(external.shape[-1]) if i not in channels]
                    torch.testing.assert_close(kwargs['external_cond'][... ,untouched], previous_external[...,untouched], atol=0, rtol=0)
                    torch.testing.assert_close(kwargs['external_cond'][:,:3], previous_external[:,:3], atol=0, rtol=0)
                local = model.codec.forward_decoder(tokens, **kwargs)['recon_state'][:,:32]
                world = world_from_decoder(infer, local, history, result['observation'])
                if mode == 'replay':
                    torch.testing.assert_close(world, result['world'], atol=3e-5, rtol=1e-5)
                speed = torch.diff(world[..., [0,2]], dim=1).norm(dim=-1)*30
                row = dict(entry=entry['entry'], stop_block=block-3, mode=mode,
                    external_root_velocity_max_error_mps=float(difference),
                    input_past_velocity_error_mps=float(past_velocity_error),
                    external_published_mean_speed_mps=float(external_raw[:,4:11,channels].norm(dim=-1).mean()),
                    decoded_published_mean_speed_mps=float(speed[:,4:11].mean()),
                    decoded_horizon_tail_mean_speed_mps=float(speed[:,23:31].mean()),
                    reconstructed_history_xz_error_m=float((world[:,3,[0,2]]-history[:,3,[0,2]]).norm()),
                    first_future_step_speed_mps=float((world[:,4,[0,2]]-history[:,3,[0,2]]).norm()*30),
                    reference_source_id=prototype['source_id'] if prototype else None,
                    offline_only=True, quality_approved=False,
                    scope='Reference rest tokens are real BONES, NOT the true future of this stair state. No oracle is deployed; only declared decoder inputs differ.')
                row.update(exact_skin_metrics(geometry, world[:,4:32], scene))
                rows.append(row)
                save_motion(args.output/(entry['entry']+f'-block{block}-{mode}.npz'), world[:,4:32], scene, rep, row)
            (args.output/'progress.json').write_text(json.dumps(dict(rows=rows), indent=2))
            print(json.dumps(dict(entry=entry['entry'], block=block, records=len(rows))), flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(rows=rows, updates=0, offline_only=True,
        velocity_condition_channels=channels, external_representation_indices=list(map(int,external_indices)),
        quality_approved=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','run','artifact','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--factual-past', action='store_true')
    main(parser.parse_args())
