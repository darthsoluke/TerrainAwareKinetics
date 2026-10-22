"""Conditional source-prefix fit diagnostic, never a free-control quality claim."""

import argparse
from pathlib import Path
import torch

from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import source_support_evidence, heightfield_gap
from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(saved)
    cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    rows = []
    for case in cases:
        source = case['world'][None].cuda()
        sj, sp = geometry.geometry(source)
        support, _ = source_support_evidence(sp[0, :, :geometry.sole_count], case['scene'])
        parts, token_accuracies, rollout_cross_entropies = [], [], []
        for block in range(case['training_segments']):
            history = source[:, block*8:block*8+4]
            intent = SoftCommandState().from_controls(history, control(case, block), 8)
            result = model(history, intent, [case['scene']], source=source[:, block*8:block*8+33])
            generated = result['world']
            tokens = model.codec.encode_into_idx(result['target']['local_motions'], fetch_overall_indices=False)
            logits = result['pose_logits'][:, :tokens.shape[1]]
            token_accuracies.append(float((logits.argmax(-1) == tokens).float().mean()))
            rollout_cross_entropies.append(float(torch.nn.functional.cross_entropy(logits.flatten(0,2), tokens.long().flatten())))
            parts.append(generated[:, 4:12])
        motion = torch.cat(parts, 1)
        pj, pp = geometry.geometry(motion)
        count = motion.shape[1]
        mask = support[4:4+count]
        predicted_soles = pp[0, :, :geometry.sole_count]
        true_soles = sp[0, 4:4+count, :geometry.sole_count]
        root_error = (motion[0, :, :3]-source[0, 4:4+count, :3]).norm(dim=-1)
        joint_error = (pj-sj[:, 4:4+count]).norm(dim=-1)
        point_error = (predicted_soles-true_soles).norm(dim=-1)
        gap, known = heightfield_gap(predicted_soles, case['scene'])
        rows.append(dict(label=case['label'], frames=count, root_mean_m=float(root_error.mean()),
            actor_token_accuracy=sum(token_accuracies)/len(token_accuracies),
            actor_rollout_ce=sum(rollout_cross_entropies)/len(rollout_cross_entropies),
            joint_mean_m=float(joint_error.mean()), joint_p99_m=float(torch.quantile(joint_error,.99)),
            true_support_vertex_frames=int(mask.sum()), support_position_mean_m=float(point_error[mask].mean()) if mask.any() else None,
            support_gap_abs_mean_m=float(gap[mask].abs().mean()) if mask.any() else None,
            support_within2cm_fraction=float((gap[mask].abs()<=.02).float().mean()) if mask.any() else None,
            unknown_points=int((~known).sum())))
    write_json(args.output/'complete.json', dict(checkpoint_sha256=sha256(args.checkpoint), rows=rows,
        optimizer_updates=0, actual_history_policy='offline source prefix reset at every8-frame chunk',
        future_truth_in_actor=False, deployable=False,
        scope='Conditional training-distribution fit only; concatenated diagnostic is not a continuous actor rollout.'))
    print(rows, flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
