"""Stage-local diagnostic of repeated authored landing poses, NOT a hold skill.

The oracle arm encodes the whole synthetic constant pose. It intentionally has
future information and must never be compared as a deployable controller.
"""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_smart_object import NativeSmartObject
from motionbricks.data.native_mb_frame import translate_global_height
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_native_mb_soma import file_sha256, load_bundle


def scores(source, output, foot_indices):
    a, b = source['posed_joints'][:, 4:], output['posed_joints'][:, 4:]
    return dict(root_height_bias_m=float((b[..., 0, 1] - a[..., 0, 1]).mean()),
                root_height_mae_m=float((b[..., 0, 1] - a[..., 0, 1]).abs().mean()),
                foot_joint_error_m=float((b[:, :, foot_indices] - a[:, :, foot_indices]).norm(dim=-1).mean()),
                terminal_foot_joint_error_m=float((b[:, -1, foot_indices] - a[:, -1, foot_indices]).norm(dim=-1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--bundle', type=Path)
    models.add_argument('--tokenizer-run', type=Path,
                        help='Source-encoded reconstruction only; never pair a new vocabulary with old Pose weights')
    parser.add_argument('--courses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, choices=(12, 64), default=12,
                        help='12 matches the observed runtime hold phase; 64 is a separate long-horizon stress test')
    parser.add_argument('--root-height', type=float,
                        help='Diagnostic common vertical translation of each constant pose, not a runtime correction')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    infer, identity = None, None
    if args.bundle:
        infer, _ = load_bundle(args.bundle, 'cuda:0')
        rep, local_rep, net = infer.global_motion_rep, infer.local_motion_rep, infer._vqvae_pose_model
        dual = infer.motion_rep.dual_rep
    else:
        from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint
        model, identity, _ = load_native_checkpoint(args.tokenizer_run, device='cuda')
        rep, local_rep, net = model.motion_rep, model.local_motion_rep, model.pose_net
        dual = rep.dual_rep
    feet = [rep.skeleton.bone_order_names.index(n) for n in ('LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')]
    report = dict(bundle_sha256=file_sha256(args.bundle) if args.bundle else None,
        tokenizer=identity, courses_sha256=file_sha256(args.courses),
        source_sha256=file_sha256(Path(__file__)), quality_approved=False,
        interpretation='Synthetic stationary endpoint probes, not observed training motions or physically validated standing poses. Oracle has dense future source tokens/root; not a runtime option.', cases=[])
    with torch.inference_mode():
        for course in json.loads(args.courses.read_text())['courses']:
            for label, frame in [('entry', course['initial_history'][-1]), ('exit', course['goals'][-1]['features'][-1])]:
                raw = torch.tensor(frame, device='cuda')[None, None].repeat(1, args.frames + 1, 1)
                original_height = float(raw[0, 0, 1])
                offset = 0. if args.root_height is None else args.root_height - original_height
                raw = translate_global_height(raw, rep, offset)
                canonical = rep.change_first_heading(raw, 0., False, False)
                local = dual.global_to_local(rep.normalize(canonical), True, True,
                    lengths=torch.tensor([args.frames + 1], device='cuda'))[:, :args.frames]
                mask = torch.zeros((1, args.frames), dtype=torch.bool, device='cuda')
                mask[:, [3, args.frames - 1]] = True
                codes = net.encode_into_idx(local)
                external = extract_feature_from_motion_rep(local, local_rep, net.decoder_external_cond_feature_mode)
                decoded = net.forward_decoder(codes, target_cond=local * mask[..., None],
                    has_target_cond=mask, external_cond=external)['recon_state']
                source_local = local_rep.inverse(local, True, return_all=True)
                oracle = local_rep.inverse(decoded, True, return_all=True)
                assert torch.isfinite(decoded).all()
                generated_scores = None
                if infer is not None:
                    torch.manual_seed(2333)
                    primitive = NativeSmartObject(infer, fixed_tokens=args.frames // 4)
                    generated, frames = primitive.predict(raw[:, :4], raw[:, :4])
                    assert frames == args.frames and torch.isfinite(generated).all()
                    source_global = rep.inverse(raw[:, :args.frames], False, return_all=True)
                    generated_global = rep.inverse(generated[:, :args.frames], False, return_all=True)
                    generated_scores = scores(source_global, generated_global, feet)
                row = dict(course=course['id'], endpoint=label, frames=args.frames,
                    original_root_height_m=original_height, coordinate_translation_y_m=offset,
                    source_root_height_m=float(raw[0, 0, 1]),
                    source_encoded_oracle=scores(source_local, oracle, feet),
                    native_generated=generated_scores)
                report['cases'].append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
