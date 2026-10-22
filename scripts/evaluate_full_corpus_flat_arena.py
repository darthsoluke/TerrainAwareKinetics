"""Run the unchanged36-second command script on a declared fixed64m flat arena."""

import argparse
import os
from pathlib import Path

from scripts.run_pfnn_parallel_workflow import gpu_snapshot
from scripts.run_pfnn_bias_workflow import save, digest


def main(a):
    snapshot, available = gpu_snapshot()
    if a.gpu not in available:
        raise ValueError('Selected evaluation GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective
    from motionbricks.data.unified_terrain import UnifiedTerrainController, PROFILE
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.evaluate_full_corpus_controller import flat_review

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    idle = dict(next(case for case in cases if case['name'] == 'idle'))
    original_scene = idle['scene']
    height = torch.as_tensor(original_scene['heightfield_y_m'])
    if not torch.isfinite(height).all() or float(height.max()-height.min()) > 1e-8:
        raise ValueError('The retained idle source must have a constant measured plane')
    floor = float(height[0, 0])
    idle['scene'] = dict(heightfield_y_m=torch.full((321, 321), floor),
        min_point_world_xyz_m=torch.tensor([-32., floor, -32.]), dx_m=.2)
    save(a.output/'arena.json', dict(static_arena_before_generation=True, side_m=64., dx_m=.2,
        floor_y_m=floor, commanded_nominal_distance_m=22.4, seed_frames_unchanged=True,
        original_heightfield_shape=list(height.shape), original_dx_m=float(original_scene['dx_m']),
        reason='The retained8.2m square idle scene cannot contain the declared36-second command script.',
        terrain_follows_prediction=False, original_failed_flat_recording_retained=True))
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    bundle_hash, checkpoint_hash = digest(a.bundle), digest(a.checkpoint)
    receipt = inference_receipt(saved, checkpoint_hash, checkpoint_hash, bundle_hash)
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    load_candidate(controller, saved, receipt, bundle_hash)
    before = tensor_digest(controller.state_dict())
    try:
        geometry = JointGeometryObjective(inference.global_motion_rep, 'cuda:0')
        with torch.no_grad():
            result = flat_review(controller, idle, geometry, a.output, False)
        if tensor_digest(controller.state_dict()) != before:
            raise ValueError('Flat-arena evaluation changed model tensors')
        save(a.output/'complete.json', dict(evaluation_completed=True, flat=result, runtime_profile=dict(PROFILE),
            checkpoint_sha256=checkpoint_hash, cases_sha256=digest(a.cases), weights_unchanged=True,
            actual_optimizer_updates=0, gpu=a.gpu, gpu_snapshot_before=snapshot,
            quality_approved=False, delivery_approved=False))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        raise
