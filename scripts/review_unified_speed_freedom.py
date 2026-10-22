"""Allow natural execution pace; test requests without output-speed gates."""
import argparse
from pathlib import Path
import torch

from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_pose_training import load_pose_checkpoint
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_unified_goal_alignment import load_world, world_review
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    initial = torch.load(a.initial, map_location='cpu', weights_only=False)
    controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
    scene, seed, placed, spawn = load_world(a, inference.global_motion_rep, a.device)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    rows = {}
    for arm in ('before', 'mask'):
        if arm == 'mask':
            load_pose_checkpoint(controller, torch.load(a.mask, map_location='cpu', weights_only=False), sha256(a.bundle))
        rows[arm] = world_review(controller, geometry, scene, placed, spawn, a.output / arm,
            max_blocks=225, speeds=(.4, .8, 1.6, 3.2, 5.))
        write_json(a.output / 'progress.json', dict(rows=rows))
    write_json(a.output / 'complete.json', dict(rows=rows, optimizer_updates=0, runtime_changed=False,
        requested_speed_samples=[.4, .8, 1.6, 3.2, 5.], generated_speed_limits=None,
        note='Requested speeds are diagnostic inputs. Generated movement may slow/accelerate freely. '
             'An unfinished60secondreview is reported as incomplete, not as proof a slower gait is invalid. '
             'Geometry/progress/lateral quality gates remain unchanged.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'mask', 'scene', 'scene-receipt', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
