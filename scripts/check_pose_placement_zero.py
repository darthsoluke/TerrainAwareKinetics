"""Assert matched zero-residual baselines before a learned Pose-placement pair."""
import argparse
from pathlib import Path
import torch

from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_training_goals import source_terminal_goal
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
    initial = torch.load(a.initial, map_location='cpu', weights_only=False)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    references, rows = {}, []
    for stage in ('input', 'output'):
        controller = UnifiedTerrainController(inference, bundle, pose_condition_stage=stage)
        controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
        controller.adaptation['pose'].output.weight.zero_()
        for case in cases:
            source = case['world'][None, :33].to(a.device)
            result = controller(source[:, :4], source_terminal_goal(source[..., :5]), case['scene'])
            values = (result['world'].clone(), result['pose_logits'].clone())
            if stage == 'input':
                references[case['label']] = values
            else:
                for actual, expected in zip(values, references[case['label']]):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                rows.append(dict(label=case['label'], forward_difference=0., token_logit_difference=0.))
        controller.close()
    write_json(a.output / 'complete.json', dict(rows=rows, optimizer_updates=0,
        initial_sha256=sha256(a.initial), matched_baselines=True, live_runtime_changed=False,
        note='Both remove only the old Pose residual by zeroing the same output matrix. '
             'They match each other, not a claim of bitwise identity to the unchanged live nonzero-residual checkpoint.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('bundle', 'initial', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
