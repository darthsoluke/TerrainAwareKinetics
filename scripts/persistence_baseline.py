import argparse, json, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import foot_indices

parser = argparse.ArgumentParser()
for name in ('cache', 'pfnn', 'stats', 'proxy'):
    parser.add_argument('--' + name, type=pathlib.Path, required=True)
parser.add_argument('--limit', type=int, default=16)
parser.add_argument('--start', type=int, default=16)
parser.add_argument('--steps', type=int, default=17)
args = parser.parse_args()
rep = build_native_representation(args.stats)
proxy = dict(np.load(args.proxy, allow_pickle=True))
feet_idx = foot_indices(proxy)
dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, 'validation')
feet_all, body_all, root_all = [], [], []
for i in range(min(args.limit, len(dataset))):
    world = torch.from_numpy(dataset[i]['world'].numpy())[None]
    pose = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')
    joints = pose['posed_joints'][0]
    stop = min(args.start + args.steps, joints.shape[0])
    step = joints[args.start:stop] - joints[args.start - 1:stop - 1]
    feet_all.append(step[:, feet_idx].norm(dim=-1).reshape(-1).numpy())
    body_all.append(step.norm(dim=-1).reshape(-1).numpy())
    root_all.append(step[:, 0].norm(dim=-1).numpy())
feet = np.concatenate(feet_all)
body = np.concatenate(body_all)
root = np.concatenate(root_all)
out = dict(
    frames=int(feet.size),
    persistence_feet_p50_m=float(np.median(feet)),
    persistence_feet_p90_m=float(np.percentile(feet, 90)),
    persistence_body_p50_m=float(np.median(body)),
    persistence_body_p90_m=float(np.percentile(body, 90)),
    persistence_root_p50_m=float(np.median(root)),
    root_speed_p50_mps=float(np.median(root) * 30.0))
print(json.dumps(out, indent=2))
