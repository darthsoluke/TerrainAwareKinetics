import argparse, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import FPS, G, foot_indices, segment_com_world

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=pathlib.Path, required=True)
parser.add_argument('--pfnn', type=pathlib.Path, required=True)
parser.add_argument('--stats', type=pathlib.Path, required=True)
parser.add_argument('--proxy', type=pathlib.Path, required=True)
parser.add_argument('--row', type=int, default=0)
args = parser.parse_args()
rep = build_native_representation(args.stats)
proxy = dict(np.load(args.proxy, allow_pickle=True))
mass = np.array(proxy['mass'], dtype=np.float64); com = np.array(proxy['com'], dtype=np.float64)
neutral = rep.skeleton.neutral_joints.detach().cpu().numpy()
bad = (mass <= 0.0) | (np.linalg.norm(com - neutral, axis=-1) > 1.0)
com[bad] = neutral[bad]; mass[bad] = 0.0
proxy['mass'] = mass; proxy['com'] = com
dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, 'validation')
world = torch.from_numpy(dataset[args.row]['world'].numpy())[None]
pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
joints = pose['posed_joints'][0]; rots = pose['global_joint_rots'][0]
coms, m = segment_com_world(joints, rots, proxy, neutral)
coms = coms.detach().cpu().numpy(); m = m.detach().cpu().numpy()
total = m.sum()
centre = (coms * m[:, None]).sum(1) / total
T = centre.shape[0]
v = np.zeros_like(centre); v[1:-1] = (centre[2:] - centre[:-2]) * FPS / 2.0
a = np.zeros_like(centre); a[1:-1] = (v[2:] - v[:-2]) * FPS / 2.0
one_frame = np.zeros(T); one_frame[1:] = (centre[1:, 1] - centre[:-1, 1]) * FPS
print('T=%d  com_y range [%.3f, %.3f]' % (T, centre[:,1].min(), centre[:,1].max()))
print('frame   com_y    v_y     a_y(central)  a_y(oneframe-diff)  linear_y')
for t in range(min(T, 14)):
    print('%5d  %7.4f %7.3f  %9.2f  %14.2f  %9.2f' % (t, centre[t,1], v[t,1], a[t,1],
          (one_frame[t] - (one_frame[t-1] if t else one_frame[t])) * FPS, a[t,1] + 9.81))
print('median a_y %.2f   median linear_y %.2f   fraction linear_y<0 %.2f' % (
      np.median(a[1:-1,1]), np.median(a[1:-1,1]) + 9.81, float((a[1:-1,1] + 9.81 < 0).mean())))
