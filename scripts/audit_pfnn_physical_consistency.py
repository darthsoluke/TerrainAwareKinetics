"""Model-free physical consistency audit on cached PFNN windows.

1. ballistic: within flight runs of at least four consecutive frames, fit a
   quadratic per axis and compare its second derivative with gravity. Short runs
   are discarded because a one- or two-frame flight cannot identify acceleration
   at 30 Hz without straddling a contact transition.
2. no-slip: world speed of each foot joint while its own contact label is on.
   Reported as a distribution because the reference itself is not slip-free.
"""
import argparse, json, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset

FPS = 30.0
G = np.array([0.0, -9.81, 0.0])
FOOT_ORDER = ('LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')
MIN_FLIGHT_RUN = 4


def segment_com_world(posed_joints, global_rots, proxy, neutral):
    mass = torch.as_tensor(proxy['mass'], dtype=posed_joints.dtype, device=posed_joints.device).clamp_min(0)
    com_bind = torch.as_tensor(proxy['com'], dtype=posed_joints.dtype, device=posed_joints.device)
    rest = torch.as_tensor(neutral, dtype=posed_joints.dtype, device=posed_joints.device)
    local = torch.einsum('...ji,...j->...i', global_rots, com_bind - rest)
    return posed_joints + local, mass


def foot_indices(proxy):
    names = [str(n) for n in proxy['names']]
    return [names.index(n) for n in FOOT_ORDER]


def flight_runs(mask, minimum):
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= minimum:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def friction_requirement(com, contacts, floor_fraction=0.2):
    """Minimum friction coefficient implied by the horizontal force demand.

    With an assumed vertical contact normal the linear momentum equation reduces to
    |sum f_horizontal| <= mu * sum f_vertical, so the implied demand is
    mu* = |d v_horizontal| / (d v_vertical + g dt). This is the friction a flat
    ground would have to supply; on tilted terrain it is a lower bound.
    Ground that cannot push (d v_vertical + g dt <= 0) is reported separately
    instead of producing a negative or infinite ratio.
    """
    n = min(len(com), len(contacts))
    com, contacts = com[:n], contacts[:n]
    velocity = np.zeros_like(com)
    velocity[1:-1] = (com[2:] - com[:-2]) * FPS / 2.0
    delta_v = velocity[1:] - velocity[:-1]
    # f/m = dv/dt - g, so the vertical force per unit mass is dv_y*FPS + 9.81.
    # An earlier version subtracted gravity, which made the demand meaningless.
    vertical = delta_v[:, 1] * FPS - G[1]
    horizontal = np.linalg.norm(delta_v[:, [0, 2]], axis=-1) * FPS
    support = contacts[1:].any(axis=-1) & contacts[:-1].any(axis=-1)
    # Frames that need almost no net support force make the ratio ill-conditioned.
    floor = floor_fraction * (-G[1])
    valid = support & (vertical > floor)
    demand = np.where(valid, horizontal / np.maximum(vertical, 1e-9), np.nan)
    diagnostics = dict(
        com_speed_p50=float(np.median(np.linalg.norm(velocity[1:-1], axis=-1))) if velocity.shape[0] > 2 else None,
        vertical_g_p10=float(np.percentile(vertical[support], 10) / (-G[1])) if support.any() else None,
        vertical_g_p50=float(np.percentile(vertical[support], 50) / (-G[1])) if support.any() else None,
        vertical_g_p90=float(np.percentile(vertical[support], 90) / (-G[1])) if support.any() else None,
        horizontal_p50=float(np.median(horizontal[support])) if support.any() else None)
    return demand[valid], int((support & ~valid).sum()), diagnostics


def angular_rate(coms, mass):
    """Orbital part of the angular momentum rate about the centre of mass.

    Units are m^2/s^2 (torque per unit mass). Contact forces acting at roughly half a
    metre from the centre of mass can supply about g * 0.5 ~ 5; a much larger value
    means the second difference dominates and the angular rows carry no signal.
    """
    total = mass.sum()
    centre = (coms * mass[:, None]).sum(1) / total
    velocity = np.zeros_like(coms)
    velocity[1:-1] = (coms[2:] - coms[:-2]) * FPS / 2.0
    centre_velocity = (velocity * mass[:, None]).sum(1) / total
    relative = coms - centre[:, None]
    momentum = (mass[None, :, None] * np.cross(relative, velocity - centre_velocity[:, None])).sum(1) / total
    rate = np.zeros_like(momentum)
    rate[1:-1] = (momentum[2:] - momentum[:-2]) * FPS / 2.0
    return np.linalg.norm(rate[1:-1], axis=-1), np.linalg.norm(momentum, axis=-1)


def matched(values, mask):
    n = min(len(values), len(mask))
    return values[:n], mask[:n]


def ballistic(com, contacts):
    runs = flight_runs(~contacts.any(axis=1), MIN_FLIGHT_RUN)
    residuals, spans = [], []
    for start, stop in runs:
        t = (np.arange(stop - start) / FPS)
        design = np.stack([np.ones_like(t), t, t * t], axis=1)
        acc = np.zeros(3)
        for axis in range(3):
            coef, *_ = np.linalg.lstsq(design, com[start:stop, axis], rcond=None)
            acc[axis] = 2.0 * coef[2]
        residuals.append(float(np.linalg.norm(acc - G)))
        spans.append(stop - start)
    return residuals, spans


def audit(row, rep, proxy, neutral, foot_joints, shift=0, floor_fraction=0.2):
    world = torch.from_numpy(np.array(row, copy=True))[None].to(rep.skeleton.joint_parents.device)
    pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
    joints = pose['posed_joints'][0]
    rots = pose['global_joint_rots'][0]
    contacts = (world[0][:, rep.indices['foot_contacts']] > 0.5).detach().cpu().numpy()
    frames = min(joints.shape[0], rots.shape[0], contacts.shape[0])
    joints, rots, contacts = joints[:frames], rots[:frames], contacts[:frames]
    # Deliberately misalign contact labels against the motion. A metric that cannot
    # detect this carries no information about contact-motion consistency.
    if shift:
        contacts = np.roll(contacts, shift, axis=0)
    coms, mass = segment_com_world(joints, rots, proxy, neutral)
    # Sum over segments (dim 1), not over time: an earlier version summed dim 0 and
    # produced a per-segment time average instead of a per-frame centre of mass.
    com = ((coms * mass[:, None]).sum(1) / mass.sum()).detach().cpu().numpy()
    residuals, spans = ballistic(com, contacts)
    feet = joints.detach().cpu().numpy()[:, foot_joints, :]
    velocity = np.zeros_like(feet)
    velocity[1:-1] = (feet[2:] - feet[:-2]) * FPS / 2.0
    demand, unsupported, friction_diag = friction_requirement(com, contacts, floor_fraction)
    torque_rate, angular_momentum = angular_rate(coms.detach().cpu().numpy(),
                                                 mass.detach().cpu().numpy())
    friction_diag['torque_rate_p50'] = float(np.median(torque_rate)) if torque_rate.size else None
    friction_diag['torque_rate_p90'] = float(np.percentile(torque_rate, 90)) if torque_rate.size else None
    friction_diag['angular_momentum_p50'] = float(np.median(angular_momentum)) if angular_momentum.size else None
    speeds = []
    for k in range(len(foot_joints)):
        values, active = matched(velocity[1:-1][:, k], contacts[1:-1, k])
        if active.any():
            speeds.append(np.linalg.norm(values[active], axis=-1))
    speed = np.concatenate(speeds) if speeds else np.zeros(0)
    return dict(frames=int(frames),
                flight_runs=len(residuals),
                flight_spans=spans,
                flight_residuals=residuals,
                friction_frames=int(demand.size),
                friction_diag=friction_diag,
                friction_unsupported=unsupported,
                friction_mu=dict(
                    p50=float(np.median(demand)) if demand.size else None,
                    p90=float(np.percentile(demand, 90)) if demand.size else None,
                    over_0_5=float((demand > 0.5).mean()) if demand.size else None,
                    over_1_0=float((demand > 1.0).mean()) if demand.size else None),
                pooled_speed=speed,
                stance_frames=int(speed.size),
                stance_speed=dict(
                    p10=float(np.percentile(speed, 10)) if speed.size else None,
                    median=float(np.median(speed)) if speed.size else None,
                    p90=float(np.percentile(speed, 90)) if speed.size else None,
                    p95=float(np.percentile(speed, 95)) if speed.size else None,
                    mean=float(speed.mean()) if speed.size else None,
                    over_5cmps=float((speed > 0.05).mean()) if speed.size else None,
                    over_10cmps=float((speed > 0.10).mean()) if speed.size else None))


def summarise(results):
    res = [v for r in results for v in r['flight_residuals']]
    slopes = [v for r in results for v in r['flight_spans']]
    allspeed = [r['stance_speed'] for r in results if r['stance_speed']['median'] is not None]
    out = dict(rows=len(results),
               usable_flight_runs=len(res),
               flight_frame_total=sum(r['frames'] for r in results),
               stance_frames=sum(r['stance_frames'] for r in results))
    if res:
        out.update(flight_residual_median_mps2=float(np.median(res)),
                   flight_residual_p90_mps2=float(np.percentile(res, 90)),
                   flight_run_median_frames=float(np.median(slopes)))
    if allspeed:
        for key in ('p10', 'median', 'p90', 'p95', 'mean', 'over_5cmps', 'over_10cmps'):
            out['stance_' + key] = float(np.median([s[key] for s in allspeed]))
    diags = [r['friction_diag'] for r in results if r['friction_diag']['com_speed_p50'] is not None]
    if diags:
        for key in ('com_speed_p50', 'vertical_g_p10', 'vertical_g_p50', 'vertical_g_p90', 'horizontal_p50',
                    'torque_rate_p50', 'torque_rate_p90', 'angular_momentum_p50'):
            out['diag_' + key] = float(np.median([d[key] for d in diags if d[key] is not None]))
    mus = [r['friction_mu'] for r in results if r['friction_mu']['p50'] is not None]
    if mus:
        for key in ('p50', 'p90', 'over_0_5', 'over_1_0'):
            out['friction_' + key] = float(np.median([m[key] for m in mus]))
        out['friction_frames'] = sum(r['friction_frames'] for r in results)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=pathlib.Path, required=True)
    parser.add_argument('--pfnn', type=pathlib.Path, required=True)
    parser.add_argument('--stats', type=pathlib.Path, required=True)
    parser.add_argument('--proxy', type=pathlib.Path, required=True)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--limit', type=int, default=48)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--shifts', default='0', help='comma-separated contact-label shifts')
    parser.add_argument('--mass-jitter', type=float, default=0.0,
                        help='log-normal relative perturbation of segment masses')
    parser.add_argument('--floor-fractions', default='0.2',
                        help='comma-separated vertical demand floors as fractions of g')
    parser.add_argument('--mass-shuffle', action='store_true',
                        help='permute segment masses across the body (extreme control)')
    args = parser.parse_args()
    rep = build_native_representation(args.stats)
    proxy = dict(np.load(args.proxy, allow_pickle=True))
    neutral = rep.skeleton.neutral_joints.detach().cpu().numpy()
    # Sanitise the proxy: a segment with negligible volume has com = 0/0 stored in the
    # archive, so its offset is set to zero and its mass is dropped instead of letting
    # a permutation control place that garbage origin on a heavy segment.
    mass = np.array(proxy['mass'], dtype=np.float64)
    com = np.array(proxy['com'], dtype=np.float64)
    neutral_np = neutral
    floor = 1e-4 * abs(mass).sum()
    # A negative volume is an integration artefact, and a stored centre of mass more
    # than a metre from its joint is the 0/0 of a near-zero-volume segment.
    displaced = np.linalg.norm(com - neutral, axis=-1) > 1.0
    negligible = (mass <= 0.0) | (np.abs(mass) < floor) | displaced
    com[negligible] = neutral_np[negligible]
    mass[negligible] = 0.0
    proxy['mass'] = mass
    proxy['com'] = com
    print('# proxy sanitised: %d negligible segments' % int(negligible.sum()))
    # Perturbation must follow sanitisation: a permuted heavy mass landing on a
    # segment whose archived com is 0/0 would otherwise produce a meaningless CoM.
    if args.mass_jitter or args.mass_shuffle:
        mass = np.array(proxy['mass'], dtype=np.float64).copy()
        if args.mass_shuffle:
            mass = mass[np.random.default_rng(7).permutation(len(mass))]
        else:
            relative = np.random.default_rng(7).normal(0.0, args.mass_jitter, len(mass))
            mass = mass * np.exp(relative)
        proxy['mass'] = mass
        print('# proxy perturbed: jitter=%s shuffle=%s' % (args.mass_jitter, args.mass_shuffle))

    foot_joints = foot_indices(proxy)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, args.split)
    results = [audit(dataset[i]['world'].numpy(), rep, proxy, neutral, foot_joints)
               for i in range(min(args.limit, len(dataset)))]
    out, pooled = {}, {}
    conditions = [(shift, float(f)) for f in args.floor_fractions.split(',')
                  for shift in [int(s) for s in args.shifts.split(',')]]
    for shift, floor_fraction in conditions:
        key_name = str(shift) if len(set(f for _, f in conditions)) == 1 else f'{shift}|{floor_fraction}'
        indices = list(range(args.skip, min(args.skip + args.limit, len(dataset))))
        rows = [audit(dataset[i]['world'].numpy(), rep, proxy, neutral, foot_joints, shift, floor_fraction)
                for i in indices]
        pooled[key_name] = np.concatenate([r.pop('pooled_speed') for r in rows]) if rows else np.zeros(0)
        out[key_name] = summarise(rows)
    print(json.dumps(out if len(out) > 1 else out['0'], indent=2))
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))
        np.savez(args.output.with_suffix('.npz'), **pooled)


if __name__ == '__main__':
    main()
