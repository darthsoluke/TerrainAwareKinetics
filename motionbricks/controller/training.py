"""Training and evaluation steps for TerrainMotionTransformer on the BONES motion-scene set.

Two kinds of update share one model:

* teacher step: real 16-frame history, predict the next source frame (the established
  pfnn_frame_actor.source_one_step_loss) plus the proposed path against the source's
  own future root path;
* closed-loop step: closed_loop.ClosedLoop.rollout, H frames on the model's own outputs,
  task + scene + masked imitation costs, critic bootstrap.

evaluate() reports the diagnostics used by the pilot gates: one-step body error against
the persistence baseline, and closed-loop drift, penetration, skating and pelvis height
against the source over the rollout horizon.
"""

import torch
import torch.nn.functional as F

from motionbricks.data.bones_env_dataset import boxes_of
from motionbricks.data.pfnn_frame_actor import source_one_step_loss

from .closed_loop import forward_xz, posed_joints
from .scene_sdf import box_sdf

#: Source frames (within the 16 target frames) for the path stations at 0.1 s steps.
PATH_FRAMES = tuple(3 * k - 1 for k in range(1, 9))


def path_loss(plan, history, targets):
    """Proposed path vs the source's own future root path, heading-free (world offsets)."""
    known = [i for i, f in enumerate(PATH_FRAMES) if f < targets.shape[1]]
    frames = [PATH_FRAMES[i] for i in known]
    proposed = plan["path_world_xz"][:, known] - history[:, -1, [0, 2]][:, None]
    truth = (targets[:, frames][..., [0, 2]] - history[:, -1, [0, 2]][:, None]).detach()
    return F.smooth_l1_loss(proposed / .3, truth / .3)


def planted_foot_loss(loop, pred, history, target):
    """Sole displacement of the emitted frame vs the source's, on the source's contact channels.

    source_one_step_loss compares joints relative to the root and the root displacement
    separately, so their small errors can combine into a sliding stance foot. This term
    compares world-frame sole displacement directly where the source foot is planted.
    """
    from motionbricks.data.pfnn_frame_actor import advance
    emitted = advance(loop.rep, history[:, -4:], pred)[:, -1:]
    joints = posed_joints(loop.rep, emitted)[:, 0]
    last = posed_joints(loop.rep, history[:, -1:])[:, 0]
    with torch.no_grad():
        truth = posed_joints(loop.rep, target[:, None])[:, 0]
        labels = target.index_select(-1, loop._contact_index(target.device))
    moved = (loop.soles(joints) - loop.soles(last))[..., [0, 2]]
    source_moved = (loop.soles(truth) - loop.soles(last))[..., [0, 2]]
    error = F.smooth_l1_loss(moved / .01, source_moved / .01, reduction="none").sum(-1)
    return (labels * error).sum() / labels.sum().clamp_min(1.0)


def teacher_step(loop, batch, path_weight=0.25, foot_weight=0.0):
    """One-step supervised losses on real history; returns (total, logged terms)."""
    history, mask, boxes = batch["history"], batch["history_mask"], boxes_of(batch)
    command = batch["commands"][:, 0]
    _, part_rays, _ = loop.observe(history, command, boxes)
    plan = loop.model.plan(history, mask, command, part_rays)
    stations = loop.prober(plan["path_world_xz"], history[:, -1, :3], forward_xz(history), boxes)
    pred = loop.model.act(history, plan, stations)
    losses = source_one_step_loss(loop.rep, pred, batch["targets"][:, 0], history[:, -4:])
    losses["path"] = path_loss(plan, history, batch["targets"])
    total = losses["total"] + path_weight * losses["path"]
    if foot_weight > 0:
        losses["planted_foot"] = planted_foot_loss(loop, pred, history, batch["targets"][:, 0])
        total = total + foot_weight * losses["planted_foot"]
    return total, {k: float(v.detach()) for k, v in losses.items()}


def closed_loop_step(loop, batch, critic, horizon):
    result = loop.rollout(batch["history"], batch["history_mask"], batch["commands"][:, :horizon], boxes_of(batch),
                          critic=critic, targets=batch["targets"][:, :horizon])
    terms = {k: float(v) for k, v in result["terms"].items()}
    terms.update(policy=float(result["policy_loss"].detach()), critic=float(result["critic_loss"].detach()))
    return result["policy_loss"], result["critic_loss"], terms


def _penetration_frames(loop, joints, boxes):
    """[B, H] deepest joint-sphere penetration per frame (joints [B, H, J, 3])."""
    batch, horizon = joints.shape[:2]
    depth = (-box_sdf(joints.flatten(1, 2), boxes) - loop.radii.to(joints).repeat(horizon)).clamp_min(0)
    return depth.reshape(batch, horizon, -1).amax(-1)


def _planted_sole_speed(loop, joints, previous, contacts, boxes=None, height=0.03):
    """[B, H, 4] horizontal sole speed (m/s), the SOURCE contact mask, and (with boxes) the
    geometric mask: the sole is within `height` of a surface in this and the previous frame."""
    soles = loop.soles(joints.flatten(0, 1)).reshape(*joints.shape[:2], 4, 3)
    before = torch.cat((loop.soles(previous)[:, None], soles[:, :-1]), 1)
    speed = (soles - before)[..., [0, 2]].norm(dim=-1) * 30
    near = None
    if boxes is not None:
        batch, horizon = soles.shape[:2]
        now = box_sdf(soles.reshape(batch, -1, 3), boxes).reshape(batch, horizon, 4) <= height
        then = box_sdf(before.reshape(batch, -1, 3), boxes).reshape(batch, horizon, 4) <= height
        near = now & then
    return speed, contacts > 0.5, near


#: A window counts as locomotion when the source root moves at least this far (m) over the horizon.
MOVING_M = 0.10


def _median(values):
    return float(values.median()) if values.numel() else float("nan")


@torch.no_grad()
def motion_metrics(loop, batch, generated, horizon):
    """Closed-loop diagnostics of generated frames [B, H, 366] against the source window.

    Penetration and planted-foot speed are reported for the generated motion AND for the
    source motion on the same windows, so gates compare against what the data does (an
    untrained model that holds still trivially neither penetrates nor slides). The
    progress ratio (generated / source root displacement on windows whose source moves at
    least MOVING_M) catches a model that satisfies the foot costs by not walking.
    """
    rep, history, targets = loop.rep, batch["history"], batch["targets"]
    boxes = boxes_of(batch)
    root = rep.skeleton.root_idx
    last2 = posed_joints(rep, history[:, -2:])
    last = last2[:, -1]
    generated_joints = posed_joints(rep, generated)
    source_joints = posed_joints(rep, targets[:, :horizon])
    origin = history[:, -1, [0, 2]]
    drift = (generated[:, -1, [0, 2]] - targets[:, horizon - 1, [0, 2]]).norm(dim=-1)
    source_move = (targets[:, horizon - 1, [0, 2]] - origin).norm(dim=-1)
    generated_move = (generated[:, -1, [0, 2]] - origin).norm(dim=-1)
    moving = source_move >= MOVING_M
    progress = generated_move / source_move.clamp_min(1e-6)
    depth = _penetration_frames(loop, generated_joints, boxes)
    source_depth = _penetration_frames(loop, source_joints, boxes)
    pelvis = (generated_joints[:, :, root, 1] - source_joints[:, :, root, 1]).abs().mean(-1)
    contacts = batch["contacts"][:, :horizon]
    speed, planted, near = _planted_sole_speed(loop, generated_joints, last, contacts, boxes)
    source_speed, _, source_near = _planted_sole_speed(loop, source_joints, last, contacts, boxes)

    def acceleration(joints):
        sequence = torch.cat((last2, joints), 1)
        return ((sequence[:, 2:] - 2 * sequence[:, 1:-1] + sequence[:, :-2]) * 900).norm(dim=-1).mean(-1)   # [B, H]
    accel, source_accel = acceleration(generated_joints), acceleration(source_joints)

    def masked_median(values, mask):
        return _median(values[mask])
    report = {
        "closed_loop_root_drift_cm_p50": float(drift.median() * 100),
        "closed_loop_root_drift_cm_p90": float(drift.quantile(0.9) * 100),
        "closed_loop_joint_penetration_over_2cm_frames": float((depth > 0.02).float().mean()),
        "source_joint_penetration_over_2cm_frames": float((source_depth > 0.02).float().mean()),
        "closed_loop_pelvis_height_error_cm": float(pelvis.mean() * 100),
        "closed_loop_planted_sole_speed_mps_p50": masked_median(speed, planted),
        "source_planted_sole_speed_mps_p50": masked_median(source_speed, planted),
        # Each motion judged on its own feet: speed of soles resting on a surface.
        "closed_loop_grounded_sole_speed_mps_p50": masked_median(speed, near),
        "source_grounded_sole_speed_mps_p50": masked_median(source_speed, source_near),
        "closed_loop_grounded_sole_fraction": float(near.float().mean()),
        "source_grounded_sole_fraction": float(source_near.float().mean()),
        "moving_windows": int(moving.sum()),
        "closed_loop_progress_ratio_p50": masked_median(progress, moving),
        "closed_loop_progress_ratio_p10": float(progress[moving].quantile(0.1)) if moving.any() else float("nan"),
        "closed_loop_joint_accel_mps2_mean": float(accel.mean()),
        "source_joint_accel_mps2_mean": float(source_accel.mean()),
    }
    groups = batch.get("group")
    if groups is not None:
        by_group = {}
        for name in sorted(set(groups)):
            rows = torch.tensor([g == name for g in groups], device=drift.device)
            frames = rows[:, None, None].expand_as(speed)
            by_group[name] = {
                "windows": int(rows.sum()),
                "drift_cm_p50": float(drift[rows].median() * 100),
                "penetration_frames": float((depth[rows] > 0.02).float().mean()),
                "source_penetration_frames": float((source_depth[rows] > 0.02).float().mean()),
                "pelvis_cm": float(pelvis[rows].mean() * 100),
                "moving_windows": int((rows & moving).sum()),
                "progress_ratio_p50": masked_median(progress, rows & moving),
                "planted_sole_speed_mps_p50": masked_median(speed, planted & frames),
                "source_planted_sole_speed_mps_p50": masked_median(source_speed, planted & frames),
                "grounded_sole_speed_mps_p50": masked_median(speed, near & frames),
                "source_grounded_sole_speed_mps_p50": masked_median(source_speed, source_near & frames),
                "grounded_sole_fraction": float(near[rows].float().mean()),
                "joint_accel_mps2_mean": float(accel[rows].mean()),
            }
        report["by_group"] = by_group
    return report


@torch.no_grad()
def evaluate(loop, batch, horizon=16):
    """Diagnostics on one validation batch (model in eval mode by the caller): the
    one-step body error against the persistence baseline, then motion_metrics on a
    closed-loop rollout of ``horizon`` frames."""
    rep, history, targets = loop.rep, batch["history"], batch["targets"]
    boxes = boxes_of(batch)
    command = batch["commands"][:, 0]
    _, part_rays, _ = loop.observe(history, command, boxes)
    plan = loop.model.plan(history, batch["history_mask"], command, part_rays)
    stations = loop.prober(plan["path_world_xz"], history[:, -1, :3], forward_xz(history), boxes)
    pred = loop.model.act(history, plan, stations)
    from motionbricks.data.pfnn_frame_actor import advance
    emitted = advance(rep, history[:, -4:], pred)[:, -1:]
    joints = posed_joints(rep, emitted)[:, 0]
    truth = posed_joints(rep, targets[:, :1])[:, 0]
    last = posed_joints(rep, history[:, -1:])[:, 0]
    root = rep.skeleton.root_idx

    def relative(j):
        return j - j[:, root:root + 1]
    one_step = (relative(joints) - relative(truth)).norm(dim=-1).mean(-1)
    persistence = (relative(last) - relative(truth)).norm(dim=-1).mean(-1)
    result = loop.rollout(history, batch["history_mask"], batch["commands"][:, :horizon], boxes)
    report = {"one_step_body_cm": float(one_step.mean() * 100), "persistence_body_cm": float(persistence.mean() * 100)}
    report.update(motion_metrics(loop, batch, result["frames"], horizon))
    report["closed_loop_terms"] = {k: float(v) for k, v in result["terms"].items()}
    return report
