"""Physical-unit reconstruction diagnostics, separate from deployment acceptance."""

import math

import torch
import torch.nn.functional as F

from motionbricks.data.bones_seed import sole_positions
from motionbricks.motionlib.core.utils.rotations import quaternion_to_matrix


def rotation_error_degrees(prediction, target):
    relative = prediction @ target.transpose(-1, -2)
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), -1)
    sine = skew.norm(dim=-1) * 0.5
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sine, cosine).abs() * (180 / math.pi)


def reconstruction_metrics(prediction, target, representation, sole_offsets, confidence=None, skip_frames=0):
    from motionbricks.data.bones_models import stance_anchor_distances
    if skip_frames < 0 or prediction.shape[1] - skip_frames < 3:
        raise ValueError("At least three unconditioned frames are required for reconstruction metrics")
    skeleton = representation.skeleton
    pred = representation.inverse(prediction, True, return_all=True)
    gt = representation.inverse(target, True, return_all=True)
    joints, target_joints = pred["posed_joints"][:, skip_frames:], gt["posed_joints"][:, skip_frames:]
    root = skeleton.root_idx
    relative = joints - joints[..., root:root + 1, :]
    target_relative = target_joints - target_joints[..., root:root + 1, :]
    body = [index for index in range(skeleton.nbjoints) if index != root]
    soles = sole_positions(joints, pred["global_joint_rots"][:, skip_frames:], skeleton, sole_offsets)
    target_soles = sole_positions(target_joints, gt["global_joint_rots"][:, skip_frames:], skeleton, sole_offsets)
    contacts, target_contacts = pred["foot_contacts"][:, skip_frames:] >= 0.5, gt["foot_contacts"][:, skip_frames:] >= 0.5
    if confidence is None:
        confidence = torch.ones_like(target_contacts, dtype=torch.bool)
    else:
        confidence = confidence[:, skip_frames:]
    if confidence.shape != target_contacts.shape:
        raise ValueError("Confidence mask does not match the evaluated contact frames")
    confidence = confidence.bool()
    support = target_contacts & confidence
    anchors = stance_anchor_distances(soles, support)
    target_anchors = stance_anchor_distances(target_soles, support)
    stance = target_contacts[:, 1:] & target_contacts[:, :-1] & confidence[:, 1:] & confidence[:, :-1]
    speed = (soles[:, 1:] - soles[:, :-1]).norm(dim=-1) * representation.fps
    target_speed = (target_soles[:, 1:] - target_soles[:, :-1]).norm(dim=-1) * representation.fps
    root_error = (joints[..., root, [0, 2]] - target_joints[..., root, [0, 2]]).norm(dim=-1)
    penetration = F.relu(-soles[..., 1] - 0.005)
    pred_heading = representation.compute_root_pos_and_rot(representation.unnormalize(prediction))[1]
    gt_heading = representation.compute_root_pos_and_rot(representation.unnormalize(target))[1]
    heading_error = rotation_error_degrees(quaternion_to_matrix(pred_heading[:, skip_frames:]),
                                           quaternion_to_matrix(gt_heading[:, skip_frames:]))
    joint_error = (joints - target_joints).norm(dim=-1)
    acceleration = (joints[:, 2:] - 2 * joints[:, 1:-1] + joints[:, :-2]) * representation.fps ** 2
    tp = int((contacts & target_contacts & confidence).sum())
    fp = int((contacts & ~target_contacts & confidence).sum())
    fn = int((~contacts & target_contacts & confidence).sum())
    metrics = {
        "mpjpe_m": float(joint_error.mean()),
        "max_joint_error_mean_m": float(joint_error.max(-1).values.mean()),
        "max_joint_error_peak_m": float(joint_error.max()),
        "joint_jitter_mps2": float(acceleration.norm(dim=-1).mean()),
        "root_jitter_mps2": float(acceleration[..., root, :].norm(dim=-1).mean()),
        "body_root_relative_mpjpe_m": float((relative[..., body, :] - target_relative[..., body, :]).norm(dim=-1).mean()),
        "root_xz_error_m": float(root_error.mean()),
        "root_endpoint_error_m": float(root_error[:, -1].mean()),
        "root_height_error_m": float((joints[..., root, 1] - target_joints[..., root, 1]).abs().mean()),
        "root_heading_error_deg": float(heading_error.mean()),
        "local_rotation_error_deg": float(rotation_error_degrees(pred["local_joint_rots"][:, skip_frames:],
                                                                 gt["local_joint_rots"][:, skip_frames:]).mean()),
        "fk_position_feature_disagreement_m": float((joints - pred["all_posed_joints"]["ric_data"][:, skip_frames:]).norm(dim=-1).mean()),
        "sole_penetration_m": float(penetration.mean()),
        "sole_penetration_max_m": float(F.relu(-soles[..., 1]).max()),
        "sole_penetration_over_2cm_fraction": float((soles[..., 1] < -0.02).float().mean()),
        "trusted_support_samples": int(support.sum()),
        "trusted_support_clearance_abs_m": float(soles[..., 1][support].abs().mean()) if support.any() else None,
        "trusted_support_clearance_abs_p95_m": float(torch.quantile(soles[..., 1][support].abs(), 0.95)) if support.any() else None,
        "trusted_stance_samples": int(stance.sum()),
        "trusted_skate_mps": float(speed[stance].mean()) if stance.any() else None,
        "trusted_skate_p95_mps": float(torch.quantile(speed[stance], 0.95)) if stance.any() else None,
        "trusted_skate_max_mps": float(speed[stance].max()) if stance.any() else None,
        "trusted_skate_m_per_frame": float(speed[stance].mean()) / representation.fps if stance.any() else None,
        "source_trusted_skate_mps": float(target_speed[stance].mean()) if stance.any() else None,
        "trusted_stance_anchor_max_m": float(anchors[support].max()) if support.any() else None,
        "source_stance_anchor_max_m": float(target_anchors[support].max()) if support.any() else None,
        "contact_tp_auto": tp, "contact_fp_auto": fp, "contact_fn_auto": fn,
        "contact_precision_auto": tp / (tp + fp) if tp + fp else None,
        "contact_recall_auto": tp / (tp + fn) if tp + fn else None,
        "contact_f1_auto": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "contact_accuracy_auto": float((contacts == target_contacts)[confidence].float().mean()) if confidence.any() else None,
    }
    # Oracle root substitution diagnoses pose quality; it is not a generated-control score.
    controlled = prediction.clone()
    controlled[..., representation.indices["root"]] = target[..., representation.indices["root"]]
    controlled_joints = representation.inverse(controlled, True)["posed_joints"]
    metrics["oracle_root_mpjpe_m"] = float((controlled_joints[:, skip_frames:] - target_joints).norm(dim=-1).mean())
    return metrics, pred, gt


def summarize_windows(windows):
    summary = {}
    keys = windows[0]["metrics"] if windows else {}
    counts = {key: sum(window["metrics"][key] for window in windows)
              for key in ("contact_tp_auto", "contact_fp_auto", "contact_fn_auto", "trusted_stance_samples",
                          "trusted_support_samples")}
    for key in keys:
        values = [window["metrics"][key] for window in windows if window["metrics"][key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    summary.update(counts)
    for key in ("sole_penetration_max_m", "trusted_support_clearance_abs_p95_m", "max_joint_error_peak_m",
                "trusted_skate_p95_mps", "trusted_skate_max_mps", "trusted_stance_anchor_max_m", "source_stance_anchor_max_m"):
        if key not in keys:
            continue
        values = [window["metrics"][key] for window in windows if window["metrics"][key] is not None]
        summary[key] = max(values) if values else None
    summary["trusted_support_clearance_abs_m"] = (
        sum(window["metrics"]["trusted_support_clearance_abs_m"] * window["metrics"]["trusted_support_samples"]
            for window in windows if window["metrics"]["trusted_support_clearance_abs_m"] is not None)
        / counts["trusted_support_samples"] if counts["trusted_support_samples"] else None)
    for key in ("trusted_skate_mps", "trusted_skate_m_per_frame", "source_trusted_skate_mps"):
        summary[key] = (sum(window["metrics"][key] * window["metrics"]["trusted_stance_samples"]
                            for window in windows if window["metrics"][key] is not None)
                        / counts["trusted_stance_samples"] if counts["trusted_stance_samples"] else None)
    tp, fp, fn = (counts[f"contact_{key}_auto"] for key in ("tp", "fp", "fn"))
    summary["contact_precision_auto"] = tp / (tp + fp) if tp + fp else None
    summary["contact_recall_auto"] = tp / (tp + fn) if tp + fn else None
    summary["contact_f1_auto"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return summary
