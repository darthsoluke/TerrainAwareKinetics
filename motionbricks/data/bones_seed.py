"""Validated planar SOMA data helpers; upstream conversion and MB feature math."""

import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats
from motionbricks.data.synthetic_dataset import collate_batch, collate_tensors


FPS = 60.0
SOMA_REVISION = "58e781898b3d7e328a676a75d3e338c45dce3ad9"
FOOT_CHANNELS = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]

#: Proxy geometry revisions. position_only_v0 subtracts the bind joint position but
#: never applies the inverse bind rotation, which leaves a 0.8-1.8 cm error because
#: the foot joints' bind rotations sit about ten degrees away from the world axes.
SOLE_PROXY_GEOMETRY_POSITION_ONLY = "position_only_v0"
SOLE_PROXY_GEOMETRY_LOCAL_FRAME = "local_frame_v1"
SOLE_PROXY_GEOMETRIES = (SOLE_PROXY_GEOMETRY_POSITION_ONLY, SOLE_PROXY_GEOMETRY_LOCAL_FRAME)
SOLE_PROXY_GEOMETRY = SOLE_PROXY_GEOMETRY_LOCAL_FRAME
#: The shipped BONES pilot datasets were converted with this revision. Switching
#: revisions changes sole heights, the inferred fixed ground plane and the contact
#: labels, so it requires a versioned rebuild rather than a silent overwrite.
SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET = SOLE_PROXY_GEOMETRY_POSITION_ONLY


def recording_group(filename):
    """Group actors, mirrors and takes of the same named source behavior."""
    name = re.sub(r"_M$", "", Path(filename).stem)
    name = re.sub(r"__A\d+$", "", name, flags=re.IGNORECASE)
    return re.sub(r"_\d+$", "", name).lower()


def split_for_group(group, seed=2333):
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) % 1000
    return "train" if bucket < 900 else "validation" if bucket < 950 else "test"


def resample_rotations_root(local, root, source_fps, target_fps=FPS):
    """Anti-alias, then sample rotations with SLERP on exact physical timestamps."""
    if source_fps <= 0 or target_fps <= 0 or len(local) < 2:
        raise ValueError("Positive frame rates and at least two frames are required")
    source_times = np.arange(len(local), dtype=np.float64) / source_fps
    target_times = np.arange(int(np.floor(source_times[-1] * target_fps + 1e-8)) + 1) / target_fps
    quaternions = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(len(local), -1, 4)
    # Remove antipodal sign changes before filtering quaternion components.
    signs = np.where((quaternions[1:] * quaternions[:-1]).sum(-1) < 0, -1.0, 1.0)
    quaternions[1:] *= np.cumprod(signs, axis=0)[..., None]
    filtered_root = np.array(root, dtype=np.float64, copy=True)
    if source_fps > target_fps * 1.05:
        if len(local) < 20:
            raise ValueError("Clip too short for downsampling anti-alias filter")
        sos = butter(4, 0.4 * target_fps, fs=source_fps, output="sos")
        filtered_root = sosfiltfilt(sos, filtered_root, axis=0)
        quaternions = sosfiltfilt(sos, quaternions, axis=0)
        norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        if norms.min() < 0.5:
            raise ValueError("Quaternion filter crossed a singular orientation")
        quaternions /= norms
    rotations = np.stack([
        Slerp(source_times, Rotation.from_quat(quaternions[:, joint]))(target_times).as_matrix()
        for joint in range(quaternions.shape[1])
    ], axis=1)
    root_out = np.stack([np.interp(target_times, source_times, filtered_root[:, axis]) for axis in range(3)], -1)
    return rotations.astype(np.float32), root_out.astype(np.float32)


def build_representation(stats_folder=None):
    stats = Stats(folder=str(stats_folder)) if stats_folder else None
    return GlobalRootGlobalJoints(FPS, SOMASkeleton30(),
                                 "somaskel30_dual_root_global_joints", stats=stats)


def official_sole_surfaces(skeleton):
    """Sole mesh vertex sets per foot channel, in the mesh bind frame.

    Returns one record per channel in FOOT_CHANNELS with the skinned foot vertices
    (assigned), the lowest slice of them (sole) and the heel/forefoot half actually
    used for the proxy (selected), plus the channel joint bind translation and bind
    rotation. Also returns the full bind rig transform array, the rig joint names and
    the mesh digest. Calibration tooling needs all of it; official_sole_offsets only
    needs the medians.
    """
    from kimodo.assets import skeleton_asset_path
    path = skeleton_asset_path("somaskel77") / "skin_standard.npz"
    with np.load(path, allow_pickle=False) as mesh:
        names = list(mesh["rig_joint_names"].astype(str))
        vertices = mesh["bind_vertices"]
        indices, weights = mesh["lbs_indices"], mesh["lbs_weights"]
        bind = mesh["bind_rig_transform"]
        surfaces = []
        for channel in FOOT_CHANNELS:
            side = "Left" if channel.startswith("Left") else "Right"
            family = [names.index(side + suffix) for suffix in ("Foot", "ToeBase", "ToeEnd")]
            influence = (weights * np.isin(indices, family)).sum(-1)
            assigned_indices = np.flatnonzero(influence > 0.8)
            assigned = vertices[assigned_indices]
            sole_indices = assigned_indices[assigned[:, 1] <= np.quantile(assigned[:, 1], 0.08)]
            sole = vertices[sole_indices]
            median_z = np.median(sole[:, 2])
            half = sole[:, 2] >= median_z if "Toe" in channel else sole[:, 2] < median_z
            selected_indices, selected = sole_indices[half], sole[half]
            if len(selected) < 3:
                raise ValueError(f"Insufficient sole mesh support for {channel}")
            index = names.index(channel)
            surfaces.append({"channel": channel, "joint": channel, "joint_index": index,
                             "assigned": assigned, "sole": sole, "selected": selected,
                             "assigned_indices": assigned_indices, "sole_indices": sole_indices,
                             "selected_indices": selected_indices,
                             "joint_bind_translation": np.array(bind[index, :3, 3], dtype=np.float64),
                             "joint_bind_rotation": np.array(bind[index, :3, :3], dtype=np.float64)})
        bind_transforms = np.array(bind, dtype=np.float64)
        return surfaces, bind_transforms, names, hashlib.sha256(path.read_bytes()).hexdigest()


def official_sole_offsets(skeleton, geometry=SOLE_PROXY_GEOMETRY):
    """Rigid sole proxies from the official skinned mesh, not ankle-joint heights.

    The offset is the proxy position expressed in its foot joint's local frame, so
    it must be rotated by the inverse bind rotation before a joint rotation is
    applied to it. position_only_v0 skips that inverse rotation and is kept only to
    reproduce data built before the defect was found.

    These points are calibration proxies, not exact contact points. The replay and
    contact review gate remains required before production training.
    """
    if geometry not in SOLE_PROXY_GEOMETRIES:
        raise ValueError(f"Unknown sole proxy geometry revision: {geometry}")
    surfaces, _, _, digest = official_sole_surfaces(skeleton)
    offsets = []
    for surface in surfaces:
        local = np.median(surface["selected"], axis=0) - surface["joint_bind_translation"]
        if geometry == SOLE_PROXY_GEOMETRY_LOCAL_FRAME:
            local = surface["joint_bind_rotation"].T @ local
        offsets.append(local)
    return torch.as_tensor(np.stack(offsets), dtype=torch.float32), digest


def sole_proxy_geometry_report(skeleton):
    """Per-channel difference between the two offset revisions, in metres."""
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    report = {}
    for surface in surfaces:
        point = np.median(surface["selected"], axis=0)
        local = point - surface["joint_bind_translation"]
        corrected = surface["joint_bind_rotation"].T @ local
        report[surface["channel"]] = {
            "position_only_v0_m": [round(float(value), 6) for value in local],
            "local_frame_v1_m": [round(float(value), 6) for value in corrected],
            "difference_m": round(float(np.linalg.norm(corrected - local)), 6),
            "bind_rotation_angle_deg": round(float(np.degrees(np.arccos(np.clip(
                (np.trace(surface["joint_bind_rotation"]) - 1) / 2, -1.0, 1.0)))), 4),
        }
    return report


def sole_positions(positions, rotations, skeleton, offsets):
    feet = skeleton.foot_joint_idx
    return positions[..., feet, :] + (rotations[..., feet, :, :] @ offsets[..., None]).squeeze(-1)


def calibration_crop(positions, skeleton, fps=FPS):
    """Remove sustained stationary T-pose calibration only near clip boundaries."""
    vectors = []
    for side in ("Left", "Right"):
        arm = positions[:, skeleton.bone_index[side + "Arm"]]
        hand = positions[:, skeleton.bone_index[side + "Hand"]]
        vectors.append(hand - arm)
    left, right = vectors
    lengths = torch.stack([left.norm(dim=-1), right.norm(dim=-1)], -1)
    horizontal = (torch.stack([left[:, 1].abs(), right[:, 1].abs()], -1) / lengths.clamp(min=1e-6) < 0.12).all(-1)
    opposed = (left * right).sum(-1) / lengths.prod(-1).clamp(min=1e-6) < -0.75
    speed = torch.gradient(positions[:, 0], spacing=(1 / fps,), dim=(0,))[0].norm(dim=-1)
    raw = (horizontal & opposed & (lengths.min(-1).values > 0.45) & (speed < 0.15)).numpy()
    changes = np.diff(np.concatenate([[False], raw, [False]]).astype(int))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    start, end = 0, len(positions)
    margin = int(round(0.2 * fps))
    for first, last in zip(starts, ends):
        if last - first < int(round(0.3 * fps)):
            continue
        if first < 3 * fps:
            start = max(start, int(last) + margin)
        elif last > len(positions) - 3 * fps:
            end = min(end, int(first) - margin)
    if end - start < 129:
        raise ValueError("Insufficient motion after removing boundary calibration")
    return start, end


def estimate_contacts(sole, fps=FPS):
    """One fixed support plane per planar clip and conservative four-channel labels."""
    velocity = torch.gradient(sole, spacing=(1.0 / fps,), dim=(0,))[0].norm(dim=-1)
    height = sole[..., 1]
    stationary = velocity < 0.25
    low = height <= torch.quantile(height.flatten(), 0.25)
    support = height[stationary & low]
    if support.numel() < max(20, int(0.1 * fps)):
        raise ValueError("No reliable stationary support points")
    plane = support.median()
    spread = torch.quantile((support - plane).abs(), 0.9)
    if spread > 0.035:
        raise ValueError(f"Uncertain planar support ({spread.item():.3f} m)")
    clearance = height - plane
    raw = (velocity < 0.20) & (clearance.abs() < 0.035)
    # Require 50 ms persistence. Do not turn short noisy transitions into support.
    width = max(3, int(round(0.05 * fps)))
    if width % 2 == 0:
        width += 1
    stable = torch.nn.functional.avg_pool1d(
        raw.T.float().unsqueeze(0), width, stride=1, padding=width // 2)[0].T > 0.99
    known_swing = (clearance > 0.06) | (velocity > 0.35)
    confidence = stable | known_swing
    penetration_fraction = (clearance < -0.04).float().mean().item()
    if penetration_fraction > 0.05:
        raise ValueError("Source soles substantially below inferred fixed plane")
    return stable.float(), confidence.float(), plane.item(), {
        "support_spread_m": spread.item(), "source_penetration_fraction": penetration_fraction,
        "contact_fraction": stable.float().mean(0).tolist(),
        "contact_confidence_fraction": confidence.float().mean().item(),
    }


def validate_planar_posture(positions, skeleton, plane, offsets):
    """Reject inverted bodies and misidentified support planes for upright V1 locomotion."""
    root_height = positions[:, skeleton.root_idx, 1] - plane
    head_height = positions[:, skeleton.bone_index["Head"], 1] - positions[:, skeleton.root_idx, 1]
    reference_sole = skeleton.neutral_joints[skeleton.foot_joint_idx] + offsets
    reference_height = -reference_sole[:, 1].median()
    ratio = root_height / reference_height
    median, lower, upper = torch.quantile(ratio, positions.new_tensor([0.5, 0.05, 0.95]))
    if median < 0.65 or median > 1.3 or lower < 0.5 or upper > 1.5:
        raise ValueError("Upright hip height inconsistent with the inferred fixed support plane")
    if torch.quantile(head_height, 0.05) < 0.15:
        raise ValueError("Inverted or deeply folded body is not upright planar locomotion")
    return {"hip_height_median_m": float(root_height.median()),
            "hip_height_reference_m": float(reference_height),
            "head_above_hips_q05_m": float(torch.quantile(head_height, 0.05))}


def deterministic_window_starts(total_frames, frames, windows_per_clip=3):
    if frames < 1 or total_frames < frames + 1 or windows_per_clip < 1:
        raise ValueError("Evaluation needs positive windows and one additional root-velocity frame")
    return np.unique(np.linspace(0, total_frames - frames - 1, windows_per_clip).astype(int)).tolist()


def validation_window_records(records, frames, windows_per_clip=3):
    return [dict(record, window_start_frame=start)
            for record in records
            for start in deterministic_window_starts(record["frames_60hz"], frames, windows_per_clip)]


SOLE_LABEL_KEYS = tuple(
    f"sole_{kind}_{side}"
    for kind in ("frame_state", "interval_state", "region_measured", "vertex_measured")
    for side in ("left", "right"))


class BonesMotionDataset(Dataset):
    def __init__(self, folder, records, stats, crop_frames=360, training=True, sole_region_folder=None):
        self.folder, self.records = Path(folder), records
        # Lightning moves the model's Stats buffers to CUDA. Workers must retain
        # their own CPU copy rather than inherit CUDA tensors through this alias.
        self.stats = Stats(eps=stats.eps, legacy=stats.legacy)
        self.stats.register_from_tensors(stats.mean.detach().cpu().clone(), stats.std.detach().cpu().clone())
        self.crop_frames, self.training = crop_frames, training
        self.sole_region_folder = Path(sole_region_folder) if sole_region_folder else None

    def __len__(self):
        return len(self.records)

    def _sole_region_labels(self, record, start, count):
        """Frame, interval and per-vertex masks for the window this crop already chose.

        One sampling index decides the window: this method never draws its own start.
        The pipeline turns count global frames into count - 1 local frames, so the local
        frame t is global frame start + t, and the local interval t is global interval
        start + t. Nothing is padded and no frame is duplicated.
        """
        path = self.sole_region_folder / f"{record['filename']}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing sole-region labels for {record['filename']}: {path}")
        frames, intervals = count - 1, count - 2
        labels = {}
        with np.load(path, allow_pickle=False) as data:
            for side in ("left", "right"):
                for key, length in (("frame_state", frames), ("interval_state", intervals),
                                    ("region_measured", intervals), ("vertex_measured", intervals)):
                    source = data[f"{key}_{side}"]
                    if source.shape[0] < start + length:
                        raise ValueError(f"{record['filename']}: {key}_{side} is shorter than the sampled window")
                    values = source[start:start + length]
                    labels[f"sole_{key}_{side}"] = torch.from_numpy(
                        values.astype(np.int64) if key.endswith("state") else values)
        return labels

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(self.folder / record["shard"], allow_pickle=False) as shard:
            motion = torch.from_numpy(shard["global_features"].copy())
            confidence = torch.from_numpy(shard["contact_confidence"].copy())
        if confidence.shape != (len(motion), 4) or not ((confidence == 0) | (confidence == 1)).all():
            raise ValueError("Four binary saved contact-confidence channels are required")
        count = min(len(motion), self.crop_frames)
        start = (int(torch.randint(len(motion) - count + 1, ())) if self.training
                 else int(record.get("window_start_frame", 0)))
        if start < 0 or start + count > len(motion):
            raise ValueError("Frozen evaluation window exceeds the saved shard")
        item = {"keyid": index, "motion": self.stats.normalize(motion[start:start + count]),
                "contact_confidence": confidence[start:start + count].bool()}
        if self.sole_region_folder is not None:
            item.update(self._sole_region_labels(record, start, count))
        return item


def collate_bones_batch(batch):
    result = collate_batch(batch)
    confidence, lengths, _ = collate_tensors([item["contact_confidence"].float() for item in batch])
    if not torch.equal(lengths, result["motion_len"]):
        raise ValueError("Contact confidence and motion must have identical frame alignment")
    result["contact_confidence"] = confidence.bool()
    present = [key for key in SOLE_LABEL_KEYS if key in batch[0]]
    if present:
        from motionbricks.data.sole_region_contact import FRAME_STATES, INTERVAL_STATES
        for item in batch:
            if [key for key in SOLE_LABEL_KEYS if key in item] != present:
                raise ValueError("Sole-region labels must be present for every item in the batch")
        for key in present:
            is_frame = key.startswith("sole_frame_state_")
            delta = 1 if is_frame else 2
            for item in batch:
                if item[key].shape[0] != len(item["motion"]) - delta:
                    raise ValueError(f"{key} must align with its item's global motion window")
            padding = (FRAME_STATES.index("undecided") if is_frame else
                       INTERVAL_STATES.index("undecided") if "interval_state" in key else False)
            sample = batch[0][key]
            padded = sample.new_full((len(batch), result["motion"].shape[1] - delta, *sample.shape[1:]), padding)
            for row, item in enumerate(batch):
                padded[row, :len(item[key])] = item[key]
            result[key] = padded
    return result
