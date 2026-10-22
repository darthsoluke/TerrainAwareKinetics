"""Integration checks on the real training path for sole-region supervision.

These drive the actual BonesTokenizer.sole_region_losses code and the actual dataset
crop, rather than re-testing the loss maths, and they check the four things that can
silently go wrong when supervision is wired into a trainer: the term reaching the
total loss with its configured weight, gradients arriving through the skinning while
labels stay detached, the window labels lining up with the window that was cropped,
and the failure modes staying loud.
"""

import numpy as np
import pytest
import torch

from motionbricks.data.bones_models import BonesTokenizer
from motionbricks.data.bones_seed import (BonesMotionDataset, SOLE_LABEL_KEYS, build_representation,
                                          collate_bones_batch, official_sole_surfaces)
from motionbricks.helper.data_training_util import (align_labels_to_sampled_segment, sole_frame_label_keys,
                                                    sole_interval_label_keys)
from motionbricks.data.sole_geometry_losses import sole_geometry_supervision
from motionbricks.data.sole_region_contact import (FRAME_STATES, INTERVAL_STATES, SOLE_LABEL_THRESHOLDS,
                                                   sole_region_indices)
from motionbricks.data.soma_skin_reference import load_official_skinner

FRAMES = 128
CONTACT = FRAME_STATES.index("contact")
STABLE = INTERVAL_STATES.index("stable_contact")


@pytest.fixture(scope="module")
def rig():
    skeleton = build_representation().skeleton
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    return {"skeleton": skeleton, "skinner": load_official_skinner(skeleton),
            "indices": {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}}


class _Harness:
    """Drives the real BonesTokenizer.sole_region_losses without building the network.

    Lightning turns self.args into a read-only property, so the method is borrowed onto
    a plain object that carries only the attributes the supervision path touches. The
    method body itself is the production one.
    """

    sole_region_losses = BonesTokenizer.sole_region_losses
    _sole_region_skinner_on = BonesTokenizer._sole_region_skinner_on


def make_tokenizer(rig, **args):
    model = _Harness()
    model.args = args
    model.motion_rep = type("Rep", (), {"fps": 60.0, "skeleton": rig["skeleton"]})()
    model.sole_region_skinner = rig["skinner"]
    model._sole_region_device = None
    model._sole_region_dtype = None
    for side in ("left", "right"):
        setattr(model, f"sole_region_indices_{side}", torch.as_tensor(rig["indices"][side.capitalize()]))
    return model


def make_prediction(rig, frames=FRAMES, batch=1):
    local = torch.eye(3, dtype=torch.float32).expand(batch, frames, 30, 3, 3).contiguous()
    root = torch.zeros(batch, frames, 3, dtype=torch.float32)
    root[..., 1] = 0.9
    _, posed, _ = rig["skeleton"].fk(local[0], root[0])
    return {"local_joint_rots": local.requires_grad_(True), "posed_joints": posed[None]}


def make_labels(frames=FRAMES, measured=True, vertices=98, batch=1):
    labels = {}
    for side in ("left", "right"):
        labels[f"sole_frame_state_{side}"] = torch.full((batch, frames), CONTACT, dtype=torch.long)
        labels[f"sole_interval_state_{side}"] = torch.full((batch, frames - 1), STABLE, dtype=torch.long)
        labels[f"sole_region_measured_{side}"] = torch.full((batch, frames - 1), measured, dtype=torch.bool)
        labels[f"sole_vertex_measured_{side}"] = torch.full((batch, frames - 1, vertices), measured,
                                                            dtype=torch.bool)
    return labels


def test_supervision_returns_weighted_terms_and_gradients_through_the_skinning(rig):
    model = make_tokenizer(rig, sole_region_coeff_contact_height_m=2.0, sole_region_coeff_penetration_m=3.0,
                           sole_region_coeff_slip_mps=4.0, sole_region_penetration_tolerance_m=0.005)
    pred = make_prediction(rig)
    totals, counts = model.sole_region_losses(make_labels(), pred)
    assert set(totals) == {"l_contact_height_m", "l_penetration_m", "l_measured_slip_mps"}
    assert counts["contact_frames_supervised"] == 2 * FRAMES * 1
    assert counts["measured_stable_intervals_supervised"] == 2 * (FRAMES - 1)
    assert counts["measured_vertices_supervised"] == 2 * (FRAMES - 1) * 98
    for value in totals.values():
        assert torch.isfinite(value)
    total = (2.0 * totals["l_contact_height_m"] + 3.0 * totals["l_penetration_m"]
             + 4.0 * totals["l_measured_slip_mps"])
    total.backward()
    grad = pred["local_joint_rots"].grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0, "the skinning must carry gradient back to the motion"


def test_a_shorter_sampled_window_is_supervised_on_its_own_frames(rig):
    model = make_tokenizer(rig)
    short = FRAMES // 2
    pred = make_prediction(rig, frames=short)
    labels = make_labels(frames=short)
    totals, counts = model.sole_region_losses(labels, pred)
    assert counts["contact_frames_supervised"] == 2 * short * 1
    assert counts["measured_stable_intervals_supervised"] == 2 * (short - 1)
    assert counts["measured_vertices_supervised"] == 2 * (short - 1) * 98
    for value in totals.values():
        assert torch.isfinite(value)


def test_labels_that_do_not_match_the_sampled_window_are_a_loud_error(rig):
    model = make_tokenizer(rig)
    pred = make_prediction(rig, frames=FRAMES)
    labels = make_labels(frames=FRAMES // 2)
    with pytest.raises(ValueError, match="covers .* frames but the sampled window has"):
        model.sole_region_losses(labels, pred)


def test_labels_never_receive_gradient(rig):
    model = make_tokenizer(rig)
    pred = make_prediction(rig)
    labels = make_labels()
    totals, _ = model.sole_region_losses(labels, pred)
    sum(totals.values()).backward()
    assert all(not value.requires_grad for value in labels.values())


def test_a_window_without_valid_contact_gives_a_finite_loss_and_zero_counts(rig):
    model = make_tokenizer(rig)
    pred = make_prediction(rig)
    totals, counts = model.sole_region_losses(make_labels(measured=False), pred)
    assert counts["contact_frames_supervised"] == 0
    assert counts["measured_stable_intervals_supervised"] == 0
    assert counts["measured_vertices_supervised"] == 0
    for value in totals.values():
        assert torch.isfinite(value)
    # Contact height and slip need evidence, so they contribute nothing here.
    assert float(totals["l_contact_height_m"]) == 0.0
    assert float(totals["l_measured_slip_mps"]) == 0.0
    # Penetration is a geometric fact about the motion and deliberately does not
    # depend on the labels, so it may still be nonzero.
    assert float(totals["l_penetration_m"]) >= 0.0


def test_a_missing_label_is_a_loud_error(rig):
    model = make_tokenizer(rig)
    pred = make_prediction(rig)
    labels = make_labels()
    del labels["sole_vertex_measured_right"]
    with pytest.raises(ValueError, match="sole_vertex_measured_right"):
        model.sole_region_losses(labels, pred)


def test_replaced_geometry_coefficients_must_be_off(rig):
    """Two penalties on the same geometry would be a silent double count."""
    from motionbricks.data.bones_models import REPLACED_GEOMETRY_COEFFICIENTS
    assert set(REPLACED_GEOMETRY_COEFFICIENTS) == {"trusted_skate_coeff", "penetration_coeff",
                                                   "stance_anchor_coeff", "support_height_coeff"}
    args = {"sole_region_supervision": True, "sole_region_labels_version": "v1", "penetration_coeff": 0.5}
    enabled = [name for name in REPLACED_GEOMETRY_COEFFICIENTS if args.get(name, 0.0)]
    assert enabled == ["penetration_coeff"]


def write_item(shards, labels_dir, name, frames, start_offset=0):
    """A synthetic shard plus its labels, with a recognisable offset in the labels."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(frames, 8)).astype(np.float32)
    np.savez_compressed(shards / f"{name}.npz", global_features=features,
                        contact_confidence=np.ones((frames, 4), dtype=bool))
    labels = {"frame_state_left": np.arange(frames, dtype=np.int8) + start_offset,
              "frame_state_right": np.arange(frames, dtype=np.int8) + start_offset,
              "interval_state_left": np.arange(frames - 1, dtype=np.int8) + start_offset,
              "interval_state_right": np.arange(frames - 1, dtype=np.int8) + start_offset,
              "region_measured_left": np.ones(frames - 1, dtype=bool),
              "region_measured_right": np.ones(frames - 1, dtype=bool),
              "vertex_measured_left": np.ones((frames - 1, 5), dtype=bool),
              "vertex_measured_right": np.ones((frames - 1, 5), dtype=bool)}
    np.savez_compressed(labels_dir / f"{name}.npz", **labels)
    return labels


def test_the_label_window_follows_the_single_sampling_index(tmp_path):
    """Frame labels are one shorter than the global window and share its start."""
    shards, labels_dir = tmp_path / "shards", tmp_path / "labels"
    shards.mkdir(), labels_dir.mkdir()
    frames = 129
    truth = write_item(shards, labels_dir, "clip", frames)
    stats = type("Stats", (), {"eps": 1e-5, "legacy": False,
                               "mean": torch.zeros(8), "std": torch.ones(8)})()
    stats.register_from_tensors = lambda *a, **k: None
    for training in (False, True):
        dataset = BonesMotionDataset(shards, [{"filename": "clip", "shard": "clip.npz"}], stats,
                                     crop_frames=frames, training=training,
                                     sole_region_folder=labels_dir)
        item = dataset[0]
        assert set(SOLE_LABEL_KEYS).issubset(item)
        assert item["sole_frame_state_left"].shape == (frames - 1,)
        assert item["sole_interval_state_left"].shape == (frames - 2,)
        assert item["sole_vertex_measured_left"].shape == (frames - 2, 5)
        start = 0
        assert int(item["sole_frame_state_left"][0]) == truth["frame_state_left"][start]
        assert int(item["sole_interval_state_left"][-1]) == truth["interval_state_left"][start + frames - 3]
    batch = collate_bones_batch([dataset[0], dataset[0]])
    assert batch["sole_frame_state_left"].shape == (2, frames - 1)
    assert batch["sole_vertex_measured_left"].shape == (2, frames - 2, 5)
    assert int(batch["sole_frame_state_left"].shape[1]) == int(batch["motion_len"][0]) - 1


def test_missing_label_files_are_reported(tmp_path):
    shards, labels_dir = tmp_path / "shards", tmp_path / "labels"
    shards.mkdir(), labels_dir.mkdir()
    write_item(shards, labels_dir, "clip", 129)
    (labels_dir / "clip.npz").rename(labels_dir / "elsewhere.npz")
    stats = type("Stats", (), {"eps": 1e-5, "legacy": False,
                               "mean": torch.zeros(8), "std": torch.ones(8)})()
    stats.register_from_tensors = lambda *a, **k: None
    dataset = BonesMotionDataset(shards, [{"filename": "clip", "shard": "clip.npz"}], stats,
                                 crop_frames=129, training=False, sole_region_folder=labels_dir)
    with pytest.raises(FileNotFoundError, match="Missing sole-region labels"):
        dataset[0]


def test_labels_follow_the_resampled_rows_and_the_random_window_start():
    """Regression: the trainer resamples rows and takes a random window, never a prefix."""
    clip, window, batch = 200, 129, 4
    values = torch.arange(batch, dtype=torch.float32)[:, None] * 1000 + torch.arange(clip, dtype=torch.float32)
    rows = torch.tensor([2, 0, 3, 1])
    starts = torch.tensor([11, 15, 0, 23])
    indices = starts[:, None] + torch.arange(window)[None, :]
    gathered = align_labels_to_sampled_segment(values, rows, indices, window - 1)
    assert gathered.shape == (batch, window - 1)
    for row in range(batch):
        assert float(gathered[row, 0]) == float(values[rows[row], starts[row]])
        assert float(gathered[row, -1]) == float(values[rows[row], starts[row] + window - 2])
    # Never the prefix, which is what the misaligned version produced.
    assert not torch.equal(gathered, values[rows][:, :window - 1])


def test_per_vertex_masks_are_gathered_on_the_same_frame_axis():
    clip, window, batch, vertices = 200, 129, 3, 5
    values = torch.zeros(batch, clip - 1, vertices, dtype=torch.bool)
    for row in range(batch):
        values[row, :, row] = True
    rows = torch.tensor([1, 2, 0])
    starts = torch.tensor([7, 0, 12])
    indices = starts[:, None] + torch.arange(window)[None, :]
    gathered = align_labels_to_sampled_segment(values, rows, indices, window - 2)
    assert gathered.shape == (batch, window - 2, vertices)
    for row in range(batch):
        assert bool(gathered[row, 0, rows[row]])


def test_alignment_rejects_indices_that_reach_past_the_labels():
    values = torch.zeros(2, 10)
    with pytest.raises(ValueError, match="reach past"):
        align_labels_to_sampled_segment(values, torch.tensor([0, 1]), torch.full((2, 5), 10), 5)
    with pytest.raises(ValueError, match="same rows"):
        align_labels_to_sampled_segment(values, torch.tensor([0]), torch.zeros(2, 5), 5)


def test_sole_label_keys_split_frame_labels_from_interval_labels():
    batch = {"motion": 1, "sole_frame_state_left": 2, "sole_frame_state_right": 3,
             "sole_interval_state_left": 4, "sole_vertex_measured_right": 5, "contact_confidence": 6}
    assert sole_frame_label_keys(batch) == ["sole_frame_state_left", "sole_frame_state_right"]
    assert sole_interval_label_keys(batch) == ["sole_interval_state_left", "sole_vertex_measured_right"]


def test_variable_clip_lengths_pad_labels_as_undecided_not_contact():
    items = []
    for frames in (129, 160):
        item = {"keyid": frames, "motion": torch.zeros(frames, 8),
                "contact_confidence": torch.ones(frames, 4, dtype=torch.bool)}
        item.update({name: values[0] for name, values in make_labels(frames=frames - 1).items()})
        items.append(item)
    batch = collate_bones_batch(items)
    assert batch["sole_frame_state_left"].shape == (2, 159)
    assert (batch["sole_frame_state_left"][0, 128:] == FRAME_STATES.index("undecided")).all()
    assert (batch["sole_interval_state_left"][0, 127:] == INTERVAL_STATES.index("undecided")).all()
    assert not batch["sole_vertex_measured_left"][0, 127:].any()
    assert not batch["sole_region_measured_left"][0, 127:].any()


def test_real_loss_includes_multi_frame_stance_displacement(rig):
    model = make_tokenizer(rig, sole_region_coeff_stance_displacement_m=10.0)
    pred = make_prediction(rig, frames=24)
    totals, counts = model.sole_region_losses(make_labels(frames=24), pred)
    assert "l_stance_displacement_m" in totals
    assert counts["stance_displacement_pairs"] > 0
    assert torch.isfinite(totals["l_stance_displacement_m"])


def test_stance_displacement_actually_enters_total_tokenizer_loss_with_its_weight():
    from kimodo.exports.motion_io import complete_motion_dict
    from motionbricks.motionlib.core.utils.stats import Stats
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE

    rep = build_representation()
    local = rep.dual_rep.local_motion_rep
    stats = Stats()
    stats.register_from_tensors(torch.zeros(len(local.indices["all"])), torch.ones(len(local.indices["all"])))
    local.stats = stats
    frames = 24
    rotations = torch.eye(3).expand(frames, 30, 3, 3).clone()
    root = torch.zeros(frames, 3)
    root[:, 1] = 1.0
    motion = complete_motion_dict(rotations, root, rep.skeleton, 60)
    target = local({"posed_joints": motion["posed_joints"], "local_joint_rots": motion["local_rot_mats"],
                    "global_joint_rots": motion["global_rot_mats"], "translation": root,
                    "foot_contacts": torch.ones(frames, 4)}, to_normalize=True)[None].detach()
    predicted = target.clone()
    predicted[..., local.indices["local_root_vel"][0]] += 0.01
    predicted[..., local.indices["ric_data"][0]] += 0.02
    predicted.requires_grad_(True)
    network = VQVAE(pose_root_mode="pose", motion_rep=local, encoder_state_dim=0, decoder_state_dim=0,
                    decoder_target_cond_dim=0, decoder_external_cond_dim=0,
                    feature_mode=["joint_positions_and_rotations", "pose", "joint_positions_and_rotations",
                                  "root_without_hip_height"], code_dim=16, output_emb_width=16, width=16,
                    depth=1, down_t=2, stride_t=2, nb_code=16, num_heads=2, kmeans_init=False)
    model = BonesTokenizer(pose_vqvae_network=network, root_vqvae_network=None, motion_rep=rep,
                           optimizer=None, scheduler=None, args={"commit_loss_coeff": 0.0,
                           "sole_region_supervision": True, "sole_region_labels_version": "test-v3"})
    batch = dict(make_labels(frames=frames), local_motions=target,
                 contact_confidence=torch.ones(1, frames, 4, dtype=torch.bool))
    output = {"recon_state": predicted, "perplexity": torch.tensor(1.0), "l_commit": torch.tensor(0.0)}
    base = model.loss(batch, output, None)["loss"]
    model.args["sole_region_coeff_stance_displacement_m"] = 10.0
    weighted = model.loss(batch, output, None)
    torch.testing.assert_close(weighted["loss"] - base, 10.0 * weighted["l_stance_displacement_m"])
    assert weighted["l_stance_displacement_m"] > 0
    model.args["kinematic_position_consistency_coeff"] = 5.0
    model.args["kinematic_velocity_consistency_coeff"] = 0.5
    consistent = model.loss(batch, output, None)
    torch.testing.assert_close(consistent["loss"] - weighted["loss"],
                               5.0 * consistent["l_kinematic_position_consistency_m"]
                               + 0.5 * consistent["l_kinematic_velocity_consistency_mps"])
    assert consistent["l_kinematic_position_consistency_m"] > 0
    assert consistent["l_kinematic_velocity_consistency_mps"] > 0
    consistent["loss"].backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0
    assert not target.requires_grad
