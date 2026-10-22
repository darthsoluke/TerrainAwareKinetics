from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_native_mb_stair_tokens import (
    compare_codebooks,
    compare_embedding_codebook_alignment,
    compare_pose_checkpoints,
    compare_token_ids,
    validate_window_rows,
    tokenizer_keyframe_curriculum,
    backbone_keyframe_curriculum,
)


def _windows():
    return [
        {"group": "stairs", "behavior": "up", "shard": "s-up", "start": i}
        for i in range(3)
    ] + [
        {"group": "stairs", "behavior": "down", "shard": "s-down", "start": i}
        for i in range(3)
    ] + [
        {"group": "flat", "behavior": behavior, "shard": f"flat-{behavior}", "start": i}
        for behavior in ("run", "walk") for i in range(3)
    ]


def test_window_contract_requires_twelve_unique_heldout_examples():
    rows = _windows()
    validate_window_rows(rows[:6], rows[6:])
    with pytest.raises(ValueError, match="Expected report-bound"):
        validate_window_rows(rows[:5], rows[5:])
    duplicate = [dict(row) for row in rows[:6]]
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="unique"):
        validate_window_rows(duplicate, rows[6:])


def test_codebook_metrics_keep_numeric_ids_unmapped_and_report_nearest_rows():
    parent = np.asarray([
        [[1., 0.], [0., 1.], [-1., 0.]],
        [[0., 1.], [1., 0.], [0., -1.]],
    ])
    child = parent.copy()
    child[0] = parent[0, [1, 0, 2]]
    result = compare_codebooks(parent, child)
    assert result["shape_head_code_dim"] == [2, 3, 2]
    assert result["heads"][0]["same_id_cosine"]["mean"] == pytest.approx(1 / 3)
    assert result["heads"][0]["nearest_old_by_cosine_same_id_fraction"] == pytest.approx(1 / 3)
    assert result["heads"][0]["nearest_old_by_cosine_old_id_counts"] == [1, 1, 1]
    with pytest.raises(ValueError, match="matching"):
        compare_codebooks(parent, child[:, :2])


def test_token_id_agreement_is_per_head_and_per_group_without_remap():
    windows = _windows()
    parent = np.zeros((12, 2, 2), dtype=np.int64)
    child = parent.copy()
    child[0, :, 0] = 1
    child[6:, :, 1] = 2
    result = compare_token_ids(parent, child, windows)
    assert result["equal_id_fraction_per_head"] == pytest.approx([11 / 12, .5])
    assert result["stairs_equal_id_fraction_all"] == pytest.approx(11 / 12)
    assert result["flat_equal_id_fraction_all"] == pytest.approx(.5)
    assert result["windows"][0]["changed_id_counts_per_head"] == [2, 0]


def test_pose_comparison_reports_checkpoint_step_lr_and_tensor_deltas():
    parent = {"global_step": 50_000, "optimizer_states": [{"param_groups": [{"lr": 1e-5}]}],
              "lr_schedulers": [{"last_epoch": 50_000, "_last_lr": [1e-5]}],
              "state_dict": {
                  "backbone_net._pose_token_emb.weight": np.zeros((4, 2)),
                  "backbone_net._proj_pose_output_logit.weight": np.zeros((3, 4)),
                  "backbone_net._proj_pose_output_logit.bias": np.zeros(4),
              }}
    child = {"global_step": 5_000, "optimizer_states": [{"param_groups": [{"lr": 5e-5}]}],
             "lr_schedulers": [{"last_epoch": 5_000, "_last_lr": [5e-5]}],
             "state_dict": {
                 "backbone_net._pose_token_emb.weight": np.ones((4, 2)),
                 "backbone_net._proj_pose_output_logit.weight": np.zeros((3, 4)),
                 "backbone_net._proj_pose_output_logit.bias": np.zeros(4),
             }}
    result = compare_pose_checkpoints(parent, child)
    assert result["parent_runtime"]["global_step"] == 50_000
    assert result["child_runtime"]["optimizer_learning_rates"][0][0]["lr"] == pytest.approx(5e-5)
    assert result["tensors"]["input_embedding"]["exact_equal"] is False
    assert result["tensors"]["input_embedding"]["delta_l2"] == pytest.approx(np.sqrt(8.))
    assert result["tensors"]["output_weight"]["exact_equal"] is True


def test_tokenizer_curriculum_reports_step_specific_target_counts(tmp_path):
    config = {"model": {"args": {
        "keyframe_num_warmup_steps": 200_000,
        "root_vqvae_max_num_keyframes": 5,
        "root_vqvae_no_keyframe_prob": .25,
        "pose_vqvae_max_num_keyframes": 10,
        "pose_vqvae_no_keyframe_prob": .25,
    }}}
    at_five_k = tokenizer_keyframe_curriculum(config, 5_000)
    assert at_five_k["modules"]["root"]["scheduled_maximum"] == 1
    assert at_five_k["modules"]["pose"]["scheduled_maximum"] == 1
    assert at_five_k["modules"]["pose"]["count_probabilities"] == pytest.approx([.25, .75] + [0.] * 9)
    at_fifty_k = tokenizer_keyframe_curriculum(config, 50_000)
    assert at_fifty_k["modules"]["root"]["scheduled_maximum"] == 1
    assert at_fifty_k["modules"]["pose"]["scheduled_maximum"] == 2
    assert at_fifty_k["modules"]["pose"]["count_probabilities"][:3] == pytest.approx([.25, .375, .375])


def test_pose_embedding_alignment_uses_head_offsets_and_skips_mask_rows():
    table = np.asarray([[1., 0.], [0., 1.], [2., 2.], [0., 1.], [1., 0.], [3., 3.]])
    codebook = np.asarray([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    report = compare_embedding_codebook_alignment(
        {"state_dict": {"backbone_net._pose_token_emb.weight": table}}, codebook)
    assert report["heads"][0]["same_id_cosine"]["mean"] == pytest.approx(1.)
    assert report["heads"][1]["same_id_cosine"]["mean"] == pytest.approx(1.)
    assert report["heads"][0]["mask_embedding_l2"] == pytest.approx(np.sqrt(8.))


def test_root_and_pose_curricula_use_their_own_stage_keyframe_parameters():
    root = {"model": {"args": {
        "keyframe_num_warmup_steps": 200_000,
        "max_num_start_keyframes": 4, "no_start_keyframe_prob": 0.,
        "max_num_end_keyframes": 4, "no_end_keyframe_prob": 0.,
    }}}
    pose = {"model": {"args": {
        "keyframe_num_warmup_steps": 200_000,
        "max_num_start_keyframes": 4, "no_start_keyframe_prob": 0.,
        "max_num_end_keyframes": 4, "no_end_keyframe_prob": 0.,
        "max_num_middle_keyframes": 8, "no_middle_keyframe_prob": .5,
    }}}
    root_result = backbone_keyframe_curriculum(root, "root", 5_000)
    pose_result = backbone_keyframe_curriculum(pose, "pose", 50_000)
    assert root_result["components"]["start"]["count_probabilities"] == pytest.approx([0., 1., 0., 0., 0.])
    assert pose_result["components"]["middle"]["count_probabilities"] == pytest.approx(
        [.5, .25, .25, 0., 0., 0., 0., 0., 0.])
