"""Focused tests for the offline decoder-height response probe."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_codec_height import (
    HEIGHT_EXTERNAL_FEATURE_MODE,
    SOURCE_EXTERNAL_FEATURE_MODE,
)
from motionbricks.data.native_mb_online_live import planar_command_goal
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.review_parc_decoder_height_response import (
    RAMP_FRAME_COUNT,
    RAMP_START_FRAME,
    _decoder_gain_record,
    apply_height_ramp,
    capture_decoder_call,
    height_feature_location,
    unavailable_planar_response,
)
from test_native_mb_codec_height import codec_pair


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("endpoint", [1.0, -1.0])
def test_meter_ramp_changes_only_derived_external_root_y(codec_pair, endpoint):
    _, _, _, _, tokenizer, _, local_motion = codec_pair
    codec = tokenizer.pose_net
    local_rep = codec.motion_rep
    external = extract_feature_from_motion_rep(
        local_motion, local_rep, codec.decoder_external_cond_feature_mode)
    location = height_feature_location(local_rep, HEIGHT_EXTERNAL_FEATURE_MODE)
    assert location["available"]
    assert location["feature_width"] == 3

    changed, ramp_info = apply_height_ramp(
        external, local_rep, HEIGHT_EXTERNAL_FEATURE_MODE, endpoint_m=endpoint)
    y_channel = location["global_root_y_channel_index"]
    other_channels = [index for index in range(external.shape[-1]) if index != y_channel]
    assert torch.equal(changed[..., other_channels], external[..., other_channels])
    assert torch.equal(changed[:, :RAMP_START_FRAME, y_channel], external[:, :RAMP_START_FRAME, y_channel])
    assert torch.equal(changed[:, RAMP_START_FRAME + RAMP_FRAME_COUNT:, y_channel],
                       external[:, RAMP_START_FRAME + RAMP_FRAME_COUNT:, y_channel])

    rep_index = np.asarray([location["global_root_y_rep_index"]], dtype=np.int64)
    raw_before = local_rep.stats.unnormalize(external[..., y_channel:y_channel + 1], index=rep_index)
    raw_after = local_rep.stats.unnormalize(changed[..., y_channel:y_channel + 1], index=rep_index)
    expected = torch.zeros_like(raw_before)
    expected[:, RAMP_START_FRAME:RAMP_START_FRAME + RAMP_FRAME_COUNT, 0] = torch.tensor(
        ramp_info["expected_ramp_m"])
    torch.testing.assert_close(raw_after - raw_before, expected, atol=2e-5, rtol=0)
    assert ramp_info["roundtrip_max_abs_error_m"] <= 2e-5


def test_feature_indices_make_planar_y_gain_unavailable_not_zero(codec_pair):
    _, _, _, _, tokenizer, _, _ = codec_pair
    local_rep = tokenizer.pose_net.motion_rep
    target = height_feature_location(local_rep, HEIGHT_EXTERNAL_FEATURE_MODE)
    planar = height_feature_location(local_rep, SOURCE_EXTERNAL_FEATURE_MODE)
    assert target["available"] and target["feature_width"] == 3
    assert not planar["available"] and planar["feature_width"] == 2
    result = unavailable_planar_response(SOURCE_EXTERNAL_FEATURE_MODE, 2)
    assert result["status"] == "unavailable_not_zero"
    assert result["gain_mean"] is None
    assert result["gain_last"] is None


def test_gain_reads_root_y_from_full_decoder_output_not_target_condition_layout(codec_pair):
    _, _, _, _, tokenizer, _, _ = codec_pair
    codec = tokenizer.pose_net
    output_mode = codec.decoder_input_feature_mode
    assert output_mode == "pose"
    output_location = height_feature_location(codec.motion_rep, output_mode)
    target_location = height_feature_location(
        codec.motion_rep, codec.decoder_target_cond_feature_mode)
    assert output_location["available"] and output_location["feature_width"] == 365
    assert output_location["global_root_y_channel_index"] == 3
    assert target_location["available"] and target_location["global_root_y_channel_index"] == 0

    frame_count = 24
    expected_y = torch.linspace(0.7, 1.3, frame_count).reshape(1, frame_count, 1)
    height_index = np.asarray([output_location["global_root_y_rep_index"]], dtype=np.int64)
    yaw_index = np.asarray(codec.motion_rep.indices["local_root_rot_vel"], dtype=np.int64)
    normalized_y = codec.motion_rep.stats.normalize(expected_y, index=height_index)
    yaw_raw = torch.full((1, frame_count, 1), -3.0)
    normalized_yaw = codec.motion_rep.stats.normalize(yaw_raw, index=yaw_index)
    normalized_counterfactual_y = codec.motion_rep.stats.normalize(
        expected_y + 0.2, index=height_index)
    reference = torch.zeros(1, frame_count, output_location["feature_width"])
    counterfactual = reference.clone()
    reference[..., output_location["global_root_y_channel_index"]] = normalized_y[..., 0]
    counterfactual[..., output_location["global_root_y_channel_index"]] = normalized_counterfactual_y[..., 0]
    reference[..., int(yaw_index[0])] = normalized_yaw[..., 0]
    counterfactual[..., int(yaw_index[0])] = normalized_yaw[..., 0]

    ramp_info = {
        "expected_ramp_m": [0.1] * 12,
        "start_frame": 4,
        "frame_indices": list(range(4, 16)),
        "roundtrip_max_abs_error_m": 0.0,
    }
    record = _decoder_gain_record(reference, counterfactual, codec, 1.0, ramp_info, frame_count)
    torch.testing.assert_close(
        torch.tensor(record["reference_decoded_root_y_m"]), expected_y[0, :, 0],
        atol=1e-6, rtol=0,
    )
    assert record["decoded_output_feature_mode"] == "pose"
    assert record["decoded_root_y_location"]["global_root_y_channel_index"] == 3
    assert record["decoded_root_y_delta_vs_reference_m"][4] == pytest.approx(0.2, abs=1e-6)


def test_decoder_capture_returns_original_result_and_restores_method_on_error():
    class TinyCodec:
        def forward_decoder(self, tokens, target_cond, has_target_cond=None, external_cond=None,
                            use_overall_indices=True, token_mask=None):
            return {"recon_state": external_cond + tokens[:, :1, :1]}

    codec = TinyCodec()
    tokens = torch.ones(1, 3, 1)
    target = torch.zeros(1, 12, 4)
    known = torch.zeros(1, 12, dtype=torch.bool)
    external = torch.zeros(1, 12, 2)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    with capture_decoder_call(codec) as calls:
        result = codec.forward_decoder(tokens, target, known, external, False, token_mask)
        assert torch.equal(result["recon_state"], torch.ones_like(external))
    assert len(calls) == 1
    assert calls[0]["kwargs"] == {}
    torch.testing.assert_close(calls[0]["args"][0], tokens)
    torch.testing.assert_close(calls[0]["recon_state"], result["recon_state"])
    assert "forward_decoder" not in codec.__dict__

    with pytest.raises(RuntimeError, match="restore"):
        with capture_decoder_call(codec):
            raise RuntimeError("restore")
    assert "forward_decoder" not in codec.__dict__


def test_requested_planar_intent_uses_existing_runtime_goal_helper():
    root = torch.zeros(1, 4, 5)
    root[..., 3] = 1.0
    goal = planar_command_goal(root, (0.0, 1.6), 0.0, 8)
    assert goal.shape == (1, 4)
    assert goal[0, 0].item() == pytest.approx(0.0)
    assert goal[0, 1].item() == pytest.approx(1.6 * (4 * 8 - 4) / 30)
    torch.testing.assert_close(goal[0, 2:], torch.tensor([1.0, 0.0]))
