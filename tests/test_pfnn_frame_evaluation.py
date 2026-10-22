"""Metric checks that distinguish Root drift from gait collapse."""

import pytest
import torch

from scripts.evaluate_pfnn_frame_actor import window_metrics
from test_pfnn_frame_actor import source, cpu_only


def test_identical_motion_has_zero_error_and_unit_gait_ratio(source):
    rep, frames = source
    for row in window_metrics(rep, frames, frames):
        assert row['root_mean_m'] == 0
        assert row['root_endpoint_m'] == 0
        assert row['body_relative_mpjpe_m'] == 0
        assert row['body_relative_velocity_rmse_mps'] == 0
        assert row['body_velocity_rms_ratio'] == pytest.approx(1)


def test_root_translation_is_not_body_deformation(source):
    rep, frames = source
    shifted = frames.clone()
    shifted[..., 0] += .5
    for row in window_metrics(rep, shifted, frames):
        assert row['root_mean_m'] == pytest.approx(.5)
        assert row['root_endpoint_m'] == pytest.approx(.5)
        assert row['body_relative_mpjpe_m'] < 1e-6
        assert row['body_velocity_rms_ratio'] == pytest.approx(1, abs=1e-4)


def test_static_pose_reports_missing_motion(source):
    rep, frames = source
    static = frames[:, 3:4].expand_as(frames).clone()
    for row in window_metrics(rep, static, frames):
        assert row['source_body_velocity_rms_mps'] > 0
        assert row['body_velocity_rms_ratio'] == pytest.approx(0, abs=1e-5)
