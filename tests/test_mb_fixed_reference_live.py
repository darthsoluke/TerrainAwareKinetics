"""Exact fixed-reference loading and real-time clock initialization."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from motionbricks.mb_sonic.reference import G1_ARRAY_FIELDS, file_sha256, load_g1_reference, save_g1_reference
from motionbricks.mb_sonic.runtime import SonicLabRuntime
from scripts.isolate_mb_continuous_reference import load_fixed_reference
from motionbricks.mb_sonic.protomotions_policy import reference_window


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'out/mb_sonic_e0_e3_20261018/continuous_reference_isolation_v1'


def test_original_video_reference_survives_new_container_without_numeric_changes(tmp_path):
    reference, _ = load_fixed_reference(CASE)
    path = tmp_path/'reference.npz'
    save_g1_reference(path, reference)
    loaded = load_g1_reference(path, expected_sha256=file_sha256(path), scene_id=reference.scene_id)
    for field in G1_ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(loaded, field), getattr(reference, field))
    with pytest.raises(ValueError, match='SHA256'):
        load_g1_reference(path, expected_sha256='0'*64, scene_id=reference.scene_id)
    with pytest.raises(ValueError, match='scene'):
        load_g1_reference(path, expected_sha256=file_sha256(path), scene_id='wrong_scene')


def test_live_fixed_reference_keeps_original_initial_state_and_absolute_clock(tmp_path):
    reference, _ = load_fixed_reference(CASE)
    path = tmp_path/'reference.npz'
    save_g1_reference(path, reference)
    config = json.loads((ROOT/'configs/mb_protomotions_wasd_v32_reference_only.json').read_text())
    config.update(mode='fixed_reference', mb_planner_backend='thread', run_mode='finite', duration_s=16.,
        scene_root=str(CASE), initial_scene_id=reference.scene_id, enabled_scene_ids=[reference.scene_id],
        scene_manifest_sha256={reference.scene_id:file_sha256(CASE/'scene/scene_manifest.json')},
        fixed_reference=dict(path=str(path), sha256=file_sha256(path)), recording_dir=str(tmp_path/'recording'))
    config.pop('scene_catalog')
    config.pop('mb_conditioning_mode')
    lab = SonicLabRuntime(config, allocation_id='fixed-reference-constructor-test')
    try:
        np.testing.assert_array_equal(lab.physics.data.qpos, reference.qpos[0])
        np.testing.assert_array_equal(lab.physics.data.qvel, np.zeros(35))
        assert lab.physics.time_s == 0.
        assert lab._provider is None and lab.config()['intent_enabled'] is False
        assert lab.config()['control_mode'] == 'fixed_reference_read_only'
    finally:
        lab.shutdown()


def test_switching_exact_packets_preserves_every_tracker_future_window():
    reference, _ = load_fixed_reference(CASE)
    offsets = np.array([0., .02, .04, .08, .16])
    packet = None
    for time in .1+np.arange(795)*.02:
        start = np.floor((time+1e-8)/.4)*.4
        end = min(start+.76, reference.trusted_until_s)
        indices = np.flatnonzero((reference.timestamps_s >= start-1e-8)&(reference.timestamps_s <= end+1e-8))
        packet = replace(reference, trusted_until_s=float(reference.timestamps_s[indices[-1]]),
            **{field:getattr(reference,field)[indices] for field in G1_ARRAY_FIELDS})
        expected = reference_window(reference,time,offsets)
        actual = reference_window(packet,time,offsets)
        for field in G1_ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(actual,field),getattr(expected,field))
