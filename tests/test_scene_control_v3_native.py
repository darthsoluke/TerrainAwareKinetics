"""CPU-only native v3 mapping and official full-skin transport parity."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix, matrix_to_cont6d
from motionbricks.scene_control_v3.native import FRAME_BYTES, NativeSOMAAdapter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapter():
    torch.set_num_threads(2)
    config = json.loads((ROOT / "configs/scene_control_v3_performance.json").read_text())
    return NativeSOMAAdapter(ROOT / config["assets"]["bundle"]["path"])


def _prediction(adapter, history, delta):
    pose = adapter.source_frame(history[:, -1:])
    rotation = angle_to_Y_rotation_matrix(history.new_tensor([0.23]))[:, None] @ pose["global_joint_rots"][:, 0]
    return dict(root_delta=history.new_tensor([delta]), heading_delta=history.new_tensor([[1.2]]),
                global_joint_rot6d=matrix_to_cont6d(rotation), contact_logits=history.new_zeros(1, 15))


@torch.inference_mode()
def test_actor_root_and_global_rotations_are_authoritative(adapter):
    history = adapter.initial_history.clone()
    before = history.clone()
    output = _prediction(adapter, history, [0.07, 0.02, -0.04])
    result = adapter.step(history, output)
    torch.testing.assert_close(history, before, atol=0, rtol=0)
    torch.testing.assert_close(result["root"][:, 0], history[:, -1, :3] + output["root_delta"], atol=2e-6, rtol=0)
    from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix
    torch.testing.assert_close(result["global_joint_rots"][:, 0], cont6d_to_matrix(output["global_joint_rot6d"]),
                               atol=2e-6, rtol=0)
    assert result["history"].shape == (1, 4, 366)
    assert result["local_history"].shape == (1, 4, 365)
    assert result["dual_history"].shape == (1, 4, 370)
    rep = adapter.representation
    torch.testing.assert_close(result["history"], reencode_played_history(result["history"], rep), atol=3e-5, rtol=0)
    torch.testing.assert_close(result["dual_history"][..., rep.dual_rep.indices["global_rep"]], result["history"])
    torch.testing.assert_close(result["dual_history"][..., rep.dual_rep.indices["local_rep"]], result["local_history"])
    ric = rep.inverse(result["history"], False, joint_positions_from="ric_data")["posed_joints"]
    fk = rep.inverse(result["history"], False, joint_positions_from="global_rot_data")["posed_joints"]
    torch.testing.assert_close(ric, fk, atol=3e-5, rtol=0)


@torch.inference_mode()
def test_official_full18056_skin_and_lossless_rig_packet(adapter):
    history = adapter.initial_history
    frame = adapter.step(history, _prediction(adapter, history, [-0.03, 0.01, 0.06]))
    official = load_official_skinner(adapter.representation.skeleton)
    expected = official.skin(frame["global_joint_rots"][0], frame["joints"][0], rot_is_global=True)
    actual = adapter.fullskin(frame["rig"])[0]
    assert actual.shape == (1, 18056, 3)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(frame["body_points"], adapter.fullskin(frame["rig"]).index_select(
        -2, adapter.surface_sampler.selected_vertex_ids), atol=3e-5, rtol=0)
    payload = adapter.pack_frame(frame)
    assert len(payload) == FRAME_BYTES == 4940
    wire = np.frombuffer(payload, dtype="<f4")
    np.testing.assert_array_equal(wire[:3], frame["root"][0, 0].numpy())
    np.testing.assert_array_equal(wire[3:].reshape(77, 4, 4), frame["rig"][0, 0].numpy())
    asset = adapter.skin_asset()
    assert len(asset["bind_vertices"]) == 18056 * 3
    assert len(asset["weights"]) == len(asset["indices"]) == 18056 * asset["influences_per_vertex"]
    assert len(asset["inverse_bind"]) == 77 * 16


@torch.inference_mode()
def test_recurrent_state_uses_predicted_frames_only(adapter):
    state = adapter.initial_history.clone()
    start_root = state[:, -1, :3].clone()
    delta = [0.012, -0.001, 0.025]
    for _ in range(6):
        result = adapter.step(state, _prediction(adapter, state, delta))
        state = result["history"]
    torch.testing.assert_close(state[:, -1, :3], start_root + state.new_tensor([delta]) * 6, atol=3e-6, rtol=0)
    assert torch.isfinite(state).all()


@torch.inference_mode()
def test_real646_source_bridge_preserves_root_and_official_fullskin(adapter):
    from motionbricks.scene_control_v2.provider import load_bound_world
    spec = json.loads((ROOT / "configs/scene_control_v2_real_provider_cpu.json").read_text())["provider"]
    world, source_joints, _, _, _ = load_bound_world(spec, adapter.representation)
    assert world.shape == (646, 366)
    selected = world[[0, 315, 645]][None]
    frame = adapter.source_frame(selected)
    torch.testing.assert_close(frame["root"], selected[..., :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(frame["joints"], source_joints[[0, 315, 645]][None], atol=3e-5, rtol=0)
    official = load_official_skinner(adapter.representation.skeleton)
    expected = official.skin(frame["global_joint_rots"][0], frame["joints"][0], rot_is_global=True)
    torch.testing.assert_close(adapter.fullskin(frame["rig"])[0], expected, atol=2e-6, rtol=0)


@torch.inference_mode()
def test_unknown_actor_scene_is_rejected_at_publication(adapter):
    frame = adapter.source_frame(adapter.initial_history[:, -1:])
    with pytest.raises(RuntimeError, match="no capacity frame published"):
        adapter.pack_actor_frame(frame, torch.tensor([False]))
    assert adapter.pack_actor_frame(frame, torch.tensor([True])) == adapter.pack_frame(frame)


@torch.inference_mode()
def test_batch_flattened_lbs_matches_official_frames_exactly(adapter):
    history = adapter.initial_history.repeat(2, 1, 1)
    history[1, :, :3] += history.new_tensor([3, 0.2, -2])
    frame = adapter.source_frame(history)
    official = load_official_skinner(adapter.representation.skeleton)
    expected = official.skin(frame["global_joint_rots"].reshape(-1, 30, 3, 3),
                             frame["joints"].reshape(-1, 30, 3), rot_is_global=True)
    actual = adapter.fullskin(frame["rig"])
    assert actual.shape == (2, 4, 18056, 3)
    torch.testing.assert_close(actual.reshape(-1, 18056, 3), expected, atol=0, rtol=0)


def test_experiment_failure_preserves_partial_receipt(tmp_path, monkeypatch):
    from scripts import benchmark_scene_control_v3 as runner
    output = tmp_path / "experiment_failure"
    monkeypatch.setattr("sys.argv", ["benchmark_scene_control_v3.py", "--mode", "prepare", "--output", str(output)])
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda _: None)

    def failed_assets(_):
        raise RuntimeError("synthetic CPU experiment failure")

    monkeypatch.setattr(runner, "asset_identities", failed_assets)
    with pytest.raises(RuntimeError, match="synthetic CPU experiment failure"):
        runner.main()
    failure = json.loads((output / "failure.json").read_text())
    partial = json.loads((output / "receipt.json").read_text())
    assert partial["status"] == "failed"
    assert failure["completed_steps"] == failure["completed_warmup"] == 0
    assert failure["progress"]["phase"] == "environment_and_immutable_assets"


@pytest.fixture
def capacity_pipeline(adapter):
    from motionbricks.scene_control_v3 import CausalSceneActor, SceneActorConfig
    from motionbricks.scene_control_v3.geometry import build_true3d_scene
    from motionbricks.scene_control_v3.pipeline import CapacityPipeline
    torch.manual_seed(20261018)
    actor = CausalSceneActor(SceneActorConfig(preview_frames=64)).eval().requires_grad_(False)
    scene = build_true3d_scene(device="cpu")
    velocity = torch.zeros(1, scene.object_count, 6)
    velocity[:, 1, 0] = 0.06
    return CapacityPipeline(adapter, actor, scene, adapter.initial_history, object_velocity=velocity)


@torch.inference_mode()
def test_shared_pipeline_commits_only_matching_rendered_ack_once(capacity_pipeline):
    pipeline = capacity_pipeline
    initial = pipeline.state
    history, hidden, pose = initial.history.clone(), initial.hidden.clone(), initial.object_pose.clone()
    body_points = initial.body_points.clone()
    root_plan = initial.root_plan_world_m.clone()
    stages = []
    frame = pipeline.propose((0.45, 0.8, 0.25), stage_callback=stages.append)
    assert len(frame.payload) == 4940
    assert pipeline.state is initial
    torch.testing.assert_close(pipeline.state.history, history, atol=0, rtol=0)
    torch.testing.assert_close(pipeline.state.hidden, hidden, atol=0, rtol=0)
    torch.testing.assert_close(pipeline.state.object_pose, pose, atol=0, rtol=0)
    torch.testing.assert_close(pipeline.state.body_points, body_points, atol=0, rtol=0)
    torch.testing.assert_close(pipeline.state.root_plan_world_m, root_plan, atol=0, rtol=0)
    assert frame.prediction["root_preview"].shape == (1, 64, 3)
    assert bool(frame.prediction["input_valid"].all())
    assert stages == ["contact_cpu_admit_pack_and_transfers", "query_and_point_construction", "encode", "actor",
                      "native_rig_and_history"]
    with pytest.raises(RuntimeError, match="awaiting its rendered ACK"):
        pipeline.propose((0, 1, 0))
    with pytest.raises(ValueError, match="does not match"):
        pipeline.commit(frame.frame_id + 1)
    assert pipeline.commit(frame.frame_id) is True
    assert pipeline.state is frame.next_state
    from motionbricks.scene_control_v3.pipeline import root_preview_to_world
    torch.testing.assert_close(pipeline.state.root_plan_world_m,
        root_preview_to_world(initial.history, frame.prediction["root_preview"]), atol=0, rtol=0)
    assert pipeline.state.root_plan_origin_frame_id == initial.frame_id
    torch.testing.assert_close(pipeline.state.body_points, frame.native_frame["body_points"][:, 0], atol=0, rtol=0)
    assert pipeline.commit(frame.frame_id) is False
    second = pipeline.propose((-0.3, 0.8, -0.25))
    assert pipeline.commit(frame.frame_id) is False
    assert pipeline.pending is second
    assert second.query_metadata["committed_plan_kind"] == "actor_root_preview"
    assert second.query_metadata["committed_plan_age_frames"] == 1
    assert pipeline.commit(second.frame_id) is True


@torch.inference_mode()
def test_pipeline_unknown_scene_does_not_publish_or_advance_state(capacity_pipeline):
    pipeline = capacity_pipeline
    initial = pipeline.state
    pipeline.object_known.zero_()
    with pytest.raises(RuntimeError, match="no capacity frame published"):
        pipeline.propose((0, 0.8, 0))
    assert pipeline.pending is None
    assert pipeline.state is initial


def test_pipeline_visual_asset_is_the_actual_query_mesh(capacity_pipeline):
    pipeline = capacity_pipeline
    asset = pipeline.scene_asset()
    assert asset["kind"] == "triangle_meshes"
    assert len(asset["objects"]) == pipeline.scene.object_count == 3
    for obj, counts in zip(asset["objects"], pipeline.scene.storage_summary["objects"]):
        assert len(obj["vertices"]) == counts["vertices"] * 3
        assert len(obj["faces"]) == counts["triangles"] * 3
    assert pipeline.contract()["actor_status"] == "untrained_capacity_probe"
    assert pipeline.contract()["quality_approved"] is False


@torch.inference_mode()
def test_root_plan_seed_uses_only_observed_history_velocity(capacity_pipeline):
    state = capacity_pipeline.state
    offsets = torch.arange(1, 65, dtype=state.history.dtype)[None, :, None]
    expected = state.history[:, -1:, :3] + (state.history[:, -1:, :3] - state.history[:, -2:-1, :3]) * offsets
    torch.testing.assert_close(state.root_plan_world_m, expected, atol=0, rtol=0)
    assert state.root_plan_kind == "history_root_velocity_extrapolation"


@torch.inference_mode()
def test_committed_plan_and_opposite_latest_intent_have_distinct_query_sources(capacity_pipeline):
    pipeline = capacity_pipeline
    state = pipeline.state
    offsets = torch.arange(1, 65, dtype=state.history.dtype)[None, :, None] / 30
    plan = state.history[:, -1:, :3] + offsets * state.history.new_tensor([0, 0, 1])
    pipeline.state = replace(state, root_plan_world_m=plan, root_plan_kind="actor_root_preview")
    frame = pipeline.propose((0, -1, 0))
    points = frame.query["points_world_m"]
    ordered_body = state.body_points[:, pipeline.surface_ids]
    torch.testing.assert_close(points[:, :256], ordered_body[:, :256], atol=0, rtol=0)
    planned_shift = points[:, 256:384] - ordered_body[:, 256:384]
    intent_shift = points[:, 384:] - ordered_body[:, 384:]
    assert (planned_shift[..., 2] > 0).all()
    assert (intent_shift[..., 2] < 0).all()
    assert torch.bincount(frame.query["query_source"][0]).tolist() == [256, 128, 128]
    assert (frame.query["query_time_offset_s"][:, :256] == 0).all()
    assert (frame.query["query_time_offset_s"][:, 256:] > 0).all()
    assert frame.query_metadata["source_counts"] == dict(current_body=256, committed_root_plan=128, latest_intent=128)
    torch.testing.assert_close(frame.native_frame["root"][:, 0], state.history[:, -1, :3] + frame.prediction["root_delta"],
                               atol=2e-6, rtol=0)


@torch.inference_mode()
def test_root_preview_native_yaw_and_query_world_frame_covariance(capacity_pipeline):
    from motionbricks.scene_control_v3.pipeline import root_preview_to_world
    history = capacity_pipeline.state.history.clone()
    history[:, -1, :3] = history.new_tensor([4, 1, 7])
    history[:, -1, 3:5] = history.new_tensor([0, 1])
    preview = history.new_tensor([[[0, 0, 1], [1, 2, 0]]])
    torch.testing.assert_close(root_preview_to_world(history, preview), history.new_tensor([[[5, 1, 7], [4, 3, 6]]]),
                               atol=1e-6, rtol=0)
    state = capacity_pipeline.state
    command = history.new_tensor([[0.4, 0.8, 0.2]])
    points, times, _ = capacity_pipeline._query_inputs(state, command)
    rotation = angle_to_Y_rotation_matrix(history.new_tensor([0.7]))
    shift = history.new_tensor([[[3, 0.5, -2]]])
    moved_history = state.history.clone()
    moved_history[..., :3] = moved_history[..., :3] @ rotation.transpose(-1, -2) + shift
    moved = replace(state, history=moved_history,
        body_points=state.body_points @ rotation.transpose(-1, -2) + shift,
        root_plan_world_m=state.root_plan_world_m @ rotation.transpose(-1, -2) + shift)
    velocity = history.new_tensor([[[0.4, 0, 0.8]]]) @ rotation.transpose(-1, -2)
    moved_command = history.new_tensor([[float(velocity[0, 0, 0]), float(velocity[0, 0, 2]), 0.9]])
    moved_points, moved_times, _ = capacity_pipeline._query_inputs(moved, moved_command)
    torch.testing.assert_close(moved_points, points @ rotation.transpose(-1, -2) + shift, atol=3e-6, rtol=0)
    torch.testing.assert_close(moved_times, times, atol=0, rtol=0)
