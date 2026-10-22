"""CPU contract tests using the released, unmodified walking references."""

from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import (
    EmbodimentConversionError, ExecutedHistory, G1_JOINT_NAMES,
    MUJOCO_FROM_ISAACLAB, OFFICIAL_BODY_NAMES, ReferenceCoverageError,
    NativeSomaHistory, load_official_reference,
    load_soma_reference, save_soma_reference,
)


ROOT = Path(__file__).resolve().parents[2]
WALK = ROOT / "gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428"


@pytest.fixture(scope="module")
def walking():
    return load_official_reference(WALK, scene_id="reference-contract-test")


def test_released_walking_schema_and_preserved_signals(walking):
    assert walking.qpos.shape == (455, 36)
    assert walking.qvel.shape == (455, 35)
    assert walking.body_pos_w.shape == (455, 14, 3)
    assert walking.body_quat_w.shape == (455, 14, 4)
    assert walking.joint_names == G1_JOINT_NAMES
    assert walking.body_names == OFFICIAL_BODY_NAMES
    for csv, actual in (("joint_pos", walking.joint_pos_isaaclab),
                        ("joint_vel", walking.joint_vel_isaaclab)):
        original = np.loadtxt(WALK / f"{csv}.csv", delimiter=",", skiprows=1)
        np.testing.assert_array_equal(actual, original)
    original = np.loadtxt(WALK / "joint_pos.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(walking.qpos[:, 7:], original[:, MUJOCO_FROM_ISAACLAB])
    np.testing.assert_array_equal(walking.qpos[:, :3], walking.body_pos_w[:, 0])
    np.testing.assert_array_equal(walking.qpos[:, 3:7], walking.body_quat_w[:, 0])
    assert len(walking.provenance["source_files_sha256"]) == 7


def test_root_angular_velocity_uses_mujoco_local_tangent(walking):
    rotation = Rotation.from_quat(walking.qpos[:, [4, 5, 6, 3]])
    np.testing.assert_allclose(rotation.apply(walking.qvel[:, 3:6]), walking.body_ang_vel_w[:, 0], atol=1e-12)
    np.testing.assert_array_equal(walking.qvel[:, :3], walking.body_lin_vel_w[:, 0])


def test_exact_official_ten_frame_stride_five_window(walking):
    result = walking.sonic_window(0.2)
    indexes = 10 + np.arange(10) * 5
    np.testing.assert_array_equal(result.qpos, walking.qpos[indexes])
    np.testing.assert_allclose(result.timestamps_s, 0.2 + np.arange(10) * 0.1, atol=1e-12)
    assert result.scene_id == walking.scene_id
    assert result.command_id == walking.command_id


def test_shifted_origin_and_accumulated_runtime_clock(walking):
    shifted = replace(walking, timestamps_s=walking.timestamps_s + 0.18,
                      trusted_until_s=float(walking.timestamps_s[-1] + 0.18))
    runtime_time = 0.18
    for frame in range(410):
        result = shifted.sonic_window(runtime_time)
        indexes = frame + np.arange(10) * 5
        np.testing.assert_array_equal(result.timestamps_s, shifted.timestamps_s[indexes])
        assert result.trusted_until_s == result.timestamps_s[-1]
        runtime_time += 0.02


def test_finite_tail_never_repeats_extrapolates_or_resets(walking):
    assert walking.latest_window_start_s == pytest.approx(8.18)
    result = walking.sonic_window(8.18)
    np.testing.assert_array_equal(result.qpos[-1], walking.qpos[-1])
    for time in (-0.02, 8.2, 9.08):
        with pytest.raises(ReferenceCoverageError):
            walking.sonic_window(time)
    with pytest.raises(ReferenceCoverageError, match="sample grid"):
        walking.sonic_window(0.01)
    with pytest.raises(ValueError, match="finite"):
        walking.sonic_window(float("nan"))


def test_raw_decoder_edges_do_not_expand_trusted_coverage(walking):
    short = replace(walking, trusted_until_s=0.8)
    with pytest.raises(ReferenceCoverageError, match="trusted coverage"):
        short.sonic_window(0.0)
    genuine_longer = replace(walking, trusted_until_s=1.6)
    assert genuine_longer.sonic_window(0.0).timestamps_s[-1] == 0.9


def test_reference_arrays_are_immutable_snapshots(walking):
    with pytest.raises(ValueError):
        walking.qpos[0, 0] = 1
    with pytest.raises(ValueError):
        walking.body_pos_w[0, 0, 0] = 1


def test_executed_g1_history_is_not_native_soma_history():
    history = ExecutedHistory(scene_id="test")
    with pytest.raises(EmbodimentConversionError, match="four SOMA contact"):
        history.native_soma_history()


def test_mirrored_release_has_the_same_explicit_schema():
    mirrored = load_official_reference(WALK.with_name(WALK.name + "_M"), scene_id="test")
    assert mirrored.body_names == OFFICIAL_BODY_NAMES
    assert mirrored.sonic_window(0).qpos.shape == (10, 36)


@pytest.fixture(scope="module")
def source_seed():
    path = ROOT / "motionbricks/out/pfnn_mixed_20261015/final_acceptance_flat_v1/scene.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["initial_history"][0].copy()


def test_real_native_seed_decodes_without_gpu_or_model(source_seed):
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference

    times = (np.arange(4) - 3) / 30
    source = decode_soma_reference(source_seed, times, build_native_representation(),
                                   scene_id="test", command_id="seed", trusted_until_s=0.0,
                                   provenance=dict(kind="source_seed"))
    assert source.world_positions_m.shape == (4, 30, 3)
    np.testing.assert_array_equal(source.raw_features, source_seed)
    np.testing.assert_allclose(source.root_positions_m, source.world_positions_m[:, 0], atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(source.local_rot_mats), 1, atol=2e-5)
    np.testing.assert_allclose(np.linalg.det(source.global_rot_mats), 1, atol=2e-5)


def test_lossless_native_reference_artifact_roundtrip(source_seed, tmp_path):
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference

    source = decode_soma_reference(source_seed, (np.arange(4) - 3) / 30, build_native_representation(),
                                   scene_id="test", command_id="seed", trusted_until_s=0.0,
                                   provenance=dict(kind="source_seed"))
    path = tmp_path / "source.npz"
    save_soma_reference(path, source)
    actual = load_soma_reference(path)
    for name in ("timestamps_s", "raw_features", "local_rot_mats", "global_rot_mats", "world_positions_m"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(source, name))
    assert actual.provenance == source.provenance
    assert actual.command_id == source.command_id
    with pytest.raises(FileExistsError):
        save_soma_reference(path, source)


def test_real_native_geometry_observations_are_audited_without_modification():
    import torch
    from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
    from motionbricks.data.unified_scene_field import field_from_heightfield
    from motionbricks.mb_sonic.mb_reference import audit_legacy_observation
    from motionbricks.mb_sonic.scene import load_scene

    manifest = ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018/scenes/000/scene_manifest.json"
    canonical = load_scene(manifest)
    with np.load(canonical.root / canonical.manifest.source["scene"]["file"], allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(archive[key].copy()) for key in (
            "heightfield_y_m", "min_point_world_xyz_m", "dx_m")}
        history = torch.from_numpy(archive["initial_history"].copy())
    root = history[0, -1, :5]
    observation = extract_local_terrain_observation(
        scene["heightfield_y_m"], scene["min_point_world_xyz_m"], scene["dx_m"],
        root[:3], torch.atan2(root[4], root[3]), root.new_tensor([0., 1.]),
    ).as_model_inputs()
    points = torch.stack((root[:3], root[:3] + root.new_tensor([.07, 0., .13])))[None, None]
    distance, _, valid = field_from_heightfield(scene).query(points)
    before = observation["points"].clone()
    report = audit_legacy_observation(canonical, history, observation,
                                      dict(points=points, distance=distance, valid=valid))
    assert report["same_query_as_E2_claimed"] is False
    assert report["model_input_modified"] is False
    assert report["local_height_quantization"]["count"] > 0
    assert report["legacy_sdf_minus_canonical_vertical_clearance"]["count"] == 2
    torch.testing.assert_close(before, observation["points"], rtol=0, atol=0)


def test_native_history_requires_explicit_embodiment_provenance_and_30hz(source_seed):
    times = (np.arange(4) - 3) / 30
    history = NativeSomaHistory(times, source_seed, "test", dict(kind="calibrated_executed_g1"))
    np.testing.assert_array_equal(history.features, source_seed)
    with pytest.raises(ValueError, match="distinguish"):
        NativeSomaHistory(times, source_seed, "test", dict(kind="reference_is_actual"))
    with pytest.raises(ValueError, match="30 Hz"):
        NativeSomaHistory(np.arange(4) / 50, source_seed, "test", dict(kind="source_seed"))


def test_trained_compact_parent_hashes_and_completed_stages():
    from motionbricks.mb_sonic.mb_reference import inspect_trained_artifacts

    base = ROOT / "motionbricks/out"
    result = inspect_trained_artifacts(base / "pfnn_mixed_20261015/final_acceptance_flat_v1",
                                       base / "a100_live_20261010/a100-terrain.pt")
    assert result["model_kind"] == "trained_legacy_full_corpus"
    assert result["mechanics_only"] is False
    assert result["current_optimizer_steps"] == 0
    assert [stage["stage"] for stage in result["training_stages"]] == ["root", "codec", "pose"]
    assert all(stage["actual_exposures"] == 12800000 for stage in result["training_stages"])


def test_untrained_probe_receipt_is_rejected_before_model_loading(tmp_path):
    from motionbricks.mb_sonic.mb_reference import inspect_trained_artifacts

    receipt_path = ROOT / "motionbricks/out/pfnn_mixed_20261015/final_acceptance_flat_v1/acceptance.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["mechanics_only"] = True
    (tmp_path / "acceptance.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="untrained v3"):
        inspect_trained_artifacts(tmp_path, Path("unused-bundle"))


@pytest.mark.parametrize("world_yaw", [0., np.pi / 2, np.pi, -np.pi / 2])
def test_world_command_heading_uses_canonical_transform(world_yaw):
    from motionbricks.mb_sonic.mb_reference import world_xy_command_to_soma
    from motionbricks.mb_sonic.scene import soma_to_mujoco

    world_forward = np.array([np.cos(world_yaw), np.sin(world_yaw)])
    velocity, heading = world_xy_command_to_soma(world_forward * .5, world_yaw)
    actual_world_velocity = soma_to_mujoco([velocity[0], 0., velocity[1]])
    actual_world_forward = soma_to_mujoco([np.sin(heading), 0., np.cos(heading)])
    np.testing.assert_allclose(actual_world_velocity[:2], world_forward * .5, atol=1e-12)
    np.testing.assert_allclose(actual_world_forward[:2], world_forward, atol=1e-12)


def test_native_and_dual_global_inverse_have_identical_semantics(source_seed):
    import torch
    from motionbricks.data.native_mb_data import build_native_representation

    native = build_native_representation()
    global_representation = native.dual_rep.global_motion_rep
    values = torch.tensor(source_seed)[None]
    left = native.inverse(values, False, return_all=False)
    right = global_representation.inverse(values, False, return_all=False)
    assert native.root_mode == global_representation.root_mode == "global"
    for key in ("root_pos", "posed_joints", "local_joint_rots", "global_joint_rots"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_scene_registration_moves_root_and_body_together_without_rewriting_motion(source_seed):
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference,source_in_scene_frame
    source=decode_soma_reference(source_seed,(np.arange(4)-3)/30,build_native_representation(),
        scene_id='scene',command_id='frame',trusted_until_s=0.,provenance={})
    shift=np.array([8.,-.2,-6.])
    mapped=source_in_scene_frame(source,shift)
    np.testing.assert_allclose(mapped.root_positions_m-source.root_positions_m, np.tile(shift,(4,1)),atol=1e-14)
    np.testing.assert_allclose(mapped.world_positions_m-mapped.root_positions_m[:,None],
        source.world_positions_m.astype(float)-source.root_positions_m.astype(float)[:,None],atol=1e-14,rtol=0)
    np.testing.assert_array_equal(mapped.global_rot_mats,source.global_rot_mats)
    np.testing.assert_array_equal(mapped.raw_features[:,3:],source.raw_features[:,3:])
    np.testing.assert_array_equal(source.raw_features,source_seed)


def test_existing_full_precision_policy_restores_previous_settings():
    import torch
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    before = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
    initialized = torch.cuda.is_initialized()
    with full_precision_evaluation():
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32) == before
    assert torch.cuda.is_initialized() == initialized


def test_precision_receipt_reads_the_original_actor_contract_without_cuda_initialization():
    import torch
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation
    from motionbricks.mb_sonic.mb_reference import native_precision_contract

    initialized = torch.cuda.is_initialized()
    with full_precision_evaluation():
        assert native_precision_contract() == dict(
            matmul="highest", cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False)
    assert torch.cuda.is_initialized() == initialized


def test_host_receipt_reports_actual_torch_thread_settings():
    import torch
    from motionbricks.mb_sonic.mb_reference import native_host_contract

    initialized = torch.cuda.is_initialized()
    assert native_host_contract() == dict(torch_intraop_threads=torch.get_num_threads(),
                                          torch_interop_threads=torch.get_num_interop_threads())
    assert torch.cuda.is_initialized() == initialized


def test_cpu_reference_decode_profiles_measured_transfer_fk_and_snapshot(source_seed):
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference

    timings = {}
    source = decode_soma_reference(source_seed, (np.arange(4) - 3) / 30, build_native_representation(),
                                   scene_id="test", command_id="seed", trusted_until_s=0.,
                                   provenance=dict(kind="source_seed"), timings=timings)
    np.testing.assert_array_equal(source.raw_features, source_seed)
    assert set(timings) == {"raw_cpu_transfer_including_cuda_wait_wall_ns", "native_fk_dispatch_wall_ns",
                            "pose_cpu_transfer_including_cuda_wait_wall_ns", "immutable_snapshot_wall_ns"}
    assert all(type(value) is int and value > 0 for value in timings.values())


@pytest.fixture(scope="module")
def generated_proposal():
    return load_soma_reference(ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018/mb_proposal_compiled_v2/proposal.npz")


@pytest.fixture(scope="module")
def b1_cpu_observer(source_seed):
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import prepare_reference_history_b1

    representation = build_native_representation()
    return representation, prepare_reference_history_b1(representation, source_seed)


def test_b1_native_grid_history_reuses_official_reencoding_without_changing_source(generated_proposal, b1_cpu_observer):
    import torch
    from motionbricks.data.native_mb_observed_history import reencode_played_history
    from motionbricks.mb_sonic.mb_reference import reference_history_b1

    representation, observer = b1_cpu_observer
    before = generated_proposal.raw_features.copy()
    result = reference_history_b1(generated_proposal, end_time_s=.2, representation=representation, observer=observer)
    expected = reencode_played_history(torch.tensor(before[6:10])[None], representation)
    np.testing.assert_array_equal(result.features, expected[0].numpy())
    np.testing.assert_array_equal(generated_proposal.raw_features, before)
    assert result.provenance["kind"] == "legacy_reference_history"
    assert result.provenance["reference_root_reanchored"] is False
    assert result.provenance["contacts"] == "reference_predictions_not_measured_forces"
    assert result.provenance["history_policy"] == "native_played_fk_reencode"
    assert result.provenance["reencode_position_error_m"] <= 2e-5
    assert result.provenance["reencode_rotation_error"] <= 2e-5


def test_b1_fractional_clock_history_uses_reference_root_and_so3(generated_proposal, b1_cpu_observer):
    import torch
    from motionbricks.mb_sonic.mb_reference import reference_history_b1
    from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix

    representation, observer = b1_cpu_observer
    result = reference_history_b1(generated_proposal, end_time_s=.18, representation=representation, observer=observer)
    for axis in range(3):
        expected = np.interp(result.timestamps_s, generated_proposal.timestamps_s,
                             generated_proposal.raw_features[:, axis])
        np.testing.assert_allclose(result.features[:, axis], expected, atol=1e-7)
    rotation = cont6d_to_matrix(torch.tensor(result.features[:, representation.indices["global_rot_data"]]).reshape(4, 30, 6))
    np.testing.assert_allclose(np.linalg.det(rotation.numpy()), 1, atol=2e-6)
    assert all(0 <= weight <= 1 for weight in result.provenance["interpolation_weights"])


@pytest.mark.parametrize("end_time", [-.02, 1.61, float("nan")])
def test_b1_reference_history_never_extrapolates(generated_proposal, end_time, b1_cpu_observer):
    from motionbricks.mb_sonic.mb_reference import reference_history_b1

    with pytest.raises(ReferenceCoverageError):
        reference_history_b1(generated_proposal, end_time_s=end_time,
                             representation=b1_cpu_observer[0], observer=b1_cpu_observer[1])


def test_b1_provenance_requires_declared_mode_and_query_origin(source_seed):
    from motionbricks.mb_sonic.mb_reference import validate_conditioning

    times = (np.arange(4) - 3) / 30
    reference = NativeSomaHistory(times, source_seed, "test", dict(kind="legacy_reference_history"))
    with pytest.raises(ValueError, match="explicit B1"):
        validate_conditioning(reference, "native", None)
    with pytest.raises(ValueError, match="query origin"):
        validate_conditioning(reference, "B1", None)
    np.testing.assert_array_equal(validate_conditioning(reference, "B1", [.1, 1., -.2]), [.1, 1., -.2])
    internal = NativeSomaHistory(times, source_seed, "test", dict(kind="open_loop_generated_context"))
    with pytest.raises(ValueError, match="Internal autoregressive"):
        validate_conditioning(internal, "B1", [0., 1., 0.])


@pytest.mark.parametrize("scene_id", ["000", "001"])
def test_actual_query_samples_keep_local_geometry_and_register_world_points(scene_id):
    import torch
    from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
    from motionbricks.data.unified_scene_field import field_from_heightfield, VolumetricEgoSampler
    from motionbricks.mb_sonic.mb_reference import reframe_scene_inputs

    from motionbricks.mb_sonic.scene import load_scene

    canonical = load_scene(ROOT / f"motionbricks/out/mb_sonic_e0_e3_20261018/scenes/{scene_id}/scene_manifest.json")
    source_path = canonical.root / canonical.manifest.source["scene"]["file"]
    with np.load(source_path, allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(archive[key].copy()) for key in (
            "heightfield_y_m", "min_point_world_xyz_m", "dx_m")}
        history = torch.from_numpy(archive["initial_history"].copy())
    before = history.clone()
    root = history[..., :5]
    query_root = root.clone()
    query_root[..., :3] += root.new_tensor([.3, -.2103, .2])
    observation = extract_local_terrain_observation(
        scene["heightfield_y_m"], scene["min_point_world_xyz_m"], scene["dx_m"],
        query_root[:, -1, :3], torch.atan2(root[:, -1, 4], root[:, -1, 3]), root.new_tensor([[0., 1.]]))
    envelope = observation.points[..., [0, 2]].norm(dim=-1).flatten(1).amax(1) + 1e-5
    local = (observation.points, observation.valid, observation.world_query_xz, envelope)
    sampler = VolumetricEgoSampler()
    volume = sampler.observe(query_root, root.new_tensor([[0., .5]]), field_from_heightfield(scene), 28 / 30)
    keys = ("points", "distance", "normal", "valid", "features")
    values = (*local, *local, *(volume[key] for key in keys))
    result = reframe_scene_inputs(root, query_root, values, sampler)
    torch.testing.assert_close(history, before, rtol=0, atol=0)
    delta=query_root[:,-1,:3]-root[:,-1,:3]
    torch.testing.assert_close(result[0], observation.points, rtol=0, atol=0)
    torch.testing.assert_close(result[12], volume['features'], rtol=0, atol=0)
    torch.testing.assert_close(result[2]+delta[:,[0,2]][:,None,None,None], observation.world_query_xz, rtol=0, atol=2e-6)
    torch.testing.assert_close(result[8]+delta[:,None,None], volume["points"], rtol=0, atol=2e-6)
    torch.testing.assert_close(result[11], volume["valid"], rtol=0, atol=0)
    torch.testing.assert_close(result[8][0, 0].mean(0), root[0, -1, :3], atol=2e-6, rtol=0)
    known = result[1]
    native_height = observation.points[..., 1] + query_root[:, -1, 1, None, None, None]
    model_height = result[0][..., 1] + root[:, -1, 1, None, None, None]+delta[:,1,None,None,None]
    torch.testing.assert_close(model_height[known], native_height[known], rtol=0, atol=2e-7)
    relative = result[8] - root[:, -1, None, None, :3]
    torch.testing.assert_close(result[12][..., :3][result[11]], torch.asinh(relative)[result[11]], atol=2e-6, rtol=0)


def test_c_provider_contract_keywords_and_reference_builder_are_importable():
    import inspect
    from motionbricks.mb_sonic.mb_reference import TrainedMBReferenceAdapter, reference_history_b1

    assert callable(reference_history_b1)
    for method in (TrainedMBReferenceAdapter.generate, TrainedMBReferenceAdapter.prepare_compiled):
        parameters = inspect.signature(method).parameters
        assert "conditioning_mode" in parameters and "query_origin_soma_xyz" in parameters
        assert "actual_root_mujoco_xyz" not in parameters


def test_native_dtype_payload_is_bitwise_lossless_for_mixed_outputs(generated_proposal):
    import torch
    from motionbricks.mb_sonic.mb_reference import payload_layout, pack_payload, unpack_payload_numpy, unpack_payload_torch

    fields = dict(raw=torch.tensor(generated_proposal.raw_features[:32])[None],
                  mask=torch.tensor(generated_proposal.raw_features[:32, -4:] > .5),
                  ids=torch.tensor([-(2 ** 63), 2 ** 63 - 1, 0, 2047], dtype=torch.int64),
                  times=torch.tensor(generated_proposal.timestamps_s), scalar=torch.tensor(2e-7))
    layout = payload_layout(fields)
    payload = pack_payload(fields, layout)
    assert payload.dtype == torch.uint8
    numpy_values = unpack_payload_numpy(payload.numpy(), layout)
    torch_values = unpack_payload_torch(payload, layout)
    for name, original in fields.items():
        torch.testing.assert_close(torch_values[name], original, rtol=0, atol=0)
        assert numpy_values[name].tobytes() == original.numpy().tobytes()


def test_raw_global_block_retains_declared_input_without_eager_fk(tmp_path):
    from motionbricks.mb_sonic.reference import RawMBBlock, save_raw_mb_block, load_raw_mb_block

    original = load_soma_reference(ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018/mb_proposal_compiled_v2/raw_block_00.npz")
    with np.load(ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018/scenes/000/source_scene.npz", allow_pickle=False) as archive:
        input_history = archive["initial_history"][0].copy()
    block = RawMBBlock(original.timestamps_s, original.raw_features, input_history,
                       original.scene_id, original.command_id, original.trusted_until_s,
                       dict(pose_expansion="offline_only", history_kind="source_seed"))
    path = tmp_path / "raw_global366.npz"
    save_raw_mb_block(path, block)
    actual = load_raw_mb_block(path)
    np.testing.assert_array_equal(actual.raw_features, original.raw_features)
    np.testing.assert_array_equal(actual.history_features, block.history_features)
    assert not hasattr(actual, "world_positions_m")
    with pytest.raises(FileExistsError):
        save_raw_mb_block(path, block)


def test_frozen_script_children_use_python_orchestration_not_outer_torch_trace():
    import ast
    import inspect
    import textwrap
    import torch
    from motionbricks.data.soma_static_execution import trace_fixed_call
    from motionbricks.mb_sonic.mb_reference import FullPlanExecution

    module = torch.nn.Linear(3, 2).eval().requires_grad_(False)
    sample = torch.tensor([[.1, .2, .3]])
    frozen = torch.jit.freeze(torch.jit.trace(module, sample))
    outer = lambda value: frozen(value) + value[:, :2]
    torch.testing.assert_close(outer(sample), frozen(sample) + sample[:, :2], atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="active trace|registered"):
        trace_fixed_call(outer, (sample,))
    tree = ast.parse(textwrap.dedent(inspect.getsource(FullPlanExecution.prepare)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(node.func, ast.Name) and node.func.id == "StaticCudaCall"
               and node.args and isinstance(node.args[0], ast.Attribute)
               and node.args[0].attr == "_captured_plan" for node in calls)
    assert not any(isinstance(node.func, ast.Name) and node.func.id == "trace_fixed_call"
                   and node.args and isinstance(node.args[0], ast.Attribute)
                   and node.args[0].attr == "_captured_plan" for node in calls)


def test_rolling_actual_anchor_uses_reference_displacement_without_root_changes(generated_proposal):
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.native_mb_observed_history import reencode_played_history
    from motionbricks.mb_sonic.mb_reference import rolling_query_origin

    first = torch.tensor(generated_proposal.raw_features[:4])[None]
    current = reencode_played_history(torch.tensor(generated_proposal.raw_features[24:28])[None],
                                      build_native_representation())
    before = current.clone()
    anchor = first[:, -1, :3] + first.new_tensor([[.3, -.2103, .2]])
    actual = rolling_query_origin(first, current, anchor)
    torch.testing.assert_close(actual, anchor + (current[:, -1, :3] - first[:, -1, :3]), atol=1e-7, rtol=0)
    torch.testing.assert_close(rolling_query_origin(first, current, first[:, -1, :3]),
                               current[:, -1, :3], atol=0, rtol=0)
    torch.testing.assert_close(current, before, rtol=0, atol=0)


@pytest.fixture
def generation_cli_arguments():
    base = ROOT / "motionbricks/out"
    return ["--scene-id", "000", "--generate", "--execution", "full_plan", "--device", "cuda",
            "--artifact", str(base / "pfnn_mixed_20261015/final_acceptance_flat_v1"),
            "--bundle", str(base / "a100_live_20261010/a100-terrain.pt"),
            "--seed", str(base / "mb_sonic_e0_e3_20261018/scenes/000/source_scene.npz"),
            "--manifest", str(base / "mb_sonic_e0_e3_20261018/scenes/000/scene_manifest.json"),
            "--output", "unused_cli_parse_only"]


def test_cli_b1_requires_and_preserves_explicit_native_query_origin(generation_cli_arguments):
    import torch
    from scripts.inspect_mb_sonic_reference import parse_arguments

    initialized = torch.cuda.is_initialized()
    native = parse_arguments(generation_cli_arguments)
    assert native.conditioning_mode == "native" and native.query_origin_soma_xyz is None
    b1 = parse_arguments(generation_cli_arguments + ["--conditioning-mode", "B1",
                                                    "--query-origin-soma-xyz", ".3", "1.02", "-.2"])
    assert b1.conditioning_mode == "B1" and b1.execution == "full_plan"
    assert b1.query_origin_soma_xyz == [.3, 1.02, -.2]
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("options", [
    ["--conditioning-mode", "B1"],
    ["--query-origin-soma-xyz", ".3", "1.", "-.2"],
    ["--conditioning-mode", "B1", "--query-origin-soma-xyz", "nan", "1.", "-.2"],
    ["--conditioning-mode", "B1", "--query-origin-soma-xyz", "inf", "1.", "-.2"],
])
def test_cli_rejects_missing_mode_origin_or_finite_contract(generation_cli_arguments, options):
    from scripts.inspect_mb_sonic_reference import parse_arguments

    with pytest.raises(SystemExit) as error:
        parse_arguments(generation_cli_arguments + options)
    assert error.value.code == 2


@pytest.fixture(scope="module")
def recorded_b1_sources():
    from motionbricks.mb_sonic.reference import SomaReference

    base = ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018"
    sources = [load_soma_reference(base / name / "proposal.npz") for name in (
        "mb_proposal_v1", "mb_proposal_compiled_v2", "mb_proposal_full_plan_v1")]
    for experiment in ("e3_b1_fullplan_v1", "e3_b1_fullplan_v2"):
        paths = sorted((base / experiment / "episode_0001_numeric").glob("*.npz"))
        assert paths, experiment
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"]))
                assert metadata["schema"] == "motionbricks.mb_sonic.numeric_record.v1"

                def decode(value):
                    if isinstance(value, dict):
                        if "array" in value:
                            return archive[value["array"]].copy()
                        return {key: decode(child) for key, child in value.items()}
                    if isinstance(value, list):
                        return [decode(child) for child in value]
                    return value

                sources.append(SomaReference(**decode(metadata["record"]["raw_source"])))
    assert len(sources) == 9
    return sources


@pytest.mark.parametrize("future_offset_s", [0., .02, .04, .18, .22, .37, .8, 1.5, 1.6])
def test_prepared_b1_all366_contacts_and_source_identity_on_actual_npzs(
        recorded_b1_sources, b1_cpu_observer, future_offset_s):
    import hashlib
    from motionbricks.mb_sonic.mb_reference import reference_history_b1, reference_history_b1_oracle

    representation, observer = b1_cpu_observer
    for source in recorded_b1_sources:
        before = hashlib.sha256(source.raw_features.tobytes() + source.global_rot_mats.tobytes()).digest()
        end = float(source.timestamps_s[3] + future_offset_s)
        actual = reference_history_b1(source, end_time_s=end, representation=representation, observer=observer)
        expected = reference_history_b1_oracle(source, end_time_s=end, representation=representation)
        np.testing.assert_array_equal(actual.features, expected.features)
        np.testing.assert_array_equal(actual.timestamps_s, expected.timestamps_s)
        contacts = representation.indices["foot_contacts"]
        np.testing.assert_array_equal(actual.features[:, contacts], expected.features[:, contacts])
        assert actual.provenance["source_brackets"] == expected.provenance["source_brackets"]
        assert actual.provenance["interpolation_weights"] == expected.provenance["interpolation_weights"]
        assert actual.provenance["rotation_keyframe_count"] <= 5
        assert actual.provenance["reencode_position_error_m"] <= 2e-5
        assert actual.provenance["reencode_rotation_error"] <= 2e-5
        assert before == hashlib.sha256(source.raw_features.tobytes() + source.global_rot_mats.tobytes()).digest()


@pytest.mark.parametrize("heading", [-3.13, -np.pi / 2, 0., np.pi / 2, 3.13])
def test_prepared_b1_native_coordinate_heading_cases_have_identical_all366_and_fk(
        generated_proposal, b1_cpu_observer, heading):
    import torch
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference, reference_history_b1, reference_history_b1_oracle

    representation, observer = b1_cpu_observer
    # Coordinate-frame tests use the existing native transform of the saved real proposal.
    turned = representation.change_first_heading(torch.tensor(generated_proposal.raw_features)[None],
                                                 heading, False, False)
    source = decode_soma_reference(turned, generated_proposal.timestamps_s, representation,
                                   scene_id=generated_proposal.scene_id, command_id="native_coordinate_frame_test",
                                   trusted_until_s=generated_proposal.trusted_until_s,
                                   provenance=dict(kind="native_coordinate_frame_test", heading=heading))
    for end in (.0, .02, .18, .22, .37, .8, 1.5, 1.6):
        actual = reference_history_b1(source, end_time_s=end, representation=representation, observer=observer)
        expected = reference_history_b1_oracle(source, end_time_s=end, representation=representation)
        np.testing.assert_array_equal(actual.features, expected.features)
        left = representation.inverse(torch.tensor(actual.features)[None], False, return_all=False)
        right = representation.inverse(torch.tensor(expected.features)[None], False, return_all=False)
        for name in ("root_pos", "posed_joints", "global_joint_rots", "local_joint_rots"):
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)


def test_cpu_b1_preparation_and_calls_preserve_rng_assets_and_explicit_execution(source_seed, generated_proposal):
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.reference import file_sha256
    from motionbricks.mb_sonic.mb_reference import (
        prepare_reference_history_b1, reference_history_b1, TRAINED_WEIGHTS_SHA256, TRAINED_BUNDLE_SHA256,
    )

    rng, cuda_initialized = torch.random.get_rng_state().clone(), torch.cuda.is_initialized()
    representation = build_native_representation()
    seed_before = source_seed.copy()
    observer = prepare_reference_history_b1(representation, source_seed)
    np.testing.assert_array_equal(source_seed, seed_before)
    json.dumps(observer.preparation, allow_nan=False)
    for end in (.02, .18, .37, .8, 1.6):
        reference_history_b1(generated_proposal, end_time_s=end, representation=representation, observer=observer)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == cuda_initialized
    assert observer.preparation["cpu_rng_state_equal"]
    assert observer.preparation["initial_all_channels_equal"]
    with pytest.raises(TypeError, match="observer"):
        reference_history_b1(generated_proposal, end_time_s=.18, representation=representation)
    with pytest.raises(ValueError, match="prepared native representation"):
        reference_history_b1(generated_proposal, end_time_s=.18,
                             representation=build_native_representation(), observer=observer)
    with pytest.raises(ValueError, match="FP32"):
        observer(torch.tensor(source_seed, dtype=torch.float64)[None])
    invalid = torch.tensor(source_seed)[None]
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="displayed FK"):
        observer(invalid)
    base = ROOT / "motionbricks/out"
    assert file_sha256(base / "pfnn_mixed_20261015/final_acceptance_flat_v1/weights.pt") == TRAINED_WEIGHTS_SHA256
    assert file_sha256(base / "a100_live_20261010/a100-terrain.pt") == TRAINED_BUNDLE_SHA256


def test_b1_oracle_body_matches_the_preserved_actual_v1_source():
    import ast
    import inspect
    from motionbricks.mb_sonic.mb_reference import reference_history_b1_oracle

    frozen = ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v1/sources/mb_reference.py"
    tree = ast.parse(frozen.read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reference_history_b1")
    oracle = ast.parse(inspect.getsource(reference_history_b1_oracle)).body[0]
    assert [ast.dump(node) for node in original.body[1:]] == [ast.dump(node) for node in oracle.body[1:]]
