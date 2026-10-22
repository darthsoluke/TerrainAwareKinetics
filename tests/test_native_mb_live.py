"""Native output ownership and 30 Hz committed protocol regressions."""

from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.native_mb_live import NativeSomaLiveBackend, native_locomotion_contract
from scripts.soma_ue_live_server import COMMITTED_SCHEMA, SCHEMA, SomaLiveSession


def curriculum_bundle(step):
    args = {"max_num_start_keyframes": 4, "max_num_end_keyframes": 4,
            "keyframe_num_warmup_steps": 200000,
            "no_start_keyframe_prob": .2, "no_end_keyframe_prob": .2}
    return {"metadata": {"root_step": step, "pose_step": step},
            "root_config": {"model": {"args": args}}, "pose_config": {"model": {"args": args}}}


def test_live_conditions_use_checkpoint_curriculum_not_configured_four_frame_maxima():
    contract = native_locomotion_contract(curriculum_bundle(50000))
    assert contract["trained_keyframe_counts"] == {"root": {"start": [1], "end": [1]},
                                                   "pose": {"start": [1], "end": [1]}}
    assert contract["context_keyframes"] == contract["target_keyframes"] == 1
    assert contract["skip_ending_target_cond"] is False
    assert contract["output_root"] == "native decoder refinement, no replacement"


def test_full_curriculum_still_uses_paper_sparse_navigation():
    contract = native_locomotion_contract(curriculum_bundle(200000))
    assert contract["trained_keyframe_counts"]["root"]["start"] == [1, 2, 3, 4]
    assert contract["target_keyframes"] == 1


def test_real_backend_initialization_packs_sparse_conditions_without_output_replacement():
    from test_native_mb_locomotion import RecordingInference, _clip
    from test_native_mb_stage_training import _motion_and_representation

    rep, _ = _motion_and_representation(frame_count=17)
    infer = RecordingInference(rep)
    bundle = curriculum_bundle(50000)
    bundle.update(initial_history=_clip(rep, frames=4)[None],
                  reference_clips={"idle": _clip(rep), "walk": _clip(rep, velocity=(0, 1.2)),
                                   "run": _clip(rep, velocity=(0, 2.5))})
    backend = NativeSomaLiveBackend(infer, bundle)
    result = backend.predict_plan([[0, 1.2]], [0])
    call = infer.calls[-1]
    assert call["has_global_root_values"].nonzero()[:, 1].tolist() == [3, 7]
    assert call["has_local_root_values"].nonzero()[:, 1].tolist() == [2, 7]
    assert call["has_local_poses"].nonzero()[:, 1].tolist() == [3, 7]
    assert call["config"]["skip_ending_target_cond"] is False
    torch.testing.assert_close(result["planned_motion"], infer.prediction[:, 4:12], rtol=0, atol=0)


def test_buffered_initialization_keeps_native_sparse_conditions_and_bounded_duration():
    from test_native_mb_locomotion import RecordingInference, _clip
    from test_native_mb_stage_training import _motion_and_representation

    rep, _ = _motion_and_representation(frame_count=17)
    infer = RecordingInference(rep)
    infer._args = {"min_tokens": 2, "max_tokens": 16}
    bundle = curriculum_bundle(50000)
    bundle.update(initial_history=_clip(rep, frames=4)[None],
                  reference_clips={"idle": _clip(rep), "walk": _clip(rep, velocity=(0, 1.2))})
    backend = NativeSomaLiveBackend(infer, bundle, reuse_motion_buffer=True)
    result = backend.predict_plan([[0, 1.2]], [0])
    call = infer.calls[-1]
    assert call["has_local_poses"].nonzero()[:, 1].tolist() == [3, 7]
    assert call["allowed_pred_num_tokens"].tolist() == [[0, 0] + [1] * 8 + [0] * 5]
    assert backend.condition_contract["drop_end_frames"] == 4
    assert call["config"]["smooth_root_traj"] is False
    torch.testing.assert_close(result["planned_motion"], infer.prediction[:, 4:12], rtol=0, atol=0)


@pytest.mark.parametrize("step", [0, True, "50000"])
def test_missing_actual_checkpoint_progress_is_rejected(step):
    with pytest.raises(ValueError, match="update count"):
        native_locomotion_contract(curriculum_bundle(step))


@pytest.mark.parametrize("velocity,style", [([0, 0], "idle"), ([0, 1.2], "walk"), ([0, 2.5], "run"), ([0, -.6], "walk")])
def test_plan_skips_decoded_context_and_preserves_every_future_value(velocity, style):
    backend = object.__new__(NativeSomaLiveBackend)
    backend.history = torch.ones(1, 4, 366)
    prediction = torch.arange(32 * 366).reshape(1, 32, 366).float()
    calls = []

    def predict(history, command, heading, **kwargs):
        calls.append((history.clone(), command, heading, kwargs))
        return prediction, torch.tensor([8])

    backend.controller = SimpleNamespace(predict=predict)
    previous = backend.history.clone()
    result = backend.predict_plan([velocity], [0])
    torch.testing.assert_close(result["planned_motion"], prediction[:, 4:12], rtol=0, atol=0)
    torch.testing.assert_close(backend.history, previous, rtol=0, atol=0)
    assert calls[0][3] == {"style": style, "random_seed": 2333}
    assert backend.last_style == style


def test_short_prediction_rejected():
    backend = object.__new__(NativeSomaLiveBackend)
    backend.history = torch.zeros(1, 4, 366)
    backend.controller = SimpleNamespace(predict=lambda *args, **kwargs: (torch.zeros(1, 8, 366), torch.tensor([2])))
    with pytest.raises(ValueError, match="eight-frame future"):
        backend.predict_plan([[0, 1]], [0])


def buffered_backend(frames=32):
    backend = object.__new__(NativeSomaLiveBackend)
    backend.history = torch.ones(1, 4, 366)
    backend.reuse_motion_buffer = True
    backend._future_buffer = backend._last_published_history = backend._last_command = None
    prediction = torch.arange(frames * 366).reshape(1, frames, 366).float() / 1000
    calls = []

    def predict(*args, **kwargs):
        calls.append(args[0].clone())
        return prediction.clone(), torch.tensor([frames // 4])

    backend.controller = SimpleNamespace(predict=predict)
    return backend, prediction, calls


def test_buffer_samples_consecutive_raw_frames_and_discards_decoder_tail():
    backend, prediction, calls = buffered_backend()
    previous = backend.history.clone()
    outputs = []
    for start in (4, 12, 20):
        result = backend.predict_plan([[0, 1.2]], [0])
        outputs.append(result["planned_motion"])
        torch.testing.assert_close(result["planned_motion"], prediction[:, start:start + 8], rtol=0, atol=0)
        torch.testing.assert_close(backend.history, previous, rtol=0, atol=0)
        backend.history = result["planned_motion"][:, -4:].clone()
        previous = backend.history.clone()
    assert len(calls) == 1
    assert backend.last_generation == {"neural_generation": False, "reason": "buffer_sample", "remaining_frames": 0}
    assert not torch.equal(torch.cat(outputs, dim=1)[:, -4:], prediction[:, -4:])
    backend.predict_plan([[0, 1.2]], [0])
    assert len(calls) == 2 and backend.last_generation["reason"] == "buffer_low"


@pytest.mark.parametrize("command,heading", [([0, 0], 0), ([0, 2.5], 0), ([0, 1.2], .4)])
def test_command_change_flushes_unpublished_buffer(command, heading):
    backend, prediction, calls = buffered_backend()
    result = backend.predict_plan([[0, 1.2]], [0])
    backend.history = result["planned_motion"][:, -4:].clone()
    changed = backend.predict_plan([command], [heading])
    assert len(calls) == 2 and backend.last_generation["reason"] == "command_changed"
    torch.testing.assert_close(changed["planned_motion"], prediction[:, 4:12], rtol=0, atol=0)


def test_constant_collision_coordinate_shift_translates_only_buffered_root_xz():
    backend, prediction, calls = buffered_backend()
    result = backend.predict_plan([[0, 1.2]], [0])
    backend.history = result["planned_motion"][:, -4:].clone()
    offset = torch.tensor([.25, 0, -.1])
    backend.history[..., :3] += offset
    expected = prediction[:, 12:20].clone()
    expected[..., :3] += offset
    buffered = backend.predict_plan([[0, 1.2]], [0])
    torch.testing.assert_close(buffered["planned_motion"], expected, rtol=0, atol=0)
    assert len(calls) == 1
    backend.history = buffered["planned_motion"][:, -4:].clone()
    expected = prediction[:, 20:28].clone()
    expected[..., :3] += offset
    # Recovered float32 translation can differ by one ULP; body values stay exact.
    actual = backend.predict_plan([[0, 1.2]], [0])["planned_motion"]
    torch.testing.assert_close(actual[..., :3], expected[..., :3], rtol=0, atol=2e-6)
    torch.testing.assert_close(actual[..., 3:], expected[..., 3:], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["body", "height", "nonuniform_root"])
def test_changed_history_cannot_reuse_incompatible_motion(change):
    backend, _, calls = buffered_backend()
    result = backend.predict_plan([[0, 1.2]], [0])
    backend.history = result["planned_motion"][:, -4:].clone()
    if change == "body":
        backend.history[..., 10] += .01
    elif change == "height":
        backend.history[..., 1] += .01
    else:
        backend.history[:, 0, 0] += .01
    backend.predict_plan([[0, 1.2]], [0])
    assert len(calls) == 2 and backend.last_generation["reason"] == "context_changed"


def test_published_motion_cannot_mutate_cached_future_or_history_snapshot():
    backend, prediction, calls = buffered_backend()
    result = backend.predict_plan([[0, 1.2]], [0])
    backend.history = result["planned_motion"][:, -4:].clone()
    with torch.inference_mode():
        result["planned_motion"].fill_(-999)
    buffered = backend.predict_plan([[0, 1.2]], [0])
    torch.testing.assert_close(buffered["planned_motion"], prediction[:, 12:20], rtol=0, atol=0)
    assert len(calls) == 1


def test_buffered_horizon_requires_room_for_context_publish_and_dropped_tail():
    backend, _, _ = buffered_backend(frames=12)
    with pytest.raises(ValueError, match="context/publish/drop"):
        backend.predict_plan([[0, 1.2]], [0])


def test_cached_path_rejects_nonfinite_heading():
    backend, _, calls = buffered_backend()
    result = backend.predict_plan([[0, 1.2]], [0])
    backend.history = result["planned_motion"][:, -4:].clone()
    with pytest.raises(ValueError, match="finite planar command"):
        backend.predict_plan([[0, 1.2]], [float("nan")])
    assert len(calls) == 1


@pytest.mark.parametrize("fps", [30, 60])
def test_explicit_native_clock_and_consecutive_committed_ticks(fps):
    motion = torch.ones(1, 8, 366)
    motion[0, :, 2] = torch.arange(1, 9) / fps
    backend = SimpleNamespace(history=torch.ones(1, 4, 366))
    backend.predict_plan = lambda *args: {"planned_motion": motion, "generation_seconds": .01}

    def inverse(value, *args, **kwargs):
        return {"posed_joints": value[..., :3].unsqueeze(-2),
                "global_joint_rots": torch.ones(1, value.shape[1], 1, 4) / 2,
                "foot_contacts": torch.ones(1, value.shape[1], 4)}

    infer = SimpleNamespace(_root_model=torch.nn.Linear(1, 1), motion_rep=SimpleNamespace(fps=fps, inverse=inverse))
    live = SomaLiveSession(infer, {}, "model", {"source_id": "soma", "joints": [{"name": "Hips", "parent_index": -1}]},
                           backend=backend)
    request = {"schema": COMMITTED_SCHEMA, "model_id": "model", "source_id": "soma", "session_id": "trial",
               "request_id": 1, "velocity_ue_mps": [1, 0, 0], "facing_ue": [1, 0, 0], "client_time_s": 1.,
               "history_kind": "initial", "history_tick": -1, "executed_tick": -1,
               "history": [], "executed_history": [], "publish_frames": 8}
    first = live.respond(request)
    assert first["fps"] == fps and first["start_tick"] == 0 and len(first["frames"]) == 8
    request.update(request_id=2, history_kind="committed", history_tick=7, executed_tick=3,
                   history=[{"sequence": 1, "frame": i, "root_m": motion[0, i, :3].tolist()} for i in range(4, 8)],
                   executed_history=[{"sequence": 1, "frame": i, "root_m": motion[0, i, :3].tolist()} for i in range(4)])
    second = live.respond(request)
    assert second["start_tick"] == 8 and second["sequence"] == 2 and second["fps"] == fps
    assert second["post_processing"] is None
    if fps == 30:
        request.update(schema=SCHEMA, request_id=3)
        with pytest.raises(ValueError, match="committed playback"):
            live.respond(request)
