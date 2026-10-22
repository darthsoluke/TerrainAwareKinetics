from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_stair_live import NativeStairLiveBackend


class Primitive:
    def __init__(self):
        self.calls = []
        self.outputs = []

    def predict(self, history, goal):
        self.calls.append((history.clone(), goal.clone()))
        result = history[:, -1:].repeat(1, 16, 1)
        result[0, :, 0] = torch.arange(16) + len(self.calls) * 100
        result[0, :, 1] = torch.arange(16) * .01 + len(self.calls)
        self.outputs.append(result.clone())
        return result, 16


def backend():
    seed = torch.zeros(4, 366)
    goals = [seed.clone() for _ in range(3)]
    for i, goal in enumerate(goals):
        goal[:, 2] = i + 1
        goal[:, 1] = (i + 1) * .2
    course = dict(id="up", initial_history=seed.tolist(), forward_xz=[0, 1],
                  goals=[dict(features=g.tolist()) for g in goals])
    primitive = Primitive()
    return NativeStairLiveBackend(SimpleNamespace(device="cpu"), {}, course, primitive=primitive), primitive


def predict(b, velocity):
    result = b.predict_plan([velocity], [0])["planned_motion"]
    b.history = result[:, -4:].clone()
    return result


def test_buffer_publishes_only_raw_predictions_without_dropping_four_frame_tail():
    b, p = backend()
    first = predict(b, [0, 1])
    second = predict(b, [0, 1])
    torch.testing.assert_close(first, p.outputs[0][:, 4:12])
    torch.testing.assert_close(second[:, :4], p.outputs[0][:, 12:16])
    torch.testing.assert_close(second[:, 4:], p.outputs[1][:, 4:8])
    torch.testing.assert_close(p.calls[1][0], p.outputs[0][:, 12:16])
    assert b.cursor == 0 and b.active_goal_index == 1
    assert b.last_generation["plans"][0]["goal_index"] == 1


def test_release_finishes_landing_then_generates_hold_from_neural_context():
    b, p = backend()
    first = predict(b, [0, 1])
    second = predict(b, [0, 0])
    assert b.primitive_state == "hold"
    torch.testing.assert_close(second[:, :4], p.outputs[0][:, 12:16])
    torch.testing.assert_close(p.calls[-1][0], p.outputs[0][:, 12:16])
    torch.testing.assert_close(p.calls[-1][1], b.goals[0][:, -1:].repeat(1, 4, 1))
    torch.testing.assert_close(second[:, 4:], p.outputs[-1][:, 4:8])
    assert b.cursor == 0


def test_exit_handoff_preserves_tail_and_reverse_returns_to_stair_intent():
    b, p = backend()
    calls = []
    class Exit:
        platform_y_m = .6
        def predict(self, history, velocity, heading, *, target_offset_xz):
            calls.append(history.clone())
            assert velocity == [[0., 0.]]
            self.offset = np.array(target_offset_xz)
            return dict(planned_motion=torch.full((1, 8, 366), 900.), neural_generation=True)
    b.exit_locomotion = Exit()
    for _ in range(4):
        predict(b, [0, 1])
    assert not calls
    output = predict(b, [0, 1])
    assert b.primitive_state == 'exit_idle'
    torch.testing.assert_close(output[:, :4], p.outputs[-1][:, 12:16])
    torch.testing.assert_close(calls[0], p.outputs[-1][:, 12:16])
    assert torch.all(output[:, 4:] == 900.)
    predict(b, [-1, 0])
    np.testing.assert_allclose(b.exit_locomotion.offset, [-.25, 0.])
    count = len(p.calls)
    output = predict(b, [0, -1])
    assert len(p.calls) == count + 1 and b.primitive_state == 'backtrack'
    assert b.active_goal_index == 1
    assert torch.all(p.calls[-1][0] == 900.)
    torch.testing.assert_close(output, p.outputs[-1][:, 4:12])


def test_reverse_selects_prior_goal_and_strafe_changes_only_authored_input():
    b, p = backend()
    for _ in range(3):
        predict(b, [0, 1])
    assert b.cursor == 1
    predict(b, [0, -1])
    assert b.active_goal_index == 0 and b.primitive_state == "backtrack"
    assert p.calls[-1][1][0, -1, 2] == 1
    b, p = backend()
    output = predict(b, [-1, 1])
    assert abs(b.lateral_offset) <= .25
    assert p.calls[0][1][0, -1, 0] == -.25
    torch.testing.assert_close(output, p.outputs[0][:, 4:12])


def test_strafe_input_offset_is_bounded_and_not_applied_to_output():
    b, p = backend()
    for _ in range(4):
        output = predict(b, [-1, 0])
        torch.testing.assert_close(output, p.outputs[-1][:, 4:12])
    assert b.lateral_offset == .25
    assert abs(float(p.calls[-1][1][0, -1, 0])) <= .25


@pytest.mark.parametrize("velocity", [[[float("nan"), 1]], [[1, 2, 3]]])
def test_nonfinite_and_bad_commands_rejected(velocity):
    b, _ = backend()
    with pytest.raises(ValueError):
        b.predict_plan(velocity, [0])


def test_course_export_contains_only_sparse_conditions_and_preserves_height(tmp_path):
    import json
    from scripts.export_native_mb_stair_courses import export

    source = np.arange(40 * 366, dtype=np.float32).reshape(40, 366) / 1000
    windows = []
    for artifact in ("00_up_start000000", "03_down_start000000"):
        np.savez(tmp_path / (artifact + ".npz"), features_source=source, fps=30)
        windows.append(dict(artifact=artifact, all_landing_endpoints=[15, 35], source={"split": "validation"},
                            support_patches=[dict(id=0, confidence=.8, height_m=.1,
                                                 polygon_xz_m=[[0, 0], [1, 0], [1, 1]])]))
    (tmp_path / "report.json").write_text(json.dumps(dict(windows=windows, bundle=dict(
        root_checkpoint_sha256="root", pose_checkpoint_sha256="pose", tokenizer=dict(checkpoint_sha256="tok")))))
    result = export(tmp_path, tmp_path / "courses.json")
    course = result["courses"][0]
    assert len(course["initial_history"]) == 4 and len(course["goals"]) == 2
    goal = np.asarray(course["goals"][0]["features"])
    np.testing.assert_allclose(goal[:, 1], source[12:16, 1])
    np.testing.assert_allclose(goal[:, 3:], source[12:16, 3:])
    np.testing.assert_allclose(goal[:, 0], source[12:16, 0] - source[3, 0])
    assert course["support_patches"][0]["polygon_ue_cm"][0][2] == 10
    assert "frames" not in course and "features_source" not in course


def test_full3d_history_rejects_any_root_rewrite():
    from scripts.native_soma_stair_live_server import StairLiveSession
    live = object.__new__(StairLiveSession)
    motion = torch.ones(1, 8, 366)
    live.plans = {1: (0, motion)}
    records = [dict(sequence=1, frame=i, root_m=[1., 1., 1.]) for i in range(4, 8)]
    torch.testing.assert_close(live.history_motion(records, 7), motion[:, 4:8])
    records[0]["root_m"][0] += .00005
    torch.testing.assert_close(live.history_motion(records, 7), motion[:, 4:8], atol=0, rtol=0)
    records[0]["root_m"][0] -= .00005
    for axis in range(3):
        records[0]["root_m"][axis] += .01
        with pytest.raises(ValueError, match="cannot rewrite"):
            live.history_motion(records, 7)
        records[0]["root_m"][axis] -= .01


def test_stair_session_requires_committed_protocol_and_pins_course():
    from scripts.native_soma_stair_live_server import StairLiveSession
    from scripts.soma_ue_live_server import COMMITTED_SCHEMA
    live = object.__new__(StairLiveSession)
    live.courses = {"up": {}, "down": {}}
    live.course_id = "up"
    with pytest.raises(ValueError, match="Reset"):
        live.respond(dict(schema=COMMITTED_SCHEMA, stair_course="down"))
    with pytest.raises(ValueError, match="v2 committed"):
        live.respond(dict(schema="motionbricks.soma_control.v1", stair_course="up"))


def test_server_builds_exit_skill_from_explicit_course_not_generated_feet(monkeypatch):
    from scripts import native_soma_stair_live_server as server
    from motionbricks.data.native_mb_data import build_native_representation
    from test_native_mb_frame import idle_reference

    rep, exit_rep = build_native_representation(), build_native_representation()
    reference = idle_reference(exit_rep)
    bundle = dict(bone_names=rep.skeleton.bone_order_names, fps=30)
    exit_bundle = dict(bundle, reference_clips={'idle': reference[0]})
    infer = SimpleNamespace(global_motion_rep=rep, device='cpu')
    exit_infer = SimpleNamespace(global_motion_rep=exit_rep, device='cpu')
    goal = reference.clone()
    goal[..., 0] = 2.
    goal[..., 2] = 3.
    course = dict(id='up', forward_xz=[0., 1.], initial_history=reference[0].tolist(),
                  goals=[dict(features=goal[0].tolist())], exit_platform_y_m=1.3)
    captured = {}
    def backend(*args, **kwargs):
        captured.update(kwargs)
        return captured
    monkeypatch.setattr(server, 'NativeSmartObject', lambda inference: inference)
    monkeypatch.setattr(server, 'NativeStairLiveBackend', backend)
    result = server.make_backend(infer, bundle, course, (exit_infer, exit_bundle))
    assert result['exit_locomotion'].platform_y_m == 1.3
    assert result['exit_locomotion'].primitive is exit_infer
    torch.testing.assert_close(result['exit_locomotion'].goal[0, -1, [0, 2]], torch.tensor([2., 3.]))
    del course['exit_platform_y_m']
    with pytest.raises(KeyError):
        server.make_backend(infer, bundle, course, (exit_infer, exit_bundle))
    exit_bundle['fps'] = 60
    with pytest.raises(ValueError, match='representation'):
        server.make_backend(infer, bundle, course, (exit_infer, exit_bundle))
