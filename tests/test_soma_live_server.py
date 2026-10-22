from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch

from scripts.soma_ue_live_server import COMMITTED_SCHEMA, SomaLiveSession, vector


def session():
    inference = SimpleNamespace(_root_model=torch.nn.Linear(1, 1), motion_rep=SimpleNamespace(fps=60))
    return SomaLiveSession(inference, {"initial_history": torch.zeros(1, 4, 366)}, "model", {"source_id": "soma"})


def records(sequence=1, start=0):
    return [{"sequence": sequence, "frame": index, "root_m": [2 + index, 1, 3]}
            for index in range(start, start + 4)]


def test_live_control_uses_deterministic_tokens_without_changing_offline_default():
    assert session().backend.argmax is True


def test_only_actually_played_roots_change_feedback_and_body_stays_unchanged():
    live = session()
    motion = torch.randn(1, 60, 366)
    motion[..., 1] = 1
    live.plans[1] = (0, motion)
    live.apply_history(records(), 3)
    torch.testing.assert_close(live.backend.history[..., 3:], motion[:, :4, 3:])
    torch.testing.assert_close(live.backend.history[0, :, :3], torch.tensor([r["root_m"] for r in records()], dtype=torch.float32))
    assert live.history_tick == 3


def test_mixed_plan_history_is_consecutive_without_resetting_to_source():
    live = session()
    motion = torch.ones(1, 60, 366)
    live.plans[1] = (0, motion)
    live.plans[2] = (4, motion * 2)
    history = records(start=2)
    history[2] = {"sequence": 2, "frame": 0, "root_m": [0, 2, 0]}
    history[3] = {"sequence": 2, "frame": 1, "root_m": [0, 2, 0]}
    live.apply_history(history, 5)
    assert live.backend.history[0, :, 7].tolist() == [1, 1, 2, 2]


@pytest.mark.parametrize("defect", ["stale", "missing", "time", "height", "nonfinite"])
def test_bad_feedback_rejected(defect):
    live = session()
    motion = torch.ones(1, 60, 366)
    live.plans[1] = (0, motion)
    history = records()
    if defect == "stale":
        live.history_tick = 3
    elif defect == "missing":
        history[0]["sequence"] = 99
    elif defect == "time":
        history[1]["frame"] = 2
    elif defect == "height":
        history[0]["root_m"][1] = 0
    else:
        history[0]["root_m"][0] = float("nan")
    with pytest.raises(ValueError):
        live.apply_history(history, 3)


def test_initialization_cannot_claim_future_history():
    with pytest.raises(ValueError):
        session().apply_history(records(), 3)


@pytest.mark.parametrize("value", [[1, 2], [1, 2, float("inf")]])
def test_commands_reject_wrong_shape_or_nonfinite(value):
    with pytest.raises(ValueError):
        vector(value, 3, "command")


def committed_fixture():
    live = session()
    motion = torch.ones(1, 60, 366)
    motion[0, :, 0] = torch.arange(60) / 60
    live.sequence = 1
    live.plans[1] = (0, motion)
    live.plan_origins[1] = torch.tensor([0., 1., 1.])
    history = [{"sequence": 1, "frame": i, "root_m": motion[0, i, :3].tolist()}
               for i in range(4, 8)]
    request = dict(history_kind="committed", history_tick=7, executed_tick=-1,
                   executed_history=[], history=history, publish_frames=8)
    return live, motion, request


def test_committed_context_is_explicitly_future_and_does_not_advance_observed_clock():
    live, motion, request = committed_fixture()
    live.apply_committed_context(request)
    torch.testing.assert_close(live.backend.history, motion[:, 4:8])
    assert live.history_tick == 7
    assert live.executed_tick == -1


@pytest.mark.parametrize("defect", ["kind", "cut", "sequence", "root", "observed", "length"])
def test_committed_context_cannot_claim_observation_or_change_published_motion(defect):
    live, _, request = committed_fixture()
    if defect == "kind":
        request["history_kind"] = "executed"
    elif defect == "cut":
        request["history_tick"] = 8
    elif defect == "sequence":
        request["history"][0]["sequence"] = 2
    elif defect == "root":
        request["history"][0]["root_m"][0] += .1
    elif defect == "observed":
        request["executed_history"] = request["history"]
    else:
        request["publish_frames"] = 12
    with pytest.raises(ValueError):
        live.apply_committed_context(request)


def test_committed_context_uses_actual_planar_origin_without_rewriting_body_or_height():
    live, first, request = committed_fixture()
    live.apply_committed_context(request)
    second = first.clone()
    second[..., 0] += 8 / 60
    live.sequence = 2
    live.plans[2] = (8, second)
    live.plan_origins[2] = first[0, 7, :3].clone()
    observed = [{"sequence": 1, "frame": i, "root_m": (first[0, i, :3] + torch.tensor([2., 0., 0.])).tolist()}
                for i in range(4, 8)]
    committed = [{"sequence": 2, "frame": i, "root_m": (second[0, i, :3] + torch.tensor([2., 0., 0.])).tolist()}
                 for i in range(4, 8)]
    live.apply_committed_context(dict(history_kind="committed", history_tick=15, executed_tick=7,
                                      executed_history=observed, history=committed, publish_frames=8))
    torch.testing.assert_close(live.backend.history[..., 3:], second[:, 4:8, 3:])
    torch.testing.assert_close(live.backend.history[..., 0], second[:, 4:8, 0] + 2)
    torch.testing.assert_close(live.backend.history[..., 1], second[:, 4:8, 1])
    assert live.executed_tick == 7


def test_live_v2_exports_only_native_committed_block_and_successive_ticks():
    live = session()
    live.manifest["joints"] = [{"name": "Hips", "parent_index": -1}]
    motion = torch.ones(1, 60, 366)
    live.backend.predict_plan = lambda *args: dict(planned_motion=motion, generation_seconds=.01)
    inverse_lengths = []

    def inverse(value, *args, **kwargs):
        inverse_lengths.append(value.shape[1])
        return dict(posed_joints=value[..., :3, None].transpose(-1, -2),
                    global_joint_rots=torch.ones(1, value.shape[1], 1, 4),
                    foot_contacts=torch.ones(1, value.shape[1], 4))

    live.inference.motion_rep.inverse = inverse
    request = dict(schema=COMMITTED_SCHEMA, model_id="model", source_id="soma", session_id="test",
                   request_id=1, velocity_ue_mps=[0, 0, 0], facing_ue=[1, 0, 0], client_time_s=1.,
                   history_kind="initial", history_tick=-1, executed_tick=-1,
                   history=[], executed_history=[], publish_frames=8)
    reply = live.respond(request)
    assert len(reply["frames"]) == 8 and reply["start_tick"] == 0
    history = [{"sequence": 1, "frame": i, "root_m": [1., 1., 1.]} for i in range(4, 8)]
    request.update(request_id=2, history_kind="committed", history_tick=7, history=history)
    reply = live.respond(request)
    assert reply["start_tick"] == 8 and len(reply["frames"]) == 8
    assert inverse_lengths == [8, 8]
    assert reply["post_processing"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU/CUDA export equivalence requires CUDA")
def test_official_cpu_export_preserves_raw_soma_pose_without_any_motion_processing():
    from motionbricks.data.bones_seed import build_representation

    cpu = build_representation()
    cuda = deepcopy(cpu).to("cuda")
    cuda.skeleton.joint_parents = cuda.skeleton.joint_parents.cpu()
    torch.manual_seed(2333)
    motion = torch.randn(1, 8, len(cpu.indices["all"]))
    with torch.inference_mode():
        a = cpu.inverse(motion, False, return_all=False, return_quat=True)
        b = cuda.inverse(motion.cuda(), False, return_all=False, return_quat=True)
    for key in ("posed_joints", "global_joint_rots", "foot_contacts"):
        torch.testing.assert_close(a[key], b[key].cpu(), atol=1e-5, rtol=1e-5)
