"""SOMA conditioning and mutation contracts for original smart locomotion."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_locomotion import NativeReferenceClips, NativeSmartLocomotion
from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix
from test_native_mb_stage_training import _motion_and_representation


@pytest.fixture(scope="module", autouse=True)
def _cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _clip(rep, frames=13, velocity=(0.0, 0.0), height=1.1, heading=0.0):
    rotations = torch.eye(3).expand(frames, rep.skeleton.nbjoints, 3, 3).clone()
    rotations[:, 0] = angle_to_Y_rotation_matrix(torch.full((frames,), heading)).float()
    translation = torch.zeros(frames, 3)
    time = torch.arange(frames) / rep.fps
    translation[:, 0] = time * velocity[0]
    translation[:, 2] = time * velocity[1]
    translation[:, 1] = height + torch.arange(frames) * 0.005
    return rep({"local_joint_rots": rotations, "translation": translation,
                "foot_contacts": torch.ones(frames, 4)},
               to_normalize=False, lengths=torch.tensor(frames))


class RecordingInference:
    device = "cpu"
    _args = {"min_tokens": 6, "max_tokens": 16}

    def __init__(self, rep, tokens=8):
        self.motion_rep = self.global_motion_rep = rep
        self.local_motion_rep = rep.dual_rep.local_motion_rep
        self._root_model = SimpleNamespace(backbone_net=SimpleNamespace(MASKED_NUM_TOKENS=18))
        self.tokens = torch.tensor([tokens])
        # Distinct arbitrary decoder values expose any output replacement/blend.
        self.prediction = torch.arange(tokens * 4 * 366).reshape(1, tokens * 4, 366).float() / 1000
        self.calls = []

    def predict(self, *args, **kwargs):
        names = ("global_root_values", "has_global_root_values", "local_root_values",
                 "has_local_root_values", "local_poses", "has_local_poses", "num_tokens")
        call = dict(zip(names, args))
        call.update(kwargs)
        self.calls.append({name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                           for name, value in call.items()})
        return self.prediction, self.tokens


@pytest.fixture
def soma_case():
    rep, _ = _motion_and_representation(frame_count=17)
    clips = {"idle": _clip(rep), "walk": _clip(rep, velocity=(0.4, 0.1), heading=0.2)}
    originals = {name: value.clone() for name, value in clips.items()}
    bank = NativeReferenceClips(rep, clips)
    for name, value in clips.items():
        torch.testing.assert_close(value, originals[name], rtol=0, atol=0)
    history = _clip(rep, frames=4, velocity=(0.15, -0.1), height=2.7, heading=-0.3)[None]
    return rep, bank, history


@pytest.mark.parametrize("method", ["_generate_spring_model_position_and_heading",
                                   "_generate_target_joint_transforms",
                                   "_predict_inbetween_motion_features"])
def test_original_methods_are_bound_directly(method):
    assert getattr(NativeSmartLocomotion, method) is getattr(full_navigation_agent, method)


@pytest.mark.parametrize("keyframes,global_indices,local_indices,pose_indices", [
    (4, list(range(8)), [0, 1, 2, 4, 5, 6, 7], list(range(8))),
    (1, [3, 7], [2, 7], [3, 7]),
])
def test_future_body_conditions_survive_decoder_skip(soma_case, keyframes, global_indices,
                                                      local_indices, pose_indices):
    rep, bank, history = soma_case
    infer = RecordingInference(rep)
    agent = NativeSmartLocomotion(infer, bank, keyframes_per_end=keyframes)
    prediction, count = agent.predict(history, (0.8, 0.2), 0.7, style="walk", random_seed=11)
    recorded = infer.calls[-1]
    assert recorded["config"] == {
        "num_inference_step": 1, "smooth_root_traj": False,
        "allow_pred_out_of_reach_num_tokens": False,
        "pose_token_sampling_use_argmax": True, "skip_ending_target_cond": False,
    }
    for name, indices in (("has_global_root_values", global_indices),
                          ("has_local_root_values", local_indices),
                          ("has_local_poses", pose_indices)):
        assert recorded[name].nonzero()[:, 1].tolist() == indices
        torch.testing.assert_close(agent.last_constraints[name], recorded[name], rtol=0, atol=0)
    assert recorded["has_local_poses"][0, 7]
    assert recorded["local_poses"][:, 4:].abs().sum() > 0
    for name in ("global_root_values", "local_root_values", "local_poses"):
        assert recorded[name].shape[:2] == (1, 8)
        assert torch.isfinite(recorded[name]).all()
    assert prediction is infer.prediction
    assert count.tolist() == [8]


def test_four_context_one_target_changes_only_masks_not_duration_or_outputs(soma_case):
    rep, bank, history = soma_case
    infer = RecordingInference(rep, tokens=11)
    history_before, prediction_before = history.clone(), infer.prediction.clone()
    baseline = NativeSmartLocomotion(infer, bank, keyframes_per_end=4)
    baseline_prediction, baseline_count = baseline.predict(
        history, (0.8, 0.2), 0.7, style="walk", random_seed=11)
    baseline_call = infer.calls[-1]

    agent = NativeSmartLocomotion(infer, bank, context_keyframes=4, keyframes_per_end=1)
    prediction, count = agent.predict(history, (0.8, 0.2), 0.7, style="walk", random_seed=11)
    call = infer.calls[-1]
    for name, indices in (("has_global_root_values", [0, 1, 2, 3, 7]),
                          ("has_local_root_values", [0, 1, 2, 7]),
                          ("has_local_poses", [0, 1, 2, 3, 7])):
        assert call[name].nonzero()[:, 1].tolist() == indices
        torch.testing.assert_close(agent.last_constraints[name], call[name], rtol=0, atol=0)
    for name in ("global_root_values", "local_root_values", "local_poses", "num_tokens",
                 "allowed_pred_num_tokens"):
        torch.testing.assert_close(call[name], baseline_call[name], rtol=0, atol=0)
    assert call["num_tokens"].tolist() == [infer._root_model.backbone_net.MASKED_NUM_TOKENS]
    assert call["allowed_pred_num_tokens"].tolist() == [[1] * 6 + [0] * 5]
    assert call["config"] == baseline_call["config"]
    assert call["config"]["skip_ending_target_cond"] is False
    assert prediction is baseline_prediction is infer.prediction
    assert count.tolist() == baseline_count.tolist() == [11]
    torch.testing.assert_close(prediction, prediction_before, rtol=0, atol=0)
    torch.testing.assert_close(history, history_before, rtol=0, atol=0)


def test_goal_height_comes_from_reference_not_drifted_history(soma_case):
    rep, bank, history = soma_case
    infer = RecordingInference(rep)
    agent = NativeSmartLocomotion(infer, bank)
    seed = 11
    row = list(bank.CLIPS).index("walk")
    start = seed % (int(bank.num_frames_per_clip[row]) - 4)
    expected = bank.global_joint_positions[row, start:start + 4, 0, 1]
    for drift in (0.0, 3.0):
        changed = history.clone()
        changed[:, :, rep.indices["global_root_pos"][1]] += drift
        agent.predict(changed, (0.8, 0.2), 0.7, style="walk", random_seed=seed)
        call = infer.calls[-1]
        torch.testing.assert_close(call["global_root_values"][0, 4:, 1], expected)
        torch.testing.assert_close(call["local_root_values"][0, 4:, 3], expected)
        assert not torch.allclose(call["global_root_values"][0, :4, 1], expected)
    torch.testing.assert_close(infer.calls[0]["global_root_values"][:, 4:],
                               infer.calls[1]["global_root_values"][:, 4:], rtol=0, atol=0)
    torch.testing.assert_close(infer.calls[0]["local_poses"][:, 4:],
                               infer.calls[1]["local_poses"][:, 4:], rtol=0, atol=0)


def _spring_values(current, velocity, target, half_life_parameter, times):
    y = (4.0 * 0.69314718056) / (half_life_parameter + 1e-5) / 2.0
    x = y * times
    decay = 1.0 / (1.0 + x + 0.48 * x.square() + 0.235 * x.pow(3))
    j0 = current - target
    j1 = velocity + j0 * y
    return (j0[..., None] + j1[..., None] * times) * decay + target[..., None]


@pytest.mark.parametrize("velocity", [(1.25, 0.0), (0.0, 1.25), (0.7, -0.4)])
def test_spring_goals_preserve_xz_meters_per_second_without_g1_double_scale(soma_case, velocity):
    rep, bank, history = soma_case
    infer = RecordingInference(rep)
    agent = NativeSmartLocomotion(infer, bank)
    heading = 0.7
    agent.predict(history, velocity, heading, style="walk", random_seed=11)
    call = infer.calls[-1]
    transforms = rep.inverse(history, False, return_quat=False, return_all=True)
    root = transforms["posed_joints"][0, :, 0]
    current = root[0, [0, 2]]
    current_velocity = (root[1, [0, 2]] - current) * rep.fps
    times = 1.0 + torch.arange(4) / rep.fps
    expected = _spring_values(current, current_velocity, current + torch.tensor(velocity), 0.8, times).T
    torch.testing.assert_close(call["global_root_values"][0, 4:, [0, 2]], expected)
    expected_velocity = torch.diff(expected, dim=0) * rep.fps
    torch.testing.assert_close(call["local_root_values"][0, 4:7, 1:3], expected_velocity,
                               rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(call["local_root_values"][0, 7, 1:3], expected_velocity[-1],
                               rtol=1e-4, atol=1e-5)
    rotations = transforms["global_joint_rots"][0, :, 0]
    angles = torch.atan2(rotations[:, 0, 2], rotations[:, 2, 2])
    angular_velocity = ((angles[1] - angles[0] + math.pi) % (2 * math.pi) - math.pi) * rep.fps
    expected_angles = _spring_values(angles[0], angular_velocity, torch.tensor(heading), 0.17, times)
    torch.testing.assert_close(call["global_root_values"][0, 4:, 3], expected_angles.cos())
    torch.testing.assert_close(call["global_root_values"][0, 4:, 4], expected_angles.sin())


@pytest.mark.parametrize("realign", [False, True])
def test_only_internal_context_is_realigned_and_prediction_inputs_and_clips_are_untouched(
        soma_case, monkeypatch, realign):
    rep, bank, history = soma_case
    infer = RecordingInference(rep)
    agent = NativeSmartLocomotion(infer, bank, source_root_realignment=realign)
    history_before, prediction_before = history.clone(), infer.prediction.clone()
    buffers_before = {name: value.clone() for name, value in bank.state_dict().items()}
    metadata_before = copy.deepcopy(bank.CLIPS)
    observed = {}

    def capture(inputs):
        observed["before"] = inputs["context_global_joint_positions"].clone()
        observed["before_rotations"] = inputs["context_global_joint_rotations"].clone()
        result = full_navigation_agent._generate_target_joint_transforms(agent, inputs)
        observed["after"] = inputs["context_global_joint_positions"].clone()
        observed["after_rotations"] = inputs["context_global_joint_rotations"].clone()
        observed["positions"] = inputs["start_root_positions"].transpose(1, 2).clone()
        observed["headings"] = inputs["start_root_headings"].clone()
        return result

    monkeypatch.setattr(agent, "_generate_target_joint_transforms", capture)
    prediction, _ = agent.predict(history, (1.0, 0.4), 0.7, style="walk", random_seed=11)
    assert prediction is infer.prediction
    torch.testing.assert_close(prediction, prediction_before, rtol=0, atol=0)
    torch.testing.assert_close(history, history_before, rtol=0, atol=0)
    assert bank.CLIPS == metadata_before
    for name, value in bank.state_dict().items():
        torch.testing.assert_close(value, buffers_before[name], rtol=0, atol=0)
    if realign:
        assert not torch.allclose(observed["after"], observed["before"])
        torch.testing.assert_close(observed["after"][:, :, 0, [0, 2]], observed["positions"])
        angles = torch.atan2(observed["after_rotations"][:, :, 0, 0, 2],
                             observed["after_rotations"][:, :, 0, 2, 2])
        torch.testing.assert_close(angles, observed["headings"], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(observed["after"][..., 1], observed["before"][..., 1])
    else:
        torch.testing.assert_close(observed["after"], observed["before"], rtol=0, atol=0)
        torch.testing.assert_close(observed["after_rotations"], observed["before_rotations"], rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [6, 11])
def test_duration_stays_masked_with_native_six_to_eleven_choices(soma_case, tokens):
    rep, bank, history = soma_case
    infer = RecordingInference(rep, tokens=tokens)
    agent = NativeSmartLocomotion(infer, bank)
    prediction, count = agent.predict(history, (0.8, 0.2), 0.7, style="walk", random_seed=11)
    call = infer.calls[-1]
    assert call["num_tokens"].tolist() == [infer._root_model.backbone_net.MASKED_NUM_TOKENS]
    assert call["allowed_pred_num_tokens"].tolist() == [[1] * 6 + [0] * 5]
    assert count.tolist() == [tokens]
    assert prediction.shape == (1, tokens * 4, 366)


def test_fixed_duration_changes_only_allowed_classes_not_masked_input(soma_case):
    rep, bank, history = soma_case
    infer = RecordingInference(rep)
    agent = NativeSmartLocomotion(infer, bank, fixed_tokens=8)
    agent.predict(history, (0.8, 0.2), 0.7, style="walk", random_seed=11)
    assert infer.calls[-1]["num_tokens"].tolist() == [18]
    assert infer.calls[-1]["allowed_pred_num_tokens"].tolist() == [[0, 0, 1] + [0] * 8]


def test_full_conditioning_matches_direct_original_spring_placement_and_packing(soma_case):
    rep, bank, history = soma_case
    adapted, original = RecordingInference(rep), RecordingInference(rep)
    agent = NativeSmartLocomotion(adapted, bank)
    prediction, tokens = agent.predict(history, (0.8, 0.2), 0.7, style="walk", random_seed=11)
    host = SimpleNamespace(NUM_FRAMES_PER_TOKEN=4, _device=torch.device("cpu"), _fps=30,
                           _clip_holder=bank, _speed_scale_min=1.0, _speed_scale_max=1.0,
                           _target_root_realignment=True, _source_root_realignment=True,
                           FORCE_CANONICALIZATION=False, SKIP_ENDING_TARGET_COND=False,
                           _inferencer=original)
    transforms = rep.inverse(history, False, return_quat=False, return_all=True)
    velocity = history.new_tensor([0.8, 0.2])
    inputs = {
        "context_global_joint_positions": transforms["posed_joints"].clone(),
        "context_global_joint_rotations": transforms["global_joint_rots"].clone(),
        "mode": torch.tensor([1]), "movement_direction": (velocity / velocity.norm())[[1, 0]][None],
        "facing_direction": history.new_tensor([[math.cos(0.7), math.sin(0.7)]]),
        "target_vel": velocity.norm().reshape(1, 1) / 2,
        "random_seed": torch.tensor([11]),
        "allowed_pred_num_tokens": history.new_tensor([[1] * 6 + [0] * 5]),
    }
    spring_names = ("target_root_position", "target_root_positions", "target_root_headings",
                    "target_root_heading", "start_root_positions", "start_root_headings")
    inputs.update(zip(spring_names, full_navigation_agent._generate_spring_model_position_and_heading(host, inputs)))
    target_names = ("target_global_joint_positions", "target_global_joint_rotations", "target_global_root_positions")
    inputs.update(zip(target_names, full_navigation_agent._generate_target_joint_transforms(host, inputs)))
    direct_prediction, frames = full_navigation_agent._predict_inbetween_motion_features(host, inputs)
    assert direct_prediction is original.prediction and prediction is adapted.prediction
    torch.testing.assert_close(tokens * 4, frames, rtol=0, atol=0)
    assert adapted.calls[-1].keys() == original.calls[-1].keys()
    for name, value in adapted.calls[-1].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, original.calls[-1][name], rtol=0, atol=0)
        else:
            assert value == original.calls[-1][name]
