from pathlib import Path

import pytest
import torch

from motionbricks.data.soma_cuda_graph import StaticCudaCall, install_backbone_graphs, install_decoder_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_replay_outputs_are_not_overwritten():
    call = StaticCudaCall(lambda x: {"value": x * 2})
    with torch.inference_mode():
        first = call(torch.ones(4, device="cuda"))
        second = call(torch.full((4,), 3., device="cuda"))
        assert torch.equal(first["value"], torch.full((4,), 2., device="cuda"))
        assert torch.equal(second["value"], torch.full((4,), 6., device="cuda"))
        strided = torch.arange(8., device='cuda')[::2]
        third = call(strided)
        torch.testing.assert_close(third['value'], strided * 2)
        # Discarded maintenance replays must not alter previously published plans.
        call.graph.replay()
        call.graph.replay()
        torch.testing.assert_close(first['value'], torch.full((4,), 2., device='cuda'))
        torch.testing.assert_close(third['value'], strided * 2)
        with pytest.raises(ValueError, match="shape"):
            call(torch.ones(5, device="cuda"))
    with pytest.raises(ValueError, match="inference-only"):
        call(torch.ones(4, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("relative_path", ["out/soma_control_v3/baseline_backend.pt",
    "out/soma_pose_rollout_20261005/trial_v3/C/backend.pt"])
@pytest.mark.parametrize("backbones", [False, True])
def test_native_mb_decoder_graph_preserves_raw_control_motion(relative_path, backbones):
    from motionbricks.data.bones_control import state_digest
    from motionbricks.data.bones_rollout import SomaLocomotionBackend
    from scripts.portable_bones_control import load_bundle

    path = Path(__file__).resolve().parents[1] / relative_path
    if not path.exists():
        pytest.skip("Original SOMA pilot bundle not present")
    inference, bundle = load_bundle(path, "cuda")
    frozen = {name: state_digest(module) for name, module in (
        ("root", inference._root_model), ("pose", inference._pose_model))}
    token_ids = []

    def record_tokens(native):
        def forward(ids, *args, **kwargs):
            token_ids.append(ids.clone())
            return native(ids, *args, **kwargs)
        return forward

    inference._vqvae_pose_model.forward_decoder = record_tokens(inference._vqvae_pose_model.forward_decoder)
    commands = [([0, 0], 0.), ([0, 1.2], 0.), ([1.2, 0], 1.57), ([0, -1.2], 0.)]
    with torch.inference_mode():
        initial = bundle["initial_history"].cuda()
        # Compare identical externally supplied histories, not divergently fed outputs.
        histories, expected = [], []
        backend = SomaLocomotionBackend(inference, initial, argmax=True, commit_frames=8)
        for velocity, heading in commands:
            histories.append(backend.history.clone())
            expected.append(backend.step([velocity], [heading])["motion"])
        expected_tokens = token_ids.copy()
        token_ids.clear()
        install_decoder_graph(inference)
        if backbones:
            install_backbone_graphs(inference)
        inference._vqvae_pose_model.forward_decoder = record_tokens(inference._vqvae_pose_model.forward_decoder)
        for history, (velocity, heading), source in zip(histories, commands, expected):
            backend = SomaLocomotionBackend(inference, history, argmax=True, commit_frames=8)
            result = backend.step([velocity], [heading])["motion"]
            if backbones:
                actual_pose = inference.motion_rep.inverse(result, False, return_all=False)
                source_pose = inference.motion_rep.inverse(source, False, return_all=False)
                position_error = (actual_pose["posed_joints"] - source_pose["posed_joints"]).norm(dim=-1).max()
                relative = source_pose["local_joint_rots"].double().transpose(-1, -2) @ actual_pose["local_joint_rots"].double()
                skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                                    relative[..., 0, 2] - relative[..., 2, 0],
                                    relative[..., 1, 0] - relative[..., 0, 1]), -1)
                angle_error = torch.atan2(skew.norm(dim=-1), relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1).rad2deg().max()
                # Reuse the existing CPU/CUDA parity's physical-unit limits;
                # dense/nested kernels are not bitwise equal, nor are mixed
                # position/rotation/velocity feature errors a common unit.
                assert position_error <= .002
                assert angle_error <= .2
                print({"graph_bundle": relative_path, "position_max_mm": position_error.item() * 1000,
                       "rotation_max_deg": angle_error.item()})
            else:
                torch.testing.assert_close(result, source, atol=1e-5, rtol=1e-5)
        assert len(token_ids) == len(expected_tokens)
        assert all(torch.equal(actual, target) for actual, target in zip(token_ids, expected_tokens))
    assert {name: state_digest(module) for name, module in (
        ("root", inference._root_model), ("pose", inference._pose_model))} == frozen
