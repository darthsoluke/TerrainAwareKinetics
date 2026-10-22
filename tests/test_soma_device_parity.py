from pathlib import Path

import json
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU/CUDA comparison requires CUDA")
def test_native_cpu_cuda_control_tokens_and_motion_match():
    from motionbricks.data.bones_rollout import SomaLocomotionBackend
    from scripts.portable_bones_control import load_bundle

    path = Path(__file__).resolve().parents[1] / "out/soma_control_v3/baseline_backend.pt"
    if not path.exists():
        pytest.skip("Pilot bundle not present")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        cpu, bundle = load_bundle(path, "cpu")
        cuda, _ = load_bundle(path, "cuda")
        captured, position_errors, angle_errors = {}, [], []
        def capture(name):
            def hook(module, args, output):
                captured[name] = output["pose_logits"][:, :16].argmax(-1).detach().cpu()
            return hook
        handles = [cpu._pose_model.backbone_net.register_forward_hook(capture("cpu")),
                   cuda._pose_model.backbone_net.register_forward_hook(capture("cuda"))]
        try:
            with torch.inference_mode():
                backend = SomaLocomotionBackend(cuda, bundle["initial_history"].cuda(), argmax=True, commit_frames=8)
                for velocity, angle in (([0, 0], 0.), ([0, 1.2], 0.), ([0, 2.7], 0.), ([1.2, 0], 1.57), ([0, -1.2], 0.)):
                    history = backend.history.detach().cpu().clone()
                    expected = backend.step([velocity], [angle])["motion"].cpu()
                    alternative = SomaLocomotionBackend(cpu, history, argmax=True, commit_frames=8).step([velocity], [angle])["motion"]
                    assert torch.equal(captured["cpu"], captured["cuda"])
                    # Motion features mix metres, rotation-matrix entries and contact
                    # channels. Screen device roundoff in physical pose units instead.
                    a = cuda.motion_rep.inverse(alternative.cuda(), False, return_all=False)
                    b = cuda.motion_rep.inverse(expected.cuda(), False, return_all=False)
                    position_error = (a["posed_joints"] - b["posed_joints"]).norm(dim=-1).max().item()
                    relative = a["local_joint_rots"] @ b["local_joint_rots"].transpose(-1, -2)
                    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
                    angle_error = torch.rad2deg(cosine.acos()).max().item()
                    position_errors.append(position_error)
                    angle_errors.append(angle_error)
                    assert position_error <= .002  # Numerical screen, not motion-quality approval.
                    assert angle_error <= .2
        finally:
            for handle in handles:
                handle.remove()
        print(json.dumps({"matched_active_token_requests": len(position_errors),
                          "max_joint_position_difference_m": max(position_errors),
                          "max_local_rotation_difference_deg": max(angle_errors),
                          "scope": "five matched external histories; not a long-rollout equivalence claim"}))
    finally:
        torch.set_num_threads(previous_threads)
