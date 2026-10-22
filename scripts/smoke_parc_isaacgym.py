"""Exercise gravity and ground contact in the official Isaac Gym GPU runtime."""

import argparse
import json
from pathlib import Path


def smoke(output):
    # Isaac Gym must load its native bindings before PyTorch is imported.
    from isaacgym import gymapi, gymtorch
    import torch

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1.0 / 60.0
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym could not create a headless PhysX simulator")
    try:
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, ground)
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
        asset = gym.create_box(sim, 0.1, 0.1, 0.1, gymapi.AssetOptions())
        transform = gymapi.Transform()
        transform.p.z = 1.0
        gym.create_actor(env, asset, transform, "falling_box", 0, 0)
        gym.prepare_sim(sim)
        roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        heights, contacts = [], []
        for _ in range(120):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)
            heights.append(float(roots[0, 2].item()))
            contacts.append(float(forces[0].norm().item()))
        assert roots.is_cuda and forces.is_cuda
        assert all(torch.isfinite(torch.tensor(heights)))
        assert heights[0] > 0.9 and min(heights) < 0.1
        assert 0.035 < heights[-1] < 0.075
        assert max(contacts) > 0
        result = {"status": "passed", "steps": 120, "dt": params.dt,
                  "gpu": torch.cuda.get_device_name(0), "tensor_device": str(roots.device),
                  "height_initial_m": heights[0], "height_final_m": heights[-1],
                  "contact_force_max_n": max(contacts), "torch": torch.__version__,
                  "runtime": "Isaac Gym Preview 4 / PhysX GPU",
                  "scope": "Engine smoke only; no PARC candidate has been physically tracked"}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    smoke(parser.parse_args().output)
