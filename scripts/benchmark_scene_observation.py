"""Per-frame latency of scene observations on the deployment GPU (target: RTX 3080).

Times, at batch 1 (one character, the runtime case) and optionally a large batch
(offline cache building):

* skeleton rays (motionbricks.data.skeleton_ray_sampler), eager and CUDA graph;
* the existing 1,536-point volume (VolumetricEgoSampler on CompositeSceneField),
  eager, and CUDA graph if it can be captured (it validates inputs on the host, so
  capture is expected to fail; that is reported, not hidden).

The scene is N random oriented boxes plus a ground slab (N = ray box capacity).
GPU times come from CUDA events around each call; the report gives p50/p99/max
and checks the budget below. Nothing is trained or written besides the report.

    python scripts/benchmark_scene_observation.py --device cuda --output obs_bench_3080.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Per-frame budget at 60 fps is 16.7 ms for everything; observations must stay a small part.
BUDGET_MS = {"rays_graph_p99": 0.5, "rays_eager_p99": 2.0}


def random_scene(count, seed=0):
    rng = np.random.default_rng(seed)
    boxes = [{"center_xyz_m": [0, -0.1, 0], "half_extent_xyz_m": [20, 0.1, 20],
              "rotation_local_to_world": np.eye(3).tolist()}]
    for _ in range(count - 1):
        yaw = rng.uniform(0, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        boxes.append({"center_xyz_m": rng.uniform([-4, 0, -4], [4, 2.5, 4]).tolist(),
                      "half_extent_xyz_m": rng.uniform(0.1, 0.8, 3).tolist(),
                      "rotation_local_to_world": [[c, 0, s], [0, 1, 0], [-s, 0, c]]})
    return boxes


def skeleton_names():
    """Joint order of the controller skeleton; falls back to the layout's own names."""
    try:
        from motionbricks.data.bones_seed import build_representation
        return list(build_representation().skeleton.bone_order_names), "controller_skeleton"
    except Exception as error:  # noqa: BLE001 - reported in the output
        from motionbricks.data.skeleton_ray_sampler import default_layout
        names = sorted({j for node in default_layout() for j in node.joints})
        return names, f"layout_names_only ({type(error).__name__}: {error})"


def timer(torch, device):
    if device.type == "cuda":
        def run(function, iterations):
            times = []
            for _ in range(iterations):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            return np.asarray(times)
    else:
        def run(function, iterations):
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                function()
                times.append((time.perf_counter() - start) * 1000)
            return np.asarray(times)
    return run


def summary(times):
    return {"p50_ms": round(float(np.percentile(times, 50)), 4), "p99_ms": round(float(np.percentile(times, 99)), 4),
            "max_ms": round(float(times.max()), 4), "iterations": int(len(times))}


def graphed(torch, function, inputs):
    """Capture function(*inputs) into a CUDA graph; returns a replay closure."""
    static = [x.clone() for x in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function(*static)
    return graph.replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boxes", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--cache-batch", type=int, default=256, help="Also time this batch size; 0 skips")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler

    device = torch.device(args.device)
    names, names_source = skeleton_names()
    sampler = SkeletonRaySampler(names, box_capacity=args.boxes)
    boxes_list = random_scene(args.boxes)
    boxes = sampler.boxes(boxes_list, device=device, dtype=torch.float32)
    run = timer(torch, device)
    report = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": torch.__version__, "python": platform.python_version(), "joint_names": names_source,
              "rays": sampler.layout.rays, "feature_dim": sampler.layout.feature_dim, "boxes": args.boxes,
              "budget_ms": BUDGET_MS, "results": {}}

    def pose(batch):
        rng = np.random.default_rng(1)
        joints = torch.tensor(rng.normal(0, 0.3, (batch, len(names), 3)) + [0, 1.0, 0], dtype=torch.float32,
                              device=device)
        heading = torch.tensor(rng.normal(size=(batch, 2)), dtype=torch.float32, device=device)
        return joints, heading

    with torch.inference_mode():
        for batch in [1] + ([args.cache_batch] if args.cache_batch else []):
            joints, heading = pose(batch)
            SkeletonRaySampler.check_inputs(joints, heading, boxes)
            call = lambda j=joints, h=heading: sampler.observe(j, h, boxes)  # noqa: E731
            run(call, args.warmup)
            report["results"][f"rays_eager_b{batch}"] = summary(run(call, args.iterations))
            if device.type == "cuda":
                replay = graphed(torch, lambda j, h: sampler.observe(j, h, boxes), (joints, heading))
                run(replay, args.warmup)
                report["results"][f"rays_graph_b{batch}"] = summary(run(replay, args.iterations))

        # Existing volume observation, same boxes, batch 1.
        from motionbricks.data.pfnn_3d_scene import CompositeSceneField
        from motionbricks.data.scene_geometry_field import SceneBox, SceneManifest
        from motionbricks.data.unified_scene_field import VolumetricEgoSampler
        manifest = SceneManifest("bench", (-30, -5, -30), (30, 10, 30), tuple(
            SceneBox(k + 1, f"box_{k}", "support", tuple(b["center_xyz_m"]), tuple(b["half_extent_xyz_m"]),
                     tuple(map(tuple, b["rotation_local_to_world"]))) for k, b in enumerate(boxes_list)))
        field = CompositeSceneField(None, manifest)
        field.prepare(device, torch.float32)
        volume = VolumetricEgoSampler()
        history = torch.zeros(1, 4, 5, device=device)
        history[..., 1] = 1.0
        history[..., 3] = 1.0
        history[:, :, 2] = torch.arange(4, device=device) * 0.03
        velocity = torch.tensor([[0.0, 1.0]], device=device)
        call = lambda: volume.observe(history, velocity, field, 1.0)  # noqa: E731
        run(call, args.warmup)
        report["results"]["volume_eager_b1"] = summary(run(call, args.iterations))
        if device.type == "cuda":
            try:
                replay = graphed(torch, lambda h, v: volume.observe(h, v, field, 1.0), (history, velocity))
                run(replay, args.warmup)
                report["results"]["volume_graph_b1"] = summary(run(replay, args.iterations))
            except Exception as error:  # noqa: BLE001 - expected: host-side validation syncs
                report["results"]["volume_graph_b1"] = {"capturable": False,
                                                        "error": f"{type(error).__name__}: {str(error)[:200]}"}

    results = report["results"]
    gates = {}
    if "rays_graph_b1" in results:
        gates["rays_graph_p99"] = results["rays_graph_b1"]["p99_ms"] <= BUDGET_MS["rays_graph_p99"]
    gates["rays_eager_p99"] = results["rays_eager_b1"]["p99_ms"] <= BUDGET_MS["rays_eager_p99"]
    report["gates_passed"] = gates
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
