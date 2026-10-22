"""Batch-one complete-frame CUDA timing of the untrained PFNN compact actor.

Run with PYTHONPATH=.;out/vendor/kimodo in the motionbricks environment. The
whole-chain pass synchronizes only before/after a frame. A separate component
pass synchronizes each stage and therefore measures an instrumented schedule.
Both execute fresh sensing, prediction and native FK on every emitted frame.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.pfnn_3d_scene import CompositeSceneField, path_observation
from motionbricks.data.pfnn_frame_actor import FrameActor, advance, load_environment_encoder
from motionbricks.data.scene_geometry_field import example_scene_manifest
from motionbricks.data.unified_scene_field import VolumetricEgoSampler
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("coarse_query_1536", "root_step", "path_observation_512", "pose_step", "advance_fk")
SOURCE_PATHS = (
    "scripts/benchmark_pfnn_frame_actor.py", "motionbricks/data/pfnn_frame_actor.py",
    "motionbricks/data/pfnn_3d_scene.py", "motionbricks/data/native_mb_observed_history.py",
    "motionbricks/data/unified_scene_field.py", "motionbricks/data/unified_terrain.py",
    "motionbricks/data/joint_sdf_observation.py", "motionbricks/data/scene_geometry_field.py",
    "motionbricks/data/bvh_heightfield.py", "motionbricks/mb_sonic/sphere_bvh_heightfield.py",
    "motionbricks/data/soma_static_execution.py", "motionbricks/data/soma_cuda_graph.py",
    "motionbricks/motionlib/core/motion_reps/motion_reps_base/seperate_root_local_body.py",
    "motionbricks/motionlib/core/motion_reps/tools/motion_features.py",
    "motionbricks/motionlib/core/utils/rotations.py", "motionbricks/motionlib/core/utils/torch_utils.py",
)


def identity(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"Asset SHA256 differs: {path}")
    return dict(path=str(path), sha256=actual, bytes=path.stat().st_size)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def percentiles(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty one-dimensional timing samples required")
    return dict(count=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                min_ms=float(values.min()), max_ms=float(values.max()), mean_ms=float(values.mean()))


def gpu_snapshot():
    result = {}
    for name, query in (
        ("gpus", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu"),
        ("processes", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory"),
    ):
        call = subprocess.run(["nvidia-smi", query, "--format=csv"], check=True,
                              capture_output=True, text=True, errors="replace")
        result[name] = call.stdout.strip()
    return result


def load_scene(path):
    with np.load(path, allow_pickle=False) as data:
        scene = {key: data[key].copy() for key in
                 ("heightfield_y_m", "min_point_world_xyz_m", "dx_m")}
        if "heightfield_valid" in data:
            scene["heightfield_valid"] = data["heightfield_valid"].copy()
        seed = torch.from_numpy(data["initial_history"].copy()).float()
    if seed.shape != (1, 4, 366):
        raise ValueError("A genuine four-frame batch-one scene seed is required")
    return scene, seed


def load_actor(args):
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    flat_receipt = json.loads((args.flat_scene.parent / "acceptance.json").read_text(encoding="utf-8"))
    bundle_id = identity(args.bundle, flat_receipt["bundle_sha256"])
    parent_id = identity(args.parent, catalog["checkpoint_sha256"])
    bundle = torch.load(args.bundle, map_location="cpu", mmap=True, weights_only=False)
    parent = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=False)
    stats = Stats()
    stats.register_from_tensors(bundle["stats_mean"].clone(), bundle["stats_std"].clone())
    rep = GlobalRootGlobalJoints(30., SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats)
    torch.manual_seed(args.seed)
    actor = FrameActor(rep).eval()
    loaded = load_environment_encoder(actor, parent["model_state"], freeze=True)
    actor.requires_grad_(False)
    return rep.to(args.device), actor.to(args.device), dict(
        bundle=bundle_id, parent=parent_id, environment_keys=list(loaded),
        actor_initialization="random_root_pose_pretrained_parent_environment_only",
        seed=args.seed, actor_parameters=sum(p.numel() for p in actor.parameters()),
        fixed_statistics="original bundle dual stats subset via native global representation",
        stats_dimensions=int(rep.stats.mean.numel()), actor_trained=False)


def cases(args, rep):
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    entry = next(entry for entry in catalog["entries"] if entry["id"] == args.rough_map)
    rough_path = args.catalog.parent / entry["artifact"] / "scene.npz"
    flat_receipt = json.loads((args.flat_scene.parent / "acceptance.json").read_text(encoding="utf-8"))
    flat_id = identity(args.flat_scene, flat_receipt["scene_sha256"])
    rough_id = identity(rough_path, entry["scene_sha256"])
    flat, flat_seed = load_scene(args.flat_scene)
    rough, rough_seed = load_scene(rough_path)
    yield "existing_flat", flat, None, flat_seed, dict(asset=flat_id, role="existing_64m_authored_acceptance_plane")
    yield "pfnn_rough_" + args.rough_map, rough, None, rough_seed, dict(asset=rough_id, role="existing_pfnn_heightfield")
    for name, spawn_y in (("low_ceiling_uneven", 0.), ("side_wall_step", 0.),
                          ("bridge", 0.), ("bridge", 2.15)):
        manifest = example_scene_manifest(name)
        seed = flat_seed.clone().to(args.device)
        translation = seed.new_tensor([0., spawn_y - float(np.min(flat["heightfield_y_m"])), 0.])
        translation[[0, 2]] = -seed[0, -1, [0, 2]]
        seed[..., :3] += translation
        seed = reencode_played_history(seed, rep)
        role = name + ("_above" if spawn_y else "_below") if name == "bridge" else name
        yield role, None, manifest, seed, dict(role="authored_3d_fixture", manifest=manifest.name,
            seed_asset=flat_id, explicit_seed_translation_xyz_m=translation.cpu().tolist())


class CountedField:
    """Count actual query calls and points; do not cache any observations."""

    def __init__(self, field):
        self.field, self.calls, self.points = field, 0, 0

    def query(self, points):
        self.calls += 1
        self.points += points.numel() // 3
        return self.field.query(points)


class FrameChain:
    def __init__(self, rep, actor, field, *, execution):
        self.rep, self.actor, self.field = rep, actor, CountedField(field)
        self.sampler = VolumetricEgoSampler()
        self.root_call = self._root
        self.pose_call = self._pose
        self.advance_call = self._advance
        self.execution = execution

    def _root(self, history, command, features, distance, valid, state):
        return self.actor.root_step(history, command,
            dict(features=features, distance=distance, valid=valid), state)

    def _pose(self, history, command, features, distance, valid, delta, state, embedding):
        return self.actor.pose_step(history, command, dict(features=features, distance=distance, valid=valid),
            dict(root_delta=delta, state=state, embedding=embedding))

    def _advance(self, history, root_next, local_rot6d, contact_logits):
        return dict(history=advance(self.rep, history, dict(root_next=root_next, local_rot6d=local_rot6d,
                                                          contact_logits=contact_logits)))

    def prepare_graphs(self, history, command, state):
        from motionbricks.data.soma_cuda_graph import StaticCudaCall
        from motionbricks.data.soma_static_execution import trace_fixed_call
        coarse = self.sampler.observe(history[..., :5], command[:, :2], self.field, .8)
        root_args = (history, command, *(coarse[key] for key in ("features", "distance", "valid")), state)
        root = self._root(*root_args)
        fine = path_observation(self.sampler, history[..., :5], root["path_world_xz"], coarse, self.field)
        pose_args = (history, command, *(fine[key] for key in ("features", "distance", "valid")),
                     root["root_delta"], root["state"], root["embedding"])
        pred = self._pose(*pose_args)
        advance_args = (history, pred["root_next"], pred["local_rot6d"], pred["contact_logits"])
        for attribute, native, inputs in (("root_call", self._root, root_args),
                                          ("pose_call", self._pose, pose_args),
                                          ("advance_call", self._advance, advance_args)):
            call = StaticCudaCall(trace_fixed_call(native, inputs, cuda_constants=True))
            call(*inputs)
            setattr(self, attribute, call)

    def step(self, history, command, state, *, components=False):
        def stage(name, function):
            started = time.perf_counter_ns()
            result = function()
            if components:
                torch.cuda.synchronize(history.device)
            costs[name] = (time.perf_counter_ns() - started) / 1e6
            return result

        costs = {}
        torch.cuda.synchronize(history.device)
        started = time.perf_counter_ns()
        coarse = stage(STAGES[0], lambda: self.sampler.observe(history[..., :5], command[:, :2], self.field, .8))
        root = stage(STAGES[1], lambda: self.root_call(history, command,
            *(coarse[key] for key in ("features", "distance", "valid")), state))
        fine = stage(STAGES[2], lambda: path_observation(self.sampler, history[..., :5],
                                                      root["path_world_xz"], coarse, self.field))
        pred = stage(STAGES[3], lambda: self.pose_call(history, command,
            *(fine[key] for key in ("features", "distance", "valid")),
            root["root_delta"], root["state"], root["embedding"]))
        output = stage(STAGES[4], lambda: self.advance_call(history, pred["root_next"],
                                                          pred["local_rot6d"], pred["contact_logits"]))
        torch.cuda.synchronize(history.device)
        total = (time.perf_counter_ns() - started) / 1e6
        return output["history"], pred["state"], root, coarse, fine, dict(total=total, **costs)


def make_field(scene, manifest, args):
    field = CompositeSceneField(scene, manifest).prepare(args.device)
    if scene is not None and args.terrain_backend == "sphere_bvh" and np.ptp(scene["heightfield_y_m"]) != 0:
        from motionbricks.mb_sonic.sphere_bvh_heightfield import SphereBVHHeightfieldPrismSDF
        field.terrain = SphereBVHHeightfieldPrismSDF(scene, args.device)
        # Its exact native prism layout/bounds are unchanged by this acceleration.
    return field


def assert_parity(eager, candidate, seed, commands, actor):
    history = seed.clone()
    state = actor.initial_state(1)
    errors = []
    commands = commands.new_tensor([[0., 0., 1., 0.], [0., .8, 1., 0.],
                                     [.8, 0., 0., 1.], [0., -.8, 0., -1.]])
    for index in range(4):
        left = eager.step(history, commands[index:index + 1], state)
        right = candidate.step(history, commands[index:index + 1], state)
        for a, b in ((left[0], right[0]), (left[1], right[1]),
                     (left[2]["path_world_xz"], right[2]["path_world_xz"])):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=1e-5)
            errors.append(float((a - b).abs().max()))
        for stage in (3, 4):
            torch.testing.assert_close(left[stage]["valid"], right[stage]["valid"], atol=0, rtol=0)
        history, state = left[:2]
    return dict(input_cases=4, max_tensor_abs_error=max(errors), native_fk_and_sensing_retained=True)


def benchmark_case(chain, seed, args):
    count = args.warmup + args.frames + args.component_frames
    t = torch.arange(count, device=args.device, dtype=torch.float32) / 30
    commands = torch.stack((.35 * torch.sin(t * 1.2), .8 * torch.cos(t * .7),
                            torch.cos(t * .25), torch.sin(t * .25)), -1)
    setup = time.perf_counter()
    parity = None
    if args.execution == "graph":
        chain.prepare_graphs(seed, commands[:1], chain.actor.initial_state(1))
        eager = FrameChain(chain.rep, chain.actor, chain.field, execution="eager")
        parity = assert_parity(eager, chain, seed, commands, chain.actor)
    setup_s = time.perf_counter() - setup
    history, state = seed.clone(), chain.actor.initial_state(1)
    rows, emitted, validity = [], [], []
    queries_before = chain.field.calls, chain.field.points
    for index in range(count):
        component_pass = index >= args.warmup + args.frames
        history, state, root, coarse, fine, costs = chain.step(
            history, commands[index:index + 1], state, components=component_pass)
        # Outside measured chain: verify and retain every actual emitted frame.
        if not torch.isfinite(history).all() or not torch.isfinite(state).all():
            raise ValueError(f"Nonfinite generated rollout at frame {index}")
        emitted.append(history[:, -1].clone())
        validity.append(torch.stack((coarse["valid"].sum(), fine["valid"].sum())))
        if index >= args.warmup:
            rows.append(dict(frame=index, pass_name="components" if component_pass else "whole_chain", **costs))
    frame_tensor = torch.stack(emitted, dim=1).cpu()
    masks = torch.stack(validity).cpu()
    whole = [row for row in rows if row["pass_name"] == "whole_chain"]
    components = [row for row in rows if row["pass_name"] == "components"]
    query_calls = chain.field.calls - queries_before[0]
    query_points = chain.field.points - queries_before[1]
    if query_calls != count * 2 or query_points != count * 2048:
        raise ValueError("Complete per-frame coarse+path sensing counts differ")
    return dict(setup_seconds=setup_s, parity=parity, completed_warmup=args.warmup,
                measured_frames=len(whole), component_frames=len(components),
                sensing_audit=dict(genuine_generated_frames=count, query_calls=query_calls,
                                   queried_points=query_points, expected_points_per_frame=2048),
                synchronized_wall_ms=percentiles([row["total"] for row in whole]),
                synchronized_component_wall_ms={key: percentiles([row[key] for row in components]) for key in STAGES},
                host_dispatch_ms={key: percentiles([row[key] for row in whole]) for key in STAGES},
                components_pass_total_ms=percentiles([row["total"] for row in components]),
                valid_samples=dict(coarse_min=int(masks[:, 0].min()), fine_min=int(masks[:, 1].min()),
                                   total_per_observation=1536),
                root_displacement_xyz_m=(frame_tensor[0, -1, :3] - seed[0, -1, :3].cpu()).tolist(),
                allocated_peak_bytes=torch.cuda.max_memory_allocated(args.device),
                reserved_peak_bytes=torch.cuda.max_memory_reserved(args.device), rows=rows), frame_tensor


@torch.inference_mode()
def run(args):
    if min(args.warmup, args.frames, args.component_frames) < 100:
        raise ValueError("Use at least 100 warmup, measured and component frames")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(args.device)
    report = dict(schema="motionbricks.pfnn_frame_actor_benchmark.v1", started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
        gpu_name=torch.cuda.get_device_name(args.device), gpu_before=gpu_snapshot(),
        execution=args.execution, terrain_backend=args.terrain_backend, batch_size=1,
        pipeline=list(STAGES), query_points_per_frame=dict(coarse=1536, predicted_path=512),
        timing_scope="synchronized complete sensing+Root+path query+Pose+native FK; no HTTP/render/skin",
        component_scope="separate continued-rollout pass synchronizing each component; perturbs scheduling",
        model_scope="untrained compact actor timing with original fixed stats and trained terrain encoder",
        shared_desktop_gpu=True, actual_optimizer_updates=0, quality_approved=False, delivery_approved=False,
        source={name: identity(ROOT / name) for name in SOURCE_PATHS}, cases=[])
    report["source_sha256"] = {name: entry["sha256"] for name, entry in report["source"].items()}
    report["passed"] = False
    report["gate_scope"] = "all six complete-frame sensing+model+native-FK cases P99<=20ms; display unmeasured"
    save(args.output / "running.json", report)
    try:
        rep, actor, report["initialization"] = load_actor(args)
        for name, scene, manifest, seed, provenance in cases(args, rep):
            if args.cases and name not in args.cases:
                continue
            print(json.dumps(dict(case=name, status="preparing")), flush=True)
            torch.cuda.reset_peak_memory_stats(args.device)
            field = make_field(scene, manifest, args)
            chain = FrameChain(rep, actor, field, execution=args.execution)
            seed = seed.to(args.device)
            result, frames = benchmark_case(chain, seed, args)
            result.update(name=name, provenance=provenance)
            torch.save(dict(initial_history=seed.cpu(), generated_frames=frames,
                            actual_optimizer_updates=0, actor_trained=False), args.output / (name + "-frames.pt"))
            save(args.output / (name + ".json"), result)
            report["cases"].append(result)
            save(args.output / "running.json", report)
            print(json.dumps(dict(case=name, wall=result["synchronized_wall_ms"],
                                  components=result["synchronized_component_wall_ms"])), flush=True)
            del chain, field
            gc.collect()
        if not report["cases"]:
            raise ValueError("No requested benchmark case was selected")
        report.update(completed=True, gpu_after=gpu_snapshot(),
            p99_20ms_timing_gate=all(case["synchronized_wall_ms"]["p99_ms"] <= 20 for case in report["cases"]),
            all_declared_cases_measured=len(report["cases"]) == 6,
            finished_utc=datetime.now(timezone.utc).isoformat())
        current_sha = {name: identity(ROOT / name)["sha256"] for name in SOURCE_PATHS}
        report["sources_unchanged_during_run"] = current_sha == report["source_sha256"]
        report["passed"] = (report["p99_20ms_timing_gate"] and report["all_declared_cases_measured"]
                            and report["sources_unchanged_during_run"])
        save(args.output / "complete.json", report)
    except Exception:
        report.update(completed=False, failure=traceback.format_exc())
        save(args.output / "failure.json", report)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "out/a100_live_20261010/a100-terrain.pt")
    parser.add_argument("--catalog", type=Path, default=ROOT / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json")
    parser.add_argument("--parent", type=Path, default=ROOT / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1/007/weights.pt")
    parser.add_argument("--flat-scene", type=Path, default=ROOT / "out/pfnn_mixed_20261015/final_acceptance_flat_v1/scene.npz")
    parser.add_argument("--rough-map", default="007")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--terrain-backend", choices=("native", "sphere_bvh"), default="sphere_bvh")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--component-frames", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17022)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
