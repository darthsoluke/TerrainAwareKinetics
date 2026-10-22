"""CPU preparation by default; explicit, bounded, untrained eager capacity timing."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/scene_control_v3_performance.json"
DEFAULT_OUTPUT = ROOT / "out/scene_control_v3_20261018/performance"


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def asset_identities(config):
    identities = {}
    entries = list(config["assets"].items()) + [(s["id"], s) for s in config["scenes"] if "path" in s]
    for name, asset in entries:
        identity = file_identity(ROOT / asset["path"])
        if identity["sha256"] != asset["sha256"]:
            raise ValueError(f"Immutable asset SHA mismatch: {name}")
        identities[name] = identity
    return identities


def selected_capacity_evidence(config, current_code):
    selected = config["selected_backend"]
    evidence = {}
    for name in ("cpu_evidence", "gpu_pilot_evidence"):
        spec = selected[name]
        identity = file_identity(ROOT / spec["path"])
        if identity["sha256"] != spec["sha256"]:
            raise ValueError(f"Selected capacity evidence SHA mismatch: {name}")
        evidence[name] = dict(identity=identity, receipt=json.loads(Path(identity["path"]).read_text()))
    cpu = evidence["cpu_evidence"]["receipt"]
    selected_cases = [c for c in cpu["cases"] if c["hidden_dim"] == selected["hidden_dim"]]
    hotpaths = [name for name in current_code if name.startswith("motionbricks/")]
    matches = {name: current_code[name]["sha256"] == cpu["code"][name]["sha256"] for name in hotpaths}
    current_evidence = all(matches.values())
    gpu = evidence["gpu_pilot_evidence"]["receipt"]
    return dict(selected_backend=selected, frozen_hot_path_sha_matches=matches,
        evidence_scope="matching_current_hot_path" if current_evidence else "historical_prior_hot_path",
        cpu_evidence=evidence["cpu_evidence"]["identity"],
        cpu_all_cases_complete=cpu["all_requested_cases_complete"], cpu_case_count=len(cpu["cases"]),
        cpu_worst_p99_ms=max(c["wall_ms"]["total"]["p99_ms"] for c in cpu["cases"]),
        selected_worst_p99_ms=max(c["wall_ms"]["total"]["p99_ms"] for c in selected_cases),
        selected_cases=[dict(scene_id=c["scene_id"], parameter_count=c["parameter_count"],
            warmup=c["completed_warmup"], measured=c["completed_steps"], wall_ms=c["wall_ms"],
            fullskin=c["fullskin"], throughput_fps=c["measured_loop_fps"]) for c in selected_cases],
        gpu_pilot_evidence=evidence["gpu_pilot_evidence"]["identity"],
        gpu_pilot_p99_ms=gpu["cases"][0]["wall_ms"]["total"]["p99_ms"],
        necessary_compute_budget_met=(max(c["wall_ms"]["total"]["p99_ms"] for c in selected_cases)
            <= config["target"]["necessary_inference_p99_ms"]) if current_evidence else None,
        capacity_envelope_accepted_by_parent=bool(current_evidence and selected["capacity_envelope_accepted_by_parent"]),
        command_associated_display_p95_ms=None, browser_evidence="pending_parent_measurement",
        quality_response="unmeasured", quality_approved=False, optimizer_updates=0)


def environment():
    return dict(python_executable=sys.executable, python_version=sys.version,
        platform=platform.platform(), pid=os.getpid(), parent_pid=os.getppid(),
        cpu_model=platform.processor(), logical_cpu_count=os.cpu_count(),
        command_line=[sys.executable, *sys.argv], cwd=str(Path.cwd()),
        torch_version=torch.__version__, torch_cuda_build=torch.version.cuda,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        numpy_version=np.__version__,
        warp_version=importlib.metadata.version("warp-lang"),
        trimesh_version=importlib.metadata.version("trimesh"),
        cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        pythonpath=os.environ.get("PYTHONPATH"))


def desktop_gpu_snapshot():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    commandlines = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|chrome|msedge|codex|nvidia' } | "
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine | ConvertTo-Json -Compress"],
        check=True, capture_output=True, text=True)
    return dict(gpu_csv=gpu.stdout.strip(), compute_process_csv=processes.stdout.strip(),
                process_identity=json.loads(commandlines.stdout),
                wddm_note="Shared desktop graphics and WDDM process memory may be unreported by compute-apps")


def percentiles(samples):
    values = np.asarray(samples, dtype=np.float64)
    return dict(count=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                min_ms=float(values.min()), max_ms=float(values.max()), mean_ms=float(values.mean()))


def native_source_proof(adapter):
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.scene_control_v2.provider import load_bound_world
    source_config = ROOT / "configs/scene_control_v2_real_provider_cpu.json"
    spec = json.loads(source_config.read_text())["provider"]
    world, joints, _, _, lineage = load_bound_world(spec, adapter.representation)
    frame = adapter.source_frame(world[[0, 315, 645]][None])
    reference = load_official_skinner(adapter.representation.skeleton)
    official = reference.skin(frame["global_joint_rots"][0], frame["joints"][0], rot_is_global=True)
    actual = adapter.fullskin(frame["rig"])[0]
    error = float((actual - official).abs().max())
    surface_reference = actual.index_select(-2, adapter.surface_sampler.selected_vertex_ids)
    surface_error = float((frame["body_points"][0] - surface_reference).abs().max())
    if error > 2e-6:
        raise ValueError("Official full18056 source skin parity failed")
    if surface_error > 3e-5:
        raise ValueError("Sparse official512 source surface parity failed")
    return dict(frames=len(world), source_frame_indices=[0, 315, 645], vertices=actual.shape[-2],
        fullskin_max_abs_m=error,
        selected_surface_max_abs_m=surface_error, selected_surface_vertices=512,
        source_joint_max_abs_m=float((frame["joints"][0] - joints[[0, 315, 645]]).abs().max()),
        root_max_abs_m=float((frame["root"][0] - world[[0, 315, 645], :3]).abs().max()),
        source_config=file_identity(source_config), lineage=lineage,
        optimizer_updates=0, quality_approved=False, role="native_source_transport_parity_not_actor_quality")


def cpu_preflight(output):
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_scene_control_v3_native.py",
               "tests/test_scene_control_v3_model.py", "tests/test_scene_control_v3_state.py",
               "tests/test_scene_control_v3_geometry.py", "tests/test_scene_control_v3_surface.py",
               "tests/test_scene_control_v3_server.py"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", SCENE_CONTROL_V3_CUDA_ALLOCATED="0",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    report = dict(command=command,
                  returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, cpu_threads=2)
    (output / "native_cpu_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if result.returncode:
        raise RuntimeError("New native CPU preflight failed; see native_cpu_preflight.json")
    return report


def memory_snapshot(device):
    import warp as wp
    free, total = torch.cuda.mem_get_info(device)
    return dict(device_free_bytes=free, device_total_bytes=total,
        cuda_driver_memory_scope="driver_view_WDDM_not_nvidia_smi_physical_memory",
        device_used_bytes=total-free,
        torch_allocated_bytes=torch.cuda.memory_allocated(device),
        torch_reserved_bytes=torch.cuda.memory_reserved(device),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        warp_mempool_used_current_bytes=wp.get_mempool_used_mem_current(device),
        warp_mempool_used_high_bytes=wp.get_mempool_used_mem_high(device))


class CapacityCase:
    def __init__(self, adapter, actor, scene, scene_spec, device):
        from motionbricks.scene_control_v3.pipeline import CapacityPipeline
        history = adapter.initial_history
        if "path" in scene_spec:
            with np.load(ROOT / scene_spec["path"], allow_pickle=False) as archive:
                history = torch.from_numpy(archive["initial_history"].copy()).to(device)
        velocity = history.new_zeros(1, scene.object_count, 6)
        if scene_spec["id"] == "true3d_clutter":
            velocity[:, 1, 0] = 0.06
        self.pipeline = CapacityPipeline(adapter, actor, scene, history, object_velocity=velocity)

    @property
    def history(self):
        return self.pipeline.state.history

    @property
    def hidden(self):
        return self.pipeline.state.hidden

    @property
    def pose(self):
        return self.pipeline.state.object_pose

    def run_step(self, cuda):
        stamps = [time.perf_counter()]
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)] if cuda else []
        if cuda:
            events[0].record()

        def stage_completed(_):
            if cuda:
                events[len(stamps)].record()
            stamps.append(time.perf_counter())

        phase = self.pipeline.state.frame_id / 30
        command = (0.45 * math.sin(phase), 0.8, 0.25 * math.cos(phase))
        candidate = self.pipeline.propose(command, stage_callback=stage_completed)
        self.pipeline.commit(candidate.frame_id)
        self.last_query_metadata = candidate.query_metadata
        stamps.append(time.perf_counter())
        stages = ("contact_cpu_admit_pack_and_transfers", "query_and_point_construction", "encode", "actor",
                  "native_rig_and_history", "transport_validity_and_commit")
        wall = {name: (stamps[i+1] - stamps[i]) * 1000 for i, name in enumerate(stages)}
        wall["total"] = (stamps[-1] - stamps[0]) * 1000
        gpu = {name: events[i].elapsed_time(events[i+1]) for i, name in enumerate(stages[:5])} if cuda else None
        return wall, gpu, candidate.native_frame, len(candidate.payload), candidate.query, candidate.scene_valid


def fullskin_measurement(adapter, frame, cuda):
    from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
    reference = move_skinner(load_official_skinner(adapter.representation.skeleton), frame["rig"].device)
    reference.skeleton_skin.to(frame["rig"].device)
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    vertices = adapter.fullskin(frame["rig"])
    if cuda:
        torch.cuda.synchronize()
    duration = (time.perf_counter() - start) * 1000
    expected = reference.skin(frame["global_joint_rots"][0], frame["joints"][0], rot_is_global=True)
    error = float((vertices[0] - expected).abs().max().cpu())
    surface_reference = vertices.index_select(-2, adapter.surface_sampler.selected_vertex_ids)
    surface_error = float((frame["body_points"] - surface_reference).abs().max().cpu())
    if error > 2e-5:
        raise ValueError("Official full18056 capacity rig parity failed")
    if surface_error > 3e-5:
        raise ValueError("Sparse official512 capacity surface parity failed")
    return dict(fullskin_ms=duration, vertices=vertices.shape[-2], fullskin_max_abs_m=error,
                selected_surface_max_abs_m=surface_error, selected_surface_vertices=512,
                measured_separately_from_rig_pipeline=True, all_official_influences=True)


@torch.inference_mode()
def benchmark(config, args, receipt, output):
    import warp as wp
    from motionbricks.scene_control_v3 import CausalSceneActor, SceneActorConfig
    from motionbricks.scene_control_v3.geometry import build_pfnn_scene, build_true3d_scene
    from motionbricks.scene_control_v3 import geometry
    from motionbricks.scene_control_v3.native import NativeSOMAAdapter
    cuda = args.mode == "cuda"
    device = "cuda:0" if cuda else "cpu"
    if cuda:
        receipt["desktop_before"] = desktop_gpu_snapshot()
        gpu0 = next(row for row in csv.reader(io.StringIO(receipt["desktop_before"]["gpu_csv"]))
                    if int(row[0]) == 0)
        receipt["allocated_gpu_baseline"] = dict(uuid=gpu0[1].strip(), free_mib=int(gpu0[6]),
                                                 utilization_percent=int(gpu0[7]))
        if int(gpu0[6]) < 3072 or int(gpu0[7]) > 40:
            raise RuntimeError("Allocated shared-desktop baseline requires >=3072 MiB free and <=40% utilization; await parent snapshot")
        torch.cuda.set_per_process_memory_fraction(config["benchmark"]["torch_memory_fraction"], device)
        props = torch.cuda.get_device_properties(device)
        receipt["device"] = dict(name=props.name, total_memory=props.total_memory,
                                 capability=list(torch.cuda.get_device_capability(device)))
        if props.name != config["target"]["gpu"]:
            raise ValueError("This receipt requires the target RTX3080; no device substitution")
    wp.init()
    options = wp.get_module_options(geometry)
    receipt["geometry_compile_options"] = {key: options[key] for key in ("enable_backward", "fast_math", "fuse_fp")}
    receipt["profile_backend"] = device
    receipt["memory_before_assets"] = memory_snapshot(device) if cuda else None
    adapter = NativeSOMAAdapter(ROOT / config["assets"]["bundle"]["path"], device)
    torch.manual_seed(config["seed"])
    started = time.perf_counter()
    deadline = started + args.max_seconds
    receipt["cases"] = []
    for scene_spec in config["scenes"]:
        if scene_spec["id"] not in args.scenes:
            continue
        cold = time.perf_counter()
        receipt["progress"] = dict(scene_id=scene_spec["id"], phase="geometry_cold_build")
        before = memory_snapshot(device) if cuda else None
        scene = (build_pfnn_scene(ROOT / scene_spec["path"], device=device) if "path" in scene_spec
                 else build_true3d_scene(device=device))
        if cuda:
            torch.cuda.synchronize()
        scene_memory = memory_snapshot(device) if cuda else None
        cold_ms = (time.perf_counter() - cold) * 1000
        for width in args.hidden_dims:
            receipt["progress"] = dict(scene_id=scene_spec["id"], hidden_dim=width, phase="actor_and_initial_state")
            torch.manual_seed(config["seed"])
            actor = CausalSceneActor(SceneActorConfig(hidden_dim=width,
                preview_frames=config["benchmark"]["preview_frames"])).to(device).eval().requires_grad_(False)
            case = CapacityCase(adapter, actor, scene, scene_spec, device)
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            result = dict(scene_id=scene_spec["id"], hidden_dim=width,
                parameter_count=sum(p.numel() for p in actor.parameters()),
                preview_frames=actor.config.preview_frames, preview_units="30Hz_frames_not_VQ_tokens",
                actor_status=actor.probe_status, classification=actor.capacity_scope,
                raw_global366_conditioning=True, geometry=scene.storage_summary,
                actual_dtypes=dict(actor=str(next(actor.parameters()).dtype), history=str(case.history.dtype),
                    object_pose=str(case.pose.dtype), official_bind_vertices=str(adapter.skinner.bind_vertices.dtype),
                    official77_neutral_joints=str(adapter.skinner.skeleton_skin.neutral_joints.dtype)),
                geometry_cold_build_ms=cold_ms, memory_before_scene=before, memory_after_scene=scene_memory,
                requested_warmup=args.warmup, requested_steps=args.steps,
                backend=device,
                completed_warmup=0, completed_steps=0, complete=False,
                query_policy=case.pipeline.contract()["query_policy"],
                query_path="256_current_surface_128_previous_committed_Root_preview_128_latest_intent",
                body_part_id_semantics="opaque_sampler_slot_mod15_capacity_grouping_not_bound_contact_labels",
                contact_observations="synthetic_CPU_capacity_fixture_15slots_active_release_rebind_not_GT_quality",
                contact_proposals="previous_actor_logits_separate_from_factual_capacity_observations",
                timed_scope="contact_admit_pack_D2H_H2D_point_construction_query_encode_actor_native_validity_pack_commit")
            result["memory_samples"] = []
            receipt["cases"].append(result)
            wall_samples, gpu_samples = [], []
            result["wall_samples_ms"] = wall_samples
            result["gpu_stream_samples_ms"] = gpu_samples
            receipt["progress"] = dict(scene_id=scene_spec["id"], hidden_dim=width, phase="warmup")
            for _ in range(args.warmup):
                if time.perf_counter() >= deadline:
                    break
                case.run_step(cuda)
                result["completed_warmup"] += 1
            receipt["progress"]["phase"] = "measured_steps"
            measured_started = time.perf_counter()
            for index in range(args.steps):
                if time.perf_counter() >= deadline:
                    break
                wall, gpu, frame, nbytes, query, valid = case.run_step(cuda)
                wall_samples.append(wall)
                if gpu is not None:
                    gpu_samples.append(gpu)
                result["completed_steps"] += 1
                if cuda and index % 30 == 0:
                    result["memory_samples"].append(memory_snapshot(device))
            measured_elapsed = time.perf_counter() - measured_started
            result["measured_loop_elapsed_seconds"] = measured_elapsed
            result["measured_loop_fps"] = result["completed_steps"] / measured_elapsed
            result["complete"] = result["completed_steps"] == args.steps and result["completed_warmup"] == args.warmup
            if wall_samples:
                result["wall_ms"] = {key: percentiles([s[key] for s in wall_samples]) for key in wall_samples[0]}
                result["gpu_stream_ms"] = ({key: percentiles([s[key] for s in gpu_samples]) for key in gpu_samples[0]}
                                           if gpu_samples else None)
                result["last_frame_bytes"] = nbytes
                result["query_metadata_last"] = case.last_query_metadata
                result["actual_dtypes"].update(root=str(frame["root"].dtype), rig=str(frame["rig"].dtype),
                                              hidden=str(case.hidden.dtype))
                result["last_valid_queries"] = int(valid.sum().cpu())
                result["last_query_signed_count"] = int((query["distance_signed"] & query["valid"]).sum().cpu())
                receipt["progress"]["phase"] = "independent_fullskin_parity"
                result["fullskin"] = [fullskin_measurement(adapter, frame, cuda)
                                      for _ in range(config["benchmark"]["fullskin_samples_per_scene"])]
                result["necessary_latency_budget_met"] = bool(
                    result["wall_ms"]["total"]["p99_ms"] <= config["target"]["necessary_inference_p99_ms"])
                result["statistically_qualified"] = bool(result["complete"] and
                    result["completed_warmup"] >= config["benchmark"]["qualification_min_warmup"] and
                    result["completed_steps"] >= config["benchmark"]["qualification_min_measured_steps"])
                result["qualification"] = "capacity_statistics" if result["statistically_qualified"] else "few_samples_not_qualified"
                result["compute_throughput_at_least30fps"] = result["measured_loop_fps"] >= config["target"]["fps"]
            result["memory_after_case"] = memory_snapshot(device) if cuda else None
            if cuda:
                samples = [before, scene_memory, result["memory_after_case"], *result["memory_samples"]]
                result["minimum_observed_cuda_driver_free_bytes"] = min(s["device_free_bytes"] for s in samples)
                result["whole_device_memory_scope"] = "nvidia_smi_desktop_before_after_is_physical_driver_samples_are_distinct_WDDM_view"
                result["scene_non_torch_reserved_delta_bytes"] = ((before["device_free_bytes"] - scene_memory["device_free_bytes"])
                    - (scene_memory["torch_reserved_bytes"] - before["torch_reserved_bytes"]))
                result["non_torch_delta_scope"] = "Warp_BVH_context_driver_and_concurrent_desktop_not_isolated_BVH_bytes"
            (output / "progress.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            del case, actor
            gc.collect()
            if time.perf_counter() >= deadline:
                break
        del scene
        gc.collect()
        if time.perf_counter() >= deadline:
            break
    receipt["elapsed_benchmark_seconds"] = time.perf_counter() - started
    expected_cases = len(args.scenes) * len(args.hidden_dims)
    receipt["all_requested_cases_complete"] = len(receipt["cases"]) == expected_cases and all(c["complete"] for c in receipt["cases"])
    receipt["memory_after_benchmark"] = memory_snapshot(device) if cuda else None
    receipt["desktop_after"] = desktop_gpu_snapshot() if cuda else None
    eligible = [c for c in receipt["cases"] if c.get("necessary_latency_budget_met")
                and c.get("statistically_qualified") and c.get("compute_throughput_at_least30fps")]
    widths = [w for w in args.hidden_dims if sum(c["hidden_dim"] == w for c in eligible) == len(args.scenes)]
    receipt["pareto_budget_eligible_hidden_dims"] = widths
    receipt["largest_capacity_within_latency_budget"] = max(widths) if widths and receipt["all_requested_cases_complete"] else None
    receipt["deployment_capacity_selected"] = False
    receipt["progress"] = dict(phase="complete" if receipt["all_requested_cases_complete"] else "bounded_incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("prepare", "cpu", "cuda"), default="prepare")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allocation-id", help="Parent's explicit GPU reservation identity; required for CUDA")
    parser.add_argument("--scenes", nargs="+", choices=("pfnn011", "pfnn007", "flat", "true3d_clutter"))
    parser.add_argument("--hidden-dims", nargs="+", type=int, choices=(256, 512, 1024))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.mode != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = json.loads(args.config.read_text())
    if args.mode == "cuda" and not args.allocation_id:
        parser.error("CUDA requires --allocation-id from the parent; default preparation is CPU-only")
    args.hidden_dims = args.hidden_dims or config["benchmark"]["hidden_dims"]
    args.scenes = args.scenes or (["true3d_clutter"] if args.mode == "cpu" else [s["id"] for s in config["scenes"]])
    args.warmup = args.warmup if args.warmup is not None else (2 if args.mode == "cpu" else config["benchmark"]["warmup_steps_per_scene"])
    args.steps = args.steps if args.steps is not None else (5 if args.mode == "cpu" else config["benchmark"]["measured_steps_per_scene"])
    args.max_seconds = args.max_seconds if args.max_seconds is not None else config["benchmark"]["max_cuda_seconds"]
    if not 0 <= args.warmup <= 100 or not 1 <= args.steps <= 300 or not 0 < args.max_seconds <= 180:
        parser.error("Bounds: warmup0..100, steps1..300, seconds(0,180]")
    if len(set(args.scenes)) != len(args.scenes) or len(set(args.hidden_dims)) != len(args.hidden_dims):
        parser.error("Duplicate cases are not permitted")
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or DEFAULT_OUTPUT / f"{args.mode}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="motionbricks.scene_control_v3.performance_receipt.v1", mode=args.mode,
        timestamp_utc=stamp, classification="architecture_capacity_only", actor_status="untrained_capacity_probe",
        optimizer_updates=0, new_training=False, old_quality_audits_rerun=False, quality_approved=False,
        learned_wasd_usable=False, quality_response="unmeasured", command_associated_display_p95_ms=None,
        display_measurement="not_measured_requires_command_id_correlated_browser_ack",
        execution="eager", cuda_graph_comparison=None,
        transport_scope=config["benchmark"]["transport_scope"],
        baseline=config["benchmark"]["baseline"], allocation_id=args.allocation_id,
        seed=config["seed"], config=file_identity(args.config), requested_scenes=args.scenes,
        requested_hidden_dims=args.hidden_dims, status="running")
    started = time.perf_counter()
    try:
        receipt["progress"] = dict(phase="environment_and_immutable_assets")
        receipt["environment"] = environment()
        receipt["selected_backend"] = config["selected_backend"]
        receipt["query_policy"] = config["query_policy"]
        receipt["assets"] = asset_identities(config)
        receipt["code"] = {name: file_identity(ROOT / name) for name in (
            "motionbricks/scene_control_v3/native.py", "motionbricks/scene_control_v3/model.py",
            "motionbricks/scene_control_v3/contracts.py", "motionbricks/scene_control_v3/state.py",
            "motionbricks/scene_control_v3/geometry.py", "motionbricks/scene_control_v3/pipeline.py",
            "motionbricks/scene_control_v3/surface.py",
            "scripts/benchmark_scene_control_v3.py")}
        receipt["capacity_evidence"] = selected_capacity_evidence(config, receipt["code"])
        receipt["server_and_frontend_code"] = {name: file_identity(ROOT / name) for name in (
            "scripts/soma_scene_control_v3_server.py", "web/scene_control_v3/index.html",
            "web/scene_control_v3/app.js", "web/scene_control_v3/styles.css",
            "web/scene_control_v3/protocol.test.mjs")}
        receipt["progress"] = dict(phase="new_v3_cpu_preflight")
        receipt["native_cpu_preflight"] = cpu_preflight(output)
        from motionbricks.scene_control_v3 import CausalSceneActor, SceneActorConfig
        receipt["capacity_profiles"] = [dict(hidden_dim=width, preview_frames=config["benchmark"]["preview_frames"],
            parameter_count=sum(p.numel() for p in CausalSceneActor(SceneActorConfig(hidden_dim=width,
                preview_frames=config["benchmark"]["preview_frames"])).parameters()))
            for width in args.hidden_dims]
        from motionbricks.scene_control_v3.native import NativeSOMAAdapter
        receipt["progress"] = dict(phase="new_native_source_transport_parity")
        with torch.inference_mode():
            adapter = NativeSOMAAdapter(ROOT / config["assets"]["bundle"]["path"])
            receipt["native_source_parity"] = native_source_proof(adapter)
            del adapter
        if args.mode != "prepare":
            benchmark(config, args, receipt, output)
        else:
            receipt["cuda_started"] = False
            receipt["performance_verdict"] = "prepared_only_no_target_gpu_measurement"
        receipt["status"] = "complete"
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started, progress=receipt.get("progress"),
            completed_warmup=sum(c["completed_warmup"] for c in receipt.get("cases", [])),
            completed_steps=sum(c["completed_steps"] for c in receipt.get("cases", [])))
        receipt["status"], receipt["failure"] = "failed", failure
        (output / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(dict(failure=str(output / "failure.json"), partial_receipt=str(output / "receipt.json"))), flush=True)
        raise
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(dict(receipt=str(output / "receipt.json"), mode=args.mode,
                         classification=receipt["classification"], optimizer_updates=0,
                         quality_approved=False)), flush=True)


if __name__ == "__main__":
    main()
