"""Run official PARC autoregressive generation on a continuous authored course.

Every output is a kinematic candidate, not a physics-recorded training sample.
Only scoring is frame-chunked, preserving the upstream additive objective.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

import run_parc_local_terrain_pilot as pilot
from build_parc_stair_course import build_course, save_course


def frame_intent_from_calls(calls, candidate_id, frame_count, seq_len, history, rewind):
    """Align commands issued before diffusion with upstream retained chunks."""
    first = seq_len - 1 - rewind
    stride = first - history
    if first <= 0 or stride <= 0 or not calls:
        raise ValueError("Invalid upstream chunk contract")
    chunks = []
    for index, call in enumerate(calls):
        length = first if index == 0 else stride
        # Only the last call can provide the timeout tail instead of a short chunk.
        if index == len(calls) - 1 and index > 0:
            length = seq_len - history
        chunks.append(np.repeat(call[candidate_id][None, :], length, axis=0))
    result = np.concatenate(chunks, axis=0)
    if len(result) < frame_count:
        raise ValueError("Command log is shorter than the generated motion")
    return result[:frame_count]


def chunked_motion_score(original, motion_frames, path_nodes, terrain, char_model,
                         body_points, w_contact, w_pen, w_path, verbose=True,
                         frame_chunk=8, batch_chunk=8):
    """The official score sums independent frame terms; never omit a frame."""
    if frame_chunk < 1 or batch_chunk < 1:
        raise ValueError("frame_chunk and batch_chunk must be positive")
    totals = None
    for start in range(0, motion_frames.root_pos.shape[1], frame_chunk):
        chunk = motion_frames.get_slice(slice(start, start + frame_chunk))
        if chunk.root_pos.shape[0] <= batch_chunk:
            row = original(chunk, path_nodes, terrain, char_model, body_points,
                           w_contact, w_pen, w_path, verbose=False)
        else:
            rows = [original(chunk.get_idx(slice(begin, begin + batch_chunk)), path_nodes,
                             terrain, char_model, body_points, w_contact, w_pen, w_path, verbose=False)
                    for begin in range(0, chunk.root_pos.shape[0], batch_chunk)]
            row = {key: torch.cat([part[key] for part in rows], dim=0) for key in rows[0]}
        totals = row if totals is None else {k: totals[k] + row[k] for k in row}
    if totals is None:
        raise ValueError("Cannot score empty motion")
    return totals


def passage_evidence(foot_centers, terrain, course):
    """Measure ordered summit visits, without asserting support or physicality."""
    xy = foot_centers[:, :2]
    heights = terrain.get_hf_val_from_points(xy).detach().cpu().numpy()
    pos = foot_centers.detach().cpu().numpy()
    rows, next_frame = [], 0
    for flight in course["stair_flights"]:
        a = np.asarray(flight["ascent_end_xyz_m"])
        b = np.asarray(flight["platform_end_xyz_m"])
        axis = 0 if flight["orientation"] == "x" else 1
        lateral = 1 - axis
        in_platform = ((pos[:, axis] >= a[axis]) & (pos[:, axis] <= b[axis])
                       & (np.abs(pos[:, lateral] - a[lateral]) <= flight["width_m"] / 2)
                       & np.isclose(heights, flight["total_rise_m"], atol=1e-4)
                       & (np.abs(pos[:, 2] - heights) < .35))
        valid = np.flatnonzero(in_platform & (np.arange(len(pos)) >= next_frame))
        first = int(valid[0]) if len(valid) else None
        rows.append({"flight": flight["flight_id"], "summit_visited_in_order": first is not None,
                     "first_summit_frame": first})
        if first is not None:
            next_frame = first + 1
    return {"all_summits_visited_in_order": all(r["summit_visited_in_order"] for r in rows),
            "flights": rows, "scope": "geometric passage only, not valid contact or physics"}


def route_evidence(foot_centers, nodes, course, xy_tolerance=.6, z_tolerance=.35):
    """Require dense directed milestones on every segment, including descents.

    These are spatial coverage observations, never a contact/physics certificate.
    Each milestone must consume a later frame, rejecting isolated summit samples.
    """
    pos = foot_centers.detach().cpu().numpy()
    path = nodes.detach().cpu().numpy()
    rows, cursor = [], 0
    for segment in course["route_segments"]:
        start, end = segment["path_node_start"], segment["path_node_end"]
        count = max(2, int(np.ceil(np.linalg.norm(path[end, :2] - path[start, :2]) / .6)) + 1)
        indices = np.unique(np.rint(np.linspace(start, end, count)).astype(int))
        visits = []
        for index in indices:
            valid = np.flatnonzero((np.linalg.norm(pos[:, :2] - path[index, :2], axis=1) <= xy_tolerance)
                                   & (np.abs(pos[:, 2] - path[index, 2]) <= z_tolerance)
                                   & (np.arange(len(pos)) >= cursor))
            frame = int(valid[0]) if len(valid) else None
            visits.append({"path_node": int(index), "frame": frame})
            if frame is not None:
                cursor = frame + 1
        rows.append({"name": segment["name"], "stage": segment["stage"],
                     "covered_in_order": all(v["frame"] is not None for v in visits), "milestones": visits})
    return {"all_segments_covered_in_order": all(r["covered_in_order"] for r in rows),
            "xy_tolerance_m": xy_tolerance, "z_tolerance_m": z_tolerance,
            "scope": "ordered geometric milestones only; not support, traversability or physics approval",
            "segments": rows}


def official_generation_settings(repo, path_class, gen_class, batch_size, max_seconds):
    config_path = repo / "data/configs/parc_2_kin_gen_default.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    path_settings, gen_settings = path_class(), gen_class()
    for key, value in cfg["mdm_path"].items():
        setattr(path_settings, key, value)
    for key, value in cfg["mdm_gen"].items():
        setattr(gen_settings, key, value)
    path_settings.mdm_batch_size = batch_size
    path_settings.max_motion_length = max_seconds
    return path_settings, gen_settings, config_path


def run(args):
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    sys.modules.setdefault("parc", importlib.import_module("PARC"))
    from parc.motion_synthesis.procgen import mdm_path
    from parc.motion_generator import gen_util
    from parc.util import file_io_helper

    os.environ["WANDB_MODE"] = "disabled"
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = pilot.load_official_model(repo, args.checkpoint.resolve(), args.device, weight_key="ema_model")
    if args.kind == 'diverse':
        from build_parc_diverse_course import build_diverse_course
        terrain, nodes, course = build_diverse_course(json.loads(args.course_spec.read_text()), args.device)
    elif args.kind == 'long_mixed':
        from build_parc_long_mixed_course import build_long_mixed_course
        terrain,nodes,course=build_long_mixed_course(args.seed,args.course_sections,args.variant,args.device)
    elif args.kind == "mixed":
        from build_parc_mixed_course import build_mixed_course
        terrain, nodes, course = build_mixed_course(args.device, args.variant, args.seed)
    elif args.kind == "platform":
        from build_parc_stair_course import build_platform_course
        terrain, nodes, course = build_platform_course(args.device, args.variant)
    elif args.kind == "extension":
        from build_parc_platform_extensions import build_extension_course
        terrain, nodes, course = build_extension_course(args.device, args.variant)
    else:
        terrain, nodes, course = build_course(args.device)
    save_course(output / "course", terrain, nodes, course)
    expected_geometry = getattr(args, 'expected_terrain_sha256', None)
    if expected_geometry and course.get('terrain_sha256') != expected_geometry:
        raise ValueError('Generation terrain differs from the versioned plan preflight')
    path_settings, gen_settings, generation_config = official_generation_settings(
        repo, mdm_path.MDMPathSettings, gen_util.MDMGenSettings, args.batch_size, args.max_seconds)
    if args.sampling_profile == 'official_eval':
        eval_config = yaml.safe_load((repo/'data/configs/parc_1_train_gen_default.yaml').read_text())
        gen_settings.ddim_stride = int(eval_config['test_ddim_stride'])
    original_score = mdm_path.compute_motion_loss
    mdm_path.compute_motion_loss = lambda *a, **kw: chunked_motion_score(
        original_score, *a, **kw, frame_chunk=args.score_frame_chunk)
    original_generate = gen_util.gen_mdm_motion
    intent_calls = []

    def observe_generate(*a, **kw):
        if a:
            raise ValueError("Upstream command capture requires explicit generation arguments")
        transform=getattr(args,'command_transform',None)
        if transform is not None:
            kw=transform(kw)
        root_xy = kw['prev_frames'].root_pos[:, -1, :2]
        delta = kw['target_world_pos'][..., :2] - root_xy
        direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        intent_calls.append(direction.detach().cpu().numpy().copy())
        return original_generate(**kw)

    gen_util.gen_mdm_motion = observe_generate
    metadata_path = output / "generation.json"
    sources = [Path(__file__), Path(pilot.__file__), Path(__file__).with_name("build_parc_stair_course.py")]
    if args.kind == "extension":
        sources.append(Path(__file__).with_name("build_parc_platform_extensions.py"))
    if args.kind == "mixed":
        sources.append(Path(__file__).with_name("build_parc_mixed_course.py"))
    if args.kind == 'diverse':
        sources.extend([Path(__file__).with_name('build_parc_diverse_course.py'),
                        Path(__file__).with_name('build_parc_mixed_course.py')])
    if args.kind == 'long_mixed':
        sources.extend([Path(__file__).with_name('build_parc_long_mixed_course.py'),
                        Path(__file__).with_name('build_parc_mixed_course.py')])
    report = {
        "status": "generating", "seed": args.seed, "batch_size": args.batch_size,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": pilot.sha256(args.checkpoint),
        "model": "official legacy demo april272025/iter_3/model_20000, format-adapted without training",
        "weight_selection": "ema_model, selected by legacy official runner when use_ema is enabled",
        "source_sha256": {p.name: pilot.sha256(p) for p in sources},
        "vendor_runtime_sha256": {name: pilot.sha256(repo / "PARC" / name) for name in (
            "motion_synthesis/procgen/mdm_path.py", "motion_generator/gen_util.py", "motion_generator/mdm.py")},
        "generation_config": str(generation_config), "generation_config_sha256": pilot.sha256(generation_config),
        "generation_overrides": {"mdm_batch_size": args.batch_size, "max_motion_length": args.max_seconds,
                                 "add_noise_to_loss": False},
        "sampling_profile": args.sampling_profile,
        "sampling_profile_note": "official_eval uses the released training evaluation DDIM stride; not a validated replacement for the data-generation profile",
        "course_kind": args.kind, "course_variant": args.variant,
        "course_distribution": course.get('distribution'),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ddim_stride": gen_settings.ddim_stride, "diffusion_timesteps": model._diffusion_timesteps,
        "cfg_scale": gen_settings.cfg_scale, "history_frames": model._num_prev_states,
        "max_seconds": args.max_seconds, "course_planar_m": course["course"]["planar_length_m"],
        "score_frame_chunk": args.score_frame_chunk, "physics_corrected": False,
        "score_batch_chunk": 8,
        "training_approved": False, "concatenated_source_clips": False, "candidates": [],
    }
    metadata_path.write_text(json.dumps(report, indent=2))
    started = time.time()
    try:
        with torch.no_grad():
            motions, terrains, info = mdm_path.generate_frames_until_end_of_path(
                path_nodes=nodes, terrain=terrain, char_model=model._kin_char_model,
                mdm_model=model, prev_frames=None, mdm_path_settings=path_settings,
                mdm_gen_settings=gen_settings, add_noise_to_loss=False,
                verbose=True, slice_terrain=False, first_heading_mode="auto")
            names = model._kin_char_model.get_body_names()
            feet = [names.index("left_foot"), names.index("right_foot")]
            for rank, (frames, candidate_terrain) in enumerate(zip(motions, terrains)):
                frames = frames.get_idx(0)
                directory = output / f"candidate_{rank:03d}"
                directory.mkdir()
                file = directory / "generated.pkl"
                command = frame_intent_from_calls(intent_calls, int(info['candidate_ids'][rank]),
                    len(frames.root_pos), model._seq_len, model._num_prev_states,
                    path_settings.rewind_num_frames)
                intent_path = directory / 'intent.npz'
                np.savez_compressed(intent_path, command_world_xy=command,
                    fps=model._sequence_fps, source_frame_indices=np.arange(len(command)),
                    provenance='target minus observed history root, captured before each diffusion call')
                file_io_helper.save_ms_file(str(file), motion_frames=frames,
                    fps=model._sequence_fps, loop_mode="CLAMP", terrain=candidate_terrain,
                    hf_mask_inds=None, body_constraints=None, min_point_offset=None, path_nodes=nodes,
                    extra_misc_data={"source": report["model"], "seed": args.seed,
                        "rank": rank, "physics_corrected": False, "training_approved": False})
                diagnostics = pilot.save_fk_world_terrain_diagnostics(
                    directory / "fk.npz", frames, candidate_terrain, model, np, torch)
                positions, _ = model._kin_char_model.forward_kinematics(
                    frames.root_pos, frames.root_rot, frames.joint_rot)
                passage = route_evidence(positions[:, feet].mean(1), nodes, course)
                row = {
                    "rank": rank, "candidate_id": int(info["candidate_ids"][rank]),
                    "frames": len(frames.root_pos), "duration_s": (len(frames.root_pos) - 1) / model._sequence_fps,
                    "endpoint_reached": bool(info["completed"][rank]), "passage": passage,
                    "weighted_contact_score": float(info["contact_losses"][rank]),
                    "weighted_penetration_score": float(info["pen_losses"][rank]),
                    "selection_score": float(info["losses"][rank]),
                    "motion": str(file), "sha256": pilot.sha256(file), "diagnostics": diagnostics,
                    "intent": str(intent_path), "intent_sha256": pilot.sha256(intent_path),
                }
                row["geometric_route_covered"] = row["endpoint_reached"] and passage["all_segments_covered_in_order"]
                report["candidates"].append(row)
                metadata_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        report["status"] = "kinematic_candidates_generated_physics_pending"
        report["geometric_coverage_candidates"] = sum(r["geometric_route_covered"] for r in report["candidates"])
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        mdm_path.compute_motion_loss = original_score
        gen_util.gen_mdm_motion = original_generate
        report["elapsed_seconds"] = time.time() - started
        metadata_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=pilot.DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=90.)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--kind", choices=("stairs", "platform", "extension", "mixed", "diverse", "long_mixed"), default="stairs")
    parser.add_argument('--course-sections',type=int,default=3)
    parser.add_argument('--sampling-profile',choices=('official_data','official_eval'),default='official_data')
    parser.add_argument('--course-spec', type=Path)
    parser.add_argument('--expected-terrain-sha256')
    parser.add_argument("--variant", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-frame-chunk", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_seconds <= 0:
        parser.error("batch size and duration must be positive")
    if args.kind == 'diverse' and args.course_spec is None:
        parser.error('--kind diverse requires --course-spec')
    run(args)
