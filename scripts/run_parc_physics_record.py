"""Run the official PARC recorder with immutable inputs and attempt provenance.

No controller, physics, motion, or retry policy is replaced. Engine origins
are read from Isaac Gym instead of inferred from its environment grid.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def validate_bundle(bundle):
    manifest = json.loads((bundle / "_bundle.json").read_text())
    seen = set()
    for row in manifest["files"]:
        p = bundle / row["path"]
        if p.resolve() != bundle.resolve().joinpath(row["path"]) or row["path"] in seen:
            raise ValueError("Escaped or duplicate bundle file")
        p.resolve().relative_to(bundle.resolve())
        if p.is_symlink() or p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
            raise ValueError("Bundle input changed: " + row["path"])
        seen.add(row["path"])
    return manifest


def install_recorder_compatibility(dm):
    """The released recorder predates DeepMimicEnv's _motion_lib -> _mlib rename."""
    library = dm._mlib
    if hasattr(dm, "_motion_lib") and dm._motion_lib is not library:
        raise ValueError("Conflicting recorder motion library")
    dm._motion_lib = library
    return {"legacy_attribute": "_motion_lib", "current_attribute": "_mlib",
            "same_object": dm._motion_lib is dm._mlib}


def is_complete_recording(start_fraction, recorded_frames, source_frames):
    return start_fraction == 0.0 and recorded_frames == source_frames


def align_terrain_spacing(config, spacings):
    if not spacings or not all(math.isfinite(x) and x > 0 for x in spacings):
        raise ValueError("Invalid source terrain spacing")
    dx = spacings[0]
    if not all(math.isclose(x, dx, rel_tol=0, abs_tol=1e-7) for x in spacings):
        raise ValueError("Mixed terrain grids require separate recording batches")
    heightmap = config["env"]["dm"]["heightmap"]
    cells = heightmap["padding"] / dx
    if abs(cells - round(cells)) > 1e-5:
        raise ValueError("Terrain padding is not a whole number of source cells")
    previous = heightmap["horizontal_scale"]
    heightmap["horizontal_scale"] = dx
    original_padding = heightmap['padding']
    # PARC truncates padding/dx to int; float32 .4 otherwise becomes zero cells.
    heightmap['padding'] = round(cells) * dx
    return {"archived_grid_spacing_m": previous, "source_and_packed_grid_spacing_m": dx,
            'padding_cells': round(cells), 'source_padding_m': original_padding,
            'aligned_padding_m': heightmap['padding'],
            "reason": "Official square builder copies cells without resampling; grid spacing must agree",
            "policy_ray_spacing_changed": False}


def validate_packed_terrain(dm):
    import torch

    total, maximum = 0, 0.0
    for index, terrain in enumerate(dm._mlib._terrains):
        if not torch.allclose(terrain.dxdy, dm._terrain.dxdy, rtol=0, atol=1e-7):
            raise ValueError("Collision/source terrain and observation grid spacing differ")
        axes = [torch.arange(n, device=terrain.hf.device) for n in terrain.hf.shape]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        points = grid * terrain.dxdy + terrain.min_point + dm._dm_motion_offsets[index, 0]
        sampled = dm._terrain.get_hf_val_from_points(points)
        error = float((sampled - terrain.hf).abs().max())
        if error > 1e-5:
            raise ValueError("Packed observation terrain differs from collision/source tile: %s" % error)
        total += terrain.hf.numel()
        maximum = max(maximum, error)
    return {"all_tile_cells_checked": total, "maximum_height_error_m": maximum,
            "grid_spacing_matches": True}


def engine_origin_contract(env):
    import numpy as np

    actual = []
    for pointer in env._envs:
        position = env._gym.get_env_origin(pointer)
        actual.append([position.x, position.y, position.z])
    expected = env._env_offsets.detach().cpu().numpy()
    actual = np.asarray(actual)
    return {'engine_origins': actual.tolist(), 'observation_offsets': expected.tolist(),
            'max_error_m': float(np.abs(actual - expected).max())}


def align_engine_origins(env):
    """Correct coordinate bookkeeping before reset, preserving all shared views."""
    import torch

    before = engine_origin_contract(env)
    actual = torch.as_tensor(before['engine_origins'], device=env._env_offsets.device,
                             dtype=env._env_offsets.dtype)
    if actual.shape != env._env_offsets.shape or not torch.isfinite(actual).all():
        raise ValueError('Invalid Isaac Gym environment origins')
    dm = env.get_dm_env()
    shared = env._get_dm_slice(env._env_offsets)
    if (dm._env_offsets.shape != shared.shape
            or dm._env_offsets.data_ptr() != shared.data_ptr()):
        raise ValueError('DeepMimic environment offsets must share the engine buffer')
    env._env_offsets.copy_(actual)
    after = engine_origin_contract(env)
    if after['max_error_m'] > 1e-5 or not torch.equal(dm._env_offsets, shared):
        raise ValueError('Isaac Gym origins did not propagate to observation offsets')
    return {'before': before, 'after': after, 'shared_buffer_preserved': True,
            'source': 'gym.get_env_origin', 'applied_before_first_reset': True}


def run(bundle, output, selection="representatives", preflight_only=False):
    bundle, output = bundle.resolve(), output.resolve()
    manifest = validate_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "initializing", "source_sha256": manifest["source_sha256"],
              "selection": selection, "saved": [], "training_approved": False,
              "physics_recording_completed": False, "recorder": "official record_dm_motions"}
    write_json(output / "report.json", report)
    started = time.monotonic()
    env = None
    try:
        from isaacgym import gymapi  # Must precede torch imports.
        import numpy as np
        import torch
        import yaml

        torch.set_num_threads(4)
        torch.manual_seed(20261009)
        np.random.seed(20261009)
        if int(np.__version__.split(".")[0]) < 2:
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        repo = bundle / "PARC"
        sys.path.insert(0, str(repo))
        sys.modules.setdefault("parc", importlib.import_module("PARC"))
        os.chdir(str(repo))
        user_config = repo / "data/configs/user_config.yaml"
        user_config.parent.mkdir(parents=True, exist_ok=True)
        if user_config.exists():
            if yaml.safe_load(user_config.read_text()) != {"DATA_DIR": str(bundle)}:
                raise ValueError("Conflicting PARC user config")
        else:
            with user_config.open("x") as stream:
                yaml.safe_dump({"DATA_DIR": str(bundle)}, stream)

        from parc.util import file_io
        from parc.motion_tracker.envs import env_builder
        from parc.motion_tracker.learning import agent_builder
        from parc.motion_tracker.learning.dm_motion_recorder import record_dm_motions

        selected = (sorted(manifest["motions"]) if selection == "all" else
                    [row["path"] for row in manifest["representatives"]])
        inputs, spacings = [], []
        for relative in selected:
            path = bundle / relative
            metadata = manifest["motions"][relative]
            data = file_io.load_ms_file(str(path))
            motion, terrain = data.motion_data, data.terrain_data
            if motion is None or terrain is None:
                raise ValueError("Official motion/terrain loader returned missing data")
            spacings.append(float(terrain.dx))
            count = metadata["source_frames"]
            if (motion.root_pos.shape != (count, 3) or motion.root_rot.shape != (count, 4)
                    or motion.joint_rot.shape != (count, 14, 4)
                    or motion.body_contacts.shape != (count, 15) or motion.fps != 30):
                raise ValueError("Unexpected source motion contract")
            arrays = {"root_pos": motion.root_pos, "root_rot": motion.root_rot,
                      "joint_rot": motion.joint_rot, "contacts": motion.body_contacts,
                      "hf": terrain.hf, "hf_maxmin": terrain.hf_maxmin,
                      "min_point": terrain.min_point}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Non-finite source data")
            inputs.append(dict(metadata, path=relative, name=path.stem,
                array_sha256={name: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                              for name, a in arrays.items()}))
        report["inputs"] = inputs
        dataset_path = output / "motions.yaml"
        dataset_path.write_text(yaml.safe_dump({"motions": [
            {"file": str(bundle / p), "weight": 1.0} for p in selected]}))
        config = yaml.safe_load((bundle / "tracker/dm_env.yaml").read_text())
        report["terrain_spacing_contract"] = align_terrain_spacing(config, spacings)
        config["env"]["dm"]["motion_file"] = str(dataset_path)
        config["env"]["dm"]["terrain_save_path"] = str(output / "terrain.pkl")
        config["env"]["output_motion_dir"] = str(output / "recorded_motions")
        env_path = output / "record_env.yaml"
        env_path.write_text(yaml.safe_dump(config))
        report["env_config_sha256"] = digest(env_path)
        report["tracker_sha256"] = digest(bundle / "tracker/model.pt")
        env = env_builder.build_env(str(env_path), len(selected), "cuda:0", False)
        report['engine_origin_alignment'] = align_engine_origins(env)
        report['engine_origin_contract'] = engine_origin_contract(env)
        write_json(output / 'report.json', report)
        if report['engine_origin_contract']['max_error_m'] > 1e-5:
            raise ValueError('Isaac Gym environment origins differ from PARC observation offsets')
        report["packed_terrain_validation"] = validate_packed_terrain(env.get_dm_env())
        obs_shape, action_shape = env.get_obs_space().shape, env.get_action_space().shape
        if obs_shape != (1312,) or action_shape != (28,):
            raise ValueError("Released tracker observation/action shape mismatch: %s %s" %
                             (obs_shape, action_shape))
        report["observation_layout"] = env._compute_obs(ret_obs_shapes=True)
        agent = agent_builder.build_agent(str(bundle / "tracker/agent_config.yaml"), env, "cuda:0")
        agent.load(str(bundle / "tracker/model.pt"))
        agent.eval()
        observation, _ = env.reset()
        with torch.no_grad():
            norm = agent._obs_norm.normalize(observation)
            action = agent._model.eval_actor(norm).mean
            critic = agent._model.eval_critic(norm)
        if not torch.isfinite(action).all() or not torch.isfinite(critic).all():
            raise ValueError("Non-finite policy preflight")
        report.update(strict_checkpoint_loaded=True, observation_shape=list(observation.shape),
                      action_shape=list(action.shape), torch=torch.__version__, numpy=np.__version__)
        write_json(output / "report.json", report)
        if preflight_only:
            report["status"] = "preflight_passed_no_recording"
        else:
            phase = {"start_fraction": 0.0}
            dm = env.get_dm_env()
            report["recorder_compatibility"] = install_recorder_compatibility(dm)
            original_start = dm.set_motion_start_time_fraction
            original_save = env.save_agent_states_to_file
            by_name = {row["name"]: row for row in inputs}

            def observe_start(fraction):
                if not torch.all(fraction == fraction[0]):
                    raise ValueError("Unexpected nonuniform official retry fractions")
                phase["start_fraction"] = float(fraction[0].item())
                return original_start(fraction)

            def observe_save(env_id, output_motion_name=None):
                if getattr(env, "_record_ref", False) or env._bypass_record_fail:
                    raise ValueError("Reference recording or failure bypass is prohibited")
                reference_name = dm.get_env_motion_name(env_id)
                source = by_name[reference_name]
                first_position = env._dm_agent_motion[env_id]["frames"][0][:3].clone()
                canonical_xy = env._get_global_xy_pos(first_position[:2], env_id)
                canonical_z = dm._terrain.get_hf_val_from_points(canonical_xy)
                original_save(env_id, output_motion_name)
                path = Path(env._output_motion_dir) / (output_motion_name + ".pkl")
                with path.open("rb") as stream:
                    recorded = pickle.load(stream)
                count = int(recorded["frames"].shape[0])
                if count < 2 or not np.isfinite(recorded["frames"]).all():
                    raise ValueError("Invalid physical recording")
                saved_terrain = recorded["terrain"]
                saved_terrain.to_torch(device=dm._terrain.hf.device)
                axes = [torch.arange(n, device=dm._terrain.hf.device) for n in saved_terrain.hf.shape]
                grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
                global_xy = grid * saved_terrain.dxdy + saved_terrain.min_point + canonical_xy
                expected_height = dm._terrain.get_hf_val_from_points(global_xy) - canonical_z
                terrain_error = float((expected_height - saved_terrain.hf).abs().max())
                if terrain_error > 1e-5:
                    raise ValueError("Saved terrain does not preserve simulated world heights")
                report["saved"].append({"candidate_key": source["candidate_key"],
                    "source_sha256": source["sha256"], "filename": path.name,
                    "sha256": digest(path), "frames": count, "fps": recorded["fps"],
                    "source_frames": source["source_frames"],
                    "start_fraction": phase["start_fraction"],
                    "whole_motion": is_complete_recording(phase["start_fraction"],
                                                          count, source["source_frames"]),
                    "recorded_state": "simulated", "contacts": "simulated_contact_force_threshold",
                    "recorded_terrain_error_m": terrain_error,
                    "localized_to_global_translation": canonical_xy.cpu().tolist() + [float(canonical_z)],
                    "training_approved": False})
                write_json(output / "report.json", report)

            dm.set_motion_start_time_fraction = observe_start
            env.save_agent_states_to_file = observe_save
            try:
                with torch.no_grad():
                    record_dm_motions(agent)
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
            for saved in report["saved"]:
                paths = list((output / "recorded_motions").rglob(saved["filename"]))
                if len(paths) != 1 or digest(paths[0]) != saved["sha256"]:
                    raise ValueError("Saved recording missing or changed after organization")
                saved["path"] = str(paths[0].relative_to(output))
            saved_ids = {r["candidate_key"] for r in report["saved"]}
            report.update(status="official_recording_finished", physics_recording_completed=True,
                whole_motion_count=sum(r["whole_motion"] for r in report["saved"]),
                partial_motion_count=sum(not r["whole_motion"] for r in report["saved"]),
                failed_candidate_ids=[r["candidate_key"] for r in inputs if r["candidate_key"] not in saved_ids])
        validate_bundle(bundle)
        report["inputs_unchanged_after_run"] = True
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
        if env is not None:
            env._gym.destroy_sim(env._sim)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", choices=("representatives", "all"), default="representatives")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.bundle, args.output, args.selection, args.preflight_only)
