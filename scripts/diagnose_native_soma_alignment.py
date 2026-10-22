"""Frozen-weight layer trace: candidate choices versus released demo defaults."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.native_mb_locomotion import NativeReferenceClips, NativeSmartLocomotion
from scripts.portable_native_mb_soma import load_bundle
from scripts.evaluate_native_mb_smart_locomotion import capture_inference, _trace_arrays, shipped_buffer_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans", type=int, default=45)
    args = parser.parse_args()
    if args.output.exists() or not 6 <= args.plans <= 90:
        raise ValueError("Use a fresh output and bounded 6-90 plans")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    infer, bundle = load_bundle(args.bundle)
    bank = NativeReferenceClips(infer.global_motion_rep,
                                {key: value.to(infer.device) for key, value in bundle["reference_clips"].items()})
    cases = (("candidate", True, False, 4, 4), ("keep_endpoint", False, False, 4, 4),
             ("released_cursor_endpoint", False, True, 4, 4), ("one_keyframe_endpoint", False, False, 1, 1),
             ("four_context_one_target", False, False, 1, 4), ("one_context_four_target", False, False, 4, 1))
    reports, arrays = {}, {}
    for name, skip_endpoint, released_cursor, keyframes, context_keyframes in cases:
        torch.manual_seed(2333)
        controller = NativeSmartLocomotion(infer, bank, keyframes_per_end=keyframes, context_keyframes=context_keyframes)
        controller.SKIP_ENDING_TARGET_COND = skip_endpoint
        history = bundle["initial_history"].to(infer.device).clone()
        pieces, rows = [history], []
        with torch.inference_mode(), capture_inference(infer) as (snapshots, _):
            for index in range(args.plans):
                velocity = [0, 0] if index < 12 or index >= 36 else [0, 1.2]
                style = "idle" if velocity == [0, 0] else "walk"
                before = history.clone()
                prediction, count = controller.predict(history, velocity, 0., style=style, random_seed=2333)
                frames = int(count.item()) * 4
                trace = _trace_arrays(infer, snapshots, frames)
                if released_cursor:
                    committed, history, start, _ = shipped_buffer_step(prediction, infer.device)
                else:
                    start = 4
                    committed = prediction[:, 4:12].clone()
                    history = committed[:, -4:].clone()
                constraints = controller.last_constraints
                root_trace = trace["root_module_global_root"]
                local_trace = trace["decoded_local_root"]
                previous = pieces[-1][0, -1, :3]
                seam = committed[0, 0, :3] - previous
                internal = torch.diff(committed[0, :, :3], dim=0)
                speed = internal[:, [0, 2]].norm(dim=-1) * 30
                rows.append({"index": index, "command": velocity, "style": style,
                             "tokens": int(count.item()), "playback_start": start,
                             "input_root_first": before[0, 0, :3].cpu().tolist(),
                             "input_root_last": before[0, -1, :3].cpu().tolist(),
                             "condition_root_first": constraints["global_root_values"][0, 0, :3].cpu().tolist(),
                             "condition_root_goal": constraints["global_root_values"][0, -1, :3].cpu().tolist(),
                             "root_module_xz_speed_max_mps": float(np.linalg.norm(np.diff(root_trace[:, [0, 2]], axis=0), axis=-1).max() * 30),
                             "decoder_local_velocity_xz_max_mps": float(np.linalg.norm(local_trace[:, 1:3], axis=-1).max()),
                             "seam_planar_step_m": float(seam[[0, 2]].norm()),
                             "internal_planar_speed_mean_mps": float(speed.mean()),
                             "internal_planar_speed_max_mps": float(speed.max()),
                             "root_last": committed[0, -1, :3].cpu().tolist()})
                pieces.append(committed)
                if index in (0, 11, 12, 20, 35, 44):
                    prefix = f"{name}_{index:03d}_"
                    arrays[prefix + "prediction"] = prediction.cpu().numpy()
                    arrays[prefix + "history"] = before.cpu().numpy()
                    arrays.update({prefix + key: value for key, value in trace.items()})
                if not torch.isfinite(history).all():
                    raise ValueError("Nonfinite history")
        played = torch.cat(pieces, 1)[0].cpu().numpy()
        arrays[name + "_played"] = played
        speed = np.linalg.norm(np.diff(played[:, [0, 2]], axis=0), axis=-1) * 30
        reports[name] = {"skip_endpoint": skip_endpoint, "released_cursor": released_cursor,
                         "keyframes": keyframes, "context_keyframes": context_keyframes, "root_path_m": float(speed.sum() / 30),
                         "speed_max_mps": float(speed.max()), "speed_mean_mps": float(speed.mean()),
                         "final_displacement_m": float(np.linalg.norm(played[-1, [0, 2]] - played[3, [0, 2]])),
                         "plans": rows}
        print(json.dumps({"case": name, **{key: value for key, value in reports[name].items() if key != "plans"}}), flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "traces.npz", **arrays)
    (args.output / "report.json").write_text(json.dumps({"bundle_sha256": file_sha256(args.bundle),
        "gpu": torch.cuda.get_device_name(), "post_processing": None, "cases": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
