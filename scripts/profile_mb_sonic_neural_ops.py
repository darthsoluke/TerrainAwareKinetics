"""Parent-allocated, single native AR8 CUDA operator decomposition; no optimizer."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch.autograd import DeviceType
REGIONS = {"mb.root": 1, "mb.pose": 4, "mb.codec_decoder": 1}
FAMILIES = (("conv", r"convolution|conv[123]d"), ("attention_fused_transformer", r"attention|transformer"),
            ("gemm", r"addmm|bmm|matmul|aten::mm$|aten::linear$"), ("norm", r"norm"),
            ("index", r"index|gather|scatter|embedding|select|slice|masked|where"),
            ("layout_copy", r"copy|clone|contiguous|view|reshape|permute|transpose|aten::to$|cat|stack"),
            ("activation", r"relu|gelu|silu|sigmoid|tanh|softmax"))

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "bundle", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    return parser

def op_family(name):
    return next((family for family, pattern in FAMILIES if re.search(pattern, name)), "other")

@contextmanager
def annotate(modules):
    scopes, handles = {}, []
    def before(name, _module, _inputs):
        scopes[name] = torch.profiler.record_function(name)
        scopes[name].__enter__()
    def after(name, _module, _inputs, _output):
        scopes.pop(name).__exit__(None, None, None)
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(partial(before, name), prepend=True))
            handles.append(module.register_forward_hook(partial(after, name), always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()

def summarize(events, averages):
    cuda_us = sum(e.device_time_total for e in events if e.device_type == DeviceType.CUDA and not e.is_user_annotation)
    regions = {e.key: dict(calls=e.count, device_us=e.device_time_total) for e in averages if e.device_type == DeviceType.CPU and e.key in REGIONS}
    if cuda_us <= 0 or any(name not in regions or regions[name]["calls"] != count
                          or regions[name]["device_us"] <= 0 for name, count in REGIONS.items()):
        raise RuntimeError("CUPTI CUDA kernels/required native region timings unavailable or call counts changed")
    families = defaultdict(float)
    for event in events:
        if event.device_type != DeviceType.CPU or not event.name.startswith("aten::"):
            continue
        parent = event.cpu_parent
        while parent is not None and parent.name not in REGIONS:
            parent = parent.cpu_parent
        region = parent.name if parent is not None else "outside_neural"
        families[region + "/" + op_family(event.name)] += event.self_device_time_total
    ops = [dict(op=e.key, family=op_family(e.key), calls=e.count, shapes=e.input_shapes,
                self_device_us=e.self_device_time_total) for e in averages
           if e.device_type == DeviceType.CPU and e.key.startswith("aten::") and e.self_device_time_total > 0]
    return dict(cuda_activity_sum_us=cuda_us, regions=regions, family_self_device_us=dict(families),
                ops=sorted(ops, key=lambda row: row["self_device_us"], reverse=True))

def run(args):
    from motionbricks.data.full_corpus_inference import tensor_digest
    from motionbricks.mb_sonic.mb_reference import TrainedMBReferenceAdapter, native_precision_contract
    from motionbricks.mb_sonic.reference import file_sha256
    from motionbricks.mb_sonic.scene import load_scene
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities() or not torch.cuda.is_available():
        raise RuntimeError("CUDA/CUPTI profiling activity unavailable")
    scene = load_scene(args.manifest)
    if scene.manifest.scene_id != "000":
        raise ValueError("This bounded diagnostic requires canonical source 000")
    adapter = TrainedMBReferenceAdapter.load(args.artifact, args.bundle, canonical_scene=scene, device=args.device)
    source = scene.root / scene.manifest.source["scene"]["file"]
    source_before, before = file_sha256(source), tensor_digest(adapter.controller.state_dict())
    with np.load(source, allow_pickle=False) as archive:
        history = torch.tensor(archive["initial_history"].copy(), device=args.device, dtype=torch.float32)
    controls = history.new_tensor([[0., .5, 1., 0.]])
    modules = dict(zip(REGIONS, (adapter.controller.root, adapter.controller.pose, adapter.controller.codec.decoder)))
    with torch.inference_mode():
        for _ in range(3):
            adapter.controller(history, controls, adapter.scene)
        torch.cuda.synchronize()
        with annotate(modules), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            adapter.controller(history, controls, adapter.scene)
            torch.cuda.synchronize()
    averages = profile.key_averages(group_by_input_shape=True)
    profile.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "operators.txt").write_text(averages.table(sort_by="self_device_time_total", row_limit=-1), encoding="utf-8")
    costs = summarize(profile.events(), averages)
    after, source_after = tensor_digest(adapter.controller.state_dict()), file_sha256(source)
    if before != after or source_before != source_after:
        raise ValueError("Frozen model/source changed during diagnostic")
    return dict(status="PASS", torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), **costs,
                model_identity=adapter.identity, model_state_sha256_before=before, model_state_sha256_after=after,
                source_sha256_before=source_before, source_sha256_after=source_after,
                precision=native_precision_contract(), warmup_calls=3, profiled_calls=1, native_tokens=8)

def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="motionbricks.mb_sonic.neural_ops.v1", allocation_id=args.allocation_id,
                  device=args.device, optimizer_steps=0, status="BLOCKED")
    try:
        report.update(run(args))
    except RuntimeError as error:
        report["reason"] = str(error)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
