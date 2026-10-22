"""Isolated CPU parser/accounting fixtures; never load assets or start CUDA."""
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent

from scripts import profile_mb_sonic_neural_ops as cli


def arguments(output):
    return ["--allocation-id", "parent-unit-fixture", "--device", "cuda", "--artifact", "fixture-artifact",
            "--bundle", "fixture-bundle", "--manifest", "fixture-manifest", "--output", str(output)]


def test_parser_requires_explicit_allocation_output_cuda_and_assets(tmp_path):
    initialized = torch.cuda.is_initialized()
    args = cli.build_parser().parse_args(arguments(tmp_path / "report"))
    assert args.device == "cuda"
    assert args.allocation_id == "parent-unit-fixture"
    assert args.output == tmp_path / "report"
    for flag in ("--allocation-id", "--output", "--device", "--artifact", "--bundle", "--manifest"):
        missing = arguments(tmp_path)
        index = missing.index(flag)
        del missing[index:index + 2]
        with pytest.raises(SystemExit) as error:
            cli.build_parser().parse_args(missing)
        assert error.value.code == 2
    for device in ("cpu", "cuda:0"):
        invalid = arguments(tmp_path)
        invalid[invalid.index("--device") + 1] = device
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(invalid)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("name,family", [
    ("aten::cudnn_convolution", "conv"), ("aten::conv1d", "conv"),
    ("aten::addmm", "gemm"), ("aten::bmm", "gemm"), ("aten::linear", "gemm"),
    ("aten::_scaled_dot_product_flash_attention", "attention_fused_transformer"),
    ("aten::_transformer_encoder_layer_fwd", "attention_fused_transformer"),
    ("aten::native_layer_norm", "norm"), ("aten::index_select", "index"),
    ("aten::clone", "layout_copy"), ("aten::gelu", "activation"), ("aten::mul", "other"),
])
def test_operator_families(name, family):
    assert cli.op_family(name) == family


@pytest.fixture
def accounting():
    root = SimpleNamespace(name="mb.root", cpu_parent=None)
    pose = SimpleNamespace(name="mb.pose", cpu_parent=None)
    decoder = SimpleNamespace(name="mb.codec_decoder", cpu_parent=None)
    nested = SimpleNamespace(name="aten::parent_convolution", cpu_parent=decoder)
    rows = [SimpleNamespace(key=name, count=count, device_time_total=100.) for name, count in cli.REGIONS.items()]
    events = [SimpleNamespace(device_type=DeviceType.CUDA, device_time_total=400., is_user_annotation=False)]
    for name, parent, cost, shape in (("aten::addmm", root, 10., [[1, 8], [8, 8]]),
                                      ("aten::_transformer_encoder_layer_fwd", pose, 20., [[1, 8, 8]]),
                                      ("aten::cudnn_convolution", nested, 30., [[1, 8, 32]]),
                                      ("aten::index", None, 40., [[1, 32]])):
        events.append(SimpleNamespace(device_type=DeviceType.CPU, name=name, cpu_parent=parent,
                                      self_device_time_total=cost, cpu_time_total=999999.))
        rows.append(SimpleNamespace(device_type=DeviceType.CPU, key=name, count=1,
                                    device_time_total=cost * 2, self_device_time_total=cost, input_shapes=shape))
    for row in rows[:3]:
        row.device_type = DeviceType.CPU
    rows.append(SimpleNamespace(device_type=DeviceType.CUDA, key="cuda-kernel-fixture",
                                self_device_time_total=400.))
    return events, rows


def test_summary_uses_cuda_activity_and_cpu_attributed_self_time_without_double_count(accounting):
    result = cli.summarize(*accounting)
    assert result["cuda_activity_sum_us"] == 400.
    assert result["family_self_device_us"] == {
        "mb.root/gemm": 10., "mb.pose/attention_fused_transformer": 20.,
        "mb.codec_decoder/conv": 30., "outside_neural/index": 40.,
    }
    assert sum(result["family_self_device_us"].values()) == 100.
    assert result["regions"]["mb.pose"] == dict(calls=4, device_us=100.)
    assert result["ops"][0]["op"] == "aten::index"
    assert result["ops"][0]["shapes"] == [[1, 32]]
    assert len(result["ops"]) == 4
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_gpu_annotation_duplicates_do_not_override_cpu_regions_or_inflate_activity(accounting):
    events, averages = accounting
    for name, count in (("mb.root", 5), ("mb.pose", 4), ("mb.codec_decoder", 3)):
        averages.append(SimpleNamespace(device_type=DeviceType.CUDA, key=name, count=count,
                                        device_time_total=10000., self_device_time_total=10000.,
                                        is_user_annotation=True))
        events.append(SimpleNamespace(device_type=DeviceType.CUDA, name=name, device_time_total=10000.,
                                      is_user_annotation=True))
    result = cli.summarize(events, averages)
    assert result["cuda_activity_sum_us"] == 400.
    assert {name: value["calls"] for name, value in result["regions"].items()} == cli.REGIONS
    assert all(value["device_us"] == 100. for value in result["regions"].values())
    assert len(result["ops"]) == 4


@pytest.mark.parametrize("change", ["no_cuda", "zero_cuda", "missing_region", "wrong_passes", "zero_region"])
def test_missing_real_device_evidence_is_blocked(accounting, change):
    events, averages = accounting
    if change == "no_cuda":
        events.pop(0)
    elif change == "zero_cuda":
        events[0].device_time_total = 0.
    elif change == "missing_region":
        averages.pop(0)
    elif change == "wrong_passes":
        averages[1].count = 3
    else:
        averages[2].device_time_total = 0.
    with pytest.raises(RuntimeError, match="CUPTI CUDA"):
        cli.summarize(events, averages)


def test_profiler_attributes_are_required_not_faked(accounting):
    events, averages = accounting
    del events[1].self_device_time_total
    with pytest.raises(AttributeError):
        cli.summarize(events, averages)


def test_local_torch_profiler_api_without_cuda_initialization():
    initialized = torch.cuda.is_initialized()
    assert isinstance(FunctionEvent.device_time_total, property)
    assert isinstance(FunctionEvent.self_device_time_total, property)
    assert "is_user_annotation" in inspect.signature(FunctionEvent).parameters
    assert "group_by_input_shape" in inspect.signature(torch.profiler.profile.key_averages).parameters
    assert "record_shapes" in inspect.signature(torch.profiler.profile).parameters
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("raise_inside", [False, True])
def test_annotation_hooks_only_selected_outer_module_and_removes_them(raise_inside):
    initialized = torch.cuda.is_initialized()
    layer = torch.nn.TransformerEncoderLayer(8, 2, batch_first=True)
    outer = torch.nn.Sequential(layer)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        with cli.annotate({"mb.pose": outer}):
            assert len(outer._forward_pre_hooks) == len(outer._forward_hooks) == 1
            assert not any(module._forward_hooks or module._forward_pre_hooks for module in layer.modules())
            assert torch.backends.mha.get_fastpath_enabled() == fastpath
            if raise_inside:
                raise ValueError("isolated fixture")
    except ValueError as error:
        assert str(error) == "isolated fixture"
    assert not outer._forward_pre_hooks and not outer._forward_hooks
    assert torch.backends.mha.get_fastpath_enabled() == fastpath
    assert torch.cuda.is_initialized() == initialized


def test_annotation_does_not_change_isolated_cpu_module_outputs():
    layer = torch.nn.Linear(3, 2).eval()
    value = torch.ones(1, 3)
    expected = layer(value)
    with cli.annotate({"mb.root": layer}):
        actual = layer(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_main_writes_blocked_without_inventing_gpu_timings(tmp_path, monkeypatch):
    initialized = torch.cuda.is_initialized()
    def unavailable(_args):
        raise RuntimeError("CUPTI unavailable in isolated CPU fixture")
    monkeypatch.setattr(cli, "run", unavailable)
    output = tmp_path / "blocked"
    assert cli.main(arguments(output)) == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "motionbricks.mb_sonic.neural_ops.v1"
    assert report["status"] == "BLOCKED"
    assert report["optimizer_steps"] == 0
    assert report["allocation_id"] == "parent-unit-fixture"
    assert "CUPTI" in report["reason"]
    assert "cuda_activity_sum_us" not in report
    assert torch.cuda.is_initialized() == initialized
