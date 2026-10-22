"""Parity and latency experiment for unchanged, fixed-shape native terrain MB."""

import argparse
import io
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch

from scripts.soma_lab_server import load_terrain_live_bundle
from motionbricks.data.native_mb_online_live import planar_command_goal, predict_planar_terrain
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.soma_cuda_graph import StaticCudaCall
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner


class FixedCall(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *inputs):
        return self.function(*inputs)


def cuda_constants(module):
    archive = io.BytesIO()
    torch.jit.save(module, archive)
    archive.seek(0)
    result = torch.jit.load(archive, map_location='cuda:0')
    for node in result.graph.nodes():
        if node.kind() == 'prim::Constant' and 'value' in node.attributeNames() and node.kindOf('value') == 't':
            value = node.t('value')
            shape_scalar = any(any(item.node().kind() == 'prim::NumToTensor' for item in use.user.inputs())
                               for use in node.output().uses())
            if value.ndim == 0 and shape_scalar:
                node.t_('value', value.cpu())
    return result


def dense_transformers(inference):
    for module in inference.modules():
        if isinstance(module, torch.nn.TransformerEncoder):
            native, zero_padding = module.forward, module.use_nested_tensor

            def forward(src, mask=None, src_key_padding_mask=None, is_causal=None,
                        native=native, zero_padding=zero_padding):
                result = native(src, mask=mask, src_key_padding_mask=src_key_padding_mask,
                                is_causal=is_causal)
                if zero_padding and src_key_padding_mask is not None:
                    result = result.masked_fill(src_key_padding_mask[..., None], 0)
                return result

            module.use_nested_tensor = False
            module.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, scene, history = load_terrain_live_bundle(args.bundle, 'cuda:0', args.scene)
    inference.requires_grad_(False)
    history = history.cuda()
    scene = {key: torch.as_tensor(value, device='cuda:0') for key, value in scene.items()}
    skinner = move_skinner(load_official_skinner(inference.motion_rep.skeleton), torch.device('cuda:0'))
    skinner.skeleton_skin.float().cuda()
    captured = {}
    inference._root_model.backbone_net.register_forward_hook(
        lambda _m, _inputs, output: captured.update(root=output['pred_global_root_values']))
    native_decode = inference._vqvae_pose_model.forward_decoder

    def decode(ids, *inputs, **kwargs):
        captured['tokens'] = ids
        return native_decode(ids, *inputs, **kwargs)

    inference._vqvae_pose_model.forward_decoder = decode
    settings = {'num_inference_step': 4, 'pose_token_sampling_use_argmax': True,
                'cache_terrain_encoder': True}

    def predict(history, goal):
        output, count = predict_planar_terrain(inference, history, goal, scene, config=settings)
        return {'motion': output, 'count': count, 'root': captured['root'], 'tokens': captured['tokens']}

    def geometry(motion):
        pose = inference.motion_rep.inverse(motion, False, return_all=False)
        vertices = skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
        observed = reencode_played_history(motion[:, -4:], inference.global_motion_rep)
        return {'vertices': vertices, 'joints': pose['posed_joints'][0], 'history': observed}

    def benchmark(call, inputs):
        for _ in range(5):
            call(*inputs)
        durations = []
        for _ in range(24):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call(*inputs)
            torch.cuda.synchronize()
            durations.append((time.perf_counter()-start)*1000)
        return {'median_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations, 95))}

    with torch.inference_mode():
        cases = []
        for velocity, heading in (([0., 0.], 0.), ([0., 1.6], 0.), ([1.6, 0.], 1.57),
                                  ([0., -1.6], 3.14), ([-1.6, 0.], -1.57)):
            goal = planar_command_goal(history[..., :5], velocity, heading, 8)
            expected = predict(history, goal)
            cases.append((history.clone(), goal, expected))
            history = reencode_played_history(expected['motion'][:, 24:28], inference.global_motion_rep)
        report = {'eager': benchmark(predict, cases[0][:2])}
        dense_transformers(inference)
        print('Tracing fixed-shape inference', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', torch.jit.TracerWarning)
            traced = torch.jit.freeze(torch.jit.trace(FixedCall(predict).eval(), cases[0][:2],
                                                       strict=False, check_trace=False))
            traced = cuda_constants(traced)
        report['traced'] = benchmark(traced, cases[0][:2])
        print(report, flush=True)
        graph = StaticCudaCall(traced)
        report['cuda_graph'] = benchmark(graph, cases[0][:2])
        report['parity'] = []
        for current, goal, expected in cases:
            actual = graph(current, goal)
            a = inference.motion_rep.inverse(actual['motion'][:, :32], False, return_all=False)
            b = inference.motion_rep.inverse(expected['motion'][:, :32], False, return_all=False)
            row = {'active_tokens_equal': torch.equal(actual['tokens'][:, :8], expected['tokens'][:, :8]),
                   'fk_max_m': float((a['posed_joints']-b['posed_joints']).norm(dim=-1).max()),
                   'feature_max': float((actual['motion'][:, :32]-expected['motion'][:, :32]).abs().max())}
            report['parity'].append(row)
        (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        motion = cases[0][2]['motion'][:, 4:12].contiguous()
        report['eager_geometry'] = benchmark(geometry, (motion,))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', torch.jit.TracerWarning)
            traced_geometry = torch.jit.freeze(torch.jit.trace(FixedCall(geometry).eval(), (motion,),
                                                                strict=False, check_trace=False))
        report['traced_geometry'] = benchmark(traced_geometry, (motion,))
        graph_geometry = StaticCudaCall(cuda_constants(traced_geometry))
        report['graph_geometry'] = benchmark(graph_geometry, (motion,))
        report['geometry_parity'] = []
        for _, _, expected in cases:
            motion = expected['motion'][:, 4:12].contiguous()
            actual_geo, eager_geo = graph_geometry(motion), geometry(motion)
            report['geometry_parity'].append({key: float((actual_geo[key]-value).abs().max())
                                              for key, value in eager_geo.items()})
        (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
