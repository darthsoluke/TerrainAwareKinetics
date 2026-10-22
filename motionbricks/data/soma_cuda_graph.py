"""Opt-in static CUDA replay of the unchanged native MotionBricks decoder.

Uses PyTorch 2.11's CUDA graph recipe (side-stream warmup, persistent inputs,
capture, copy/replay). This is execution scheduling, not a pose correction.
Single inference thread only; tensor signatures and inference mode are enforced.
"""

import torch


def _copy_output(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _copy_output(item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError("Graph outputs must be tensors, dictionaries, or None")


class StaticCudaCall:
    def __init__(self, function):
        self.function = function
        self.graph = None
        self.signature = None

    def __call__(self, *inputs):
        if not torch.is_inference_mode_enabled():
            raise ValueError("CUDA replay is inference-only")
        if not inputs or any(not isinstance(x, torch.Tensor) or not x.is_cuda for x in inputs):
            raise ValueError("Static CUDA inputs must all be CUDA tensors")
        # Sources are copied into persistent buffers before replay. Their stride
        # may differ (especially singleton batch views of a played prefix); only
        # the persistent buffers' strides belong to the captured graph.
        signature = tuple((x.shape, x.dtype, x.device) for x in inputs)
        if self.graph is None:
            self.signature = signature
            self.inputs = tuple(x.clone() for x in inputs)
            stream = torch.cuda.Stream(device=inputs[0].device)
            current = torch.cuda.current_stream(inputs[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.function(*self.inputs)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.function(*self.inputs)
        elif signature != self.signature:
            raise ValueError("CUDA graph input shape, dtype, or device changed")
        for destination, source in zip(self.inputs, inputs):
            destination.copy_(source)
        self.graph.replay()
        # Previous plans/history must remain immutable after the next replay.
        return _copy_output(self.output)


def install_decoder_graph(inference):
    from motionbricks.data.bones_models import ConditionedBonesVQVAE
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE

    tokenizer = inference._vqvae_pose_model
    if type(tokenizer) is not ConditionedBonesVQVAE:
        raise TypeError("Only the tested native SOMA tokenizer adapter can be captured")
    if tokenizer.training or any(parameter.requires_grad for parameter in tokenizer.parameters()):
        raise ValueError("CUDA decoder requires a frozen evaluation model")
    # The serialized root policy uses Python-list advanced indexing, which creates
    # a host index copy. Leave that unchanged eager step outside the captured net.
    native = VQVAE.forward_decoder
    call = StaticCudaCall(lambda ids, target, target_mask, external, token_mask:
                          native(tokenizer, ids, target, target_mask, external, False, token_mask))

    def forward(ids, target_cond, has_target_cond=None, external_cond=None,
                use_overall_indices=True, token_mask=None):
        if use_overall_indices or any(value is None for value in (target_cond, has_target_cond, external_cond, token_mask)):
            raise ValueError("Captured MB decoder requires dense conditions and per-head indices")
        return tokenizer._apply_condition(call(ids, target_cond, has_target_cond, external_cond, token_mask), external_cond)

    tokenizer.forward_decoder = forward
    return call


def install_backbone_graphs(inference):
    """Capture native Root/Pose calls using PyTorch's dense Transformer path.

    NestedTensor conversion has data-dependent shapes and host mask checks.
    Dense masked attention is the supported alternative, not a new network.
    Opt-in only: active token IDs and output motion require equivalence tests.
    """
    from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network

    root = inference._root_model.backbone_net
    pose = inference._pose_model.backbone_net
    if type(root) is not root_backbone_network or type(pose) is not pose_backbone_network:
        raise TypeError("Only native MB Root/Pose backbones are tested")
    if not torch.backends.mha.get_fastpath_enabled() or getattr(inference, "goal_library", None) is not None:
        raise ValueError("Captured backbones require the tested MHA path and goal-free control")
    for network in (root, pose):
        if network.training or any(parameter.requires_grad for parameter in network.parameters()):
            raise ValueError("CUDA backbones require frozen evaluation models")
        for module in network.modules():
            if isinstance(module, torch.nn.TransformerEncoder):
                if module.norm is not None:
                    raise ValueError("Only the native MB encoders without a final norm are tested")
                native_encoder = module.forward
                nested_padding = module.use_nested_tensor

                def dense_encoder(src, mask=None, src_key_padding_mask=None, is_causal=None,
                                  native=native_encoder, zero_padding=nested_padding):
                    output = native(src, mask=mask, src_key_padding_mask=src_key_padding_mask,
                                    is_causal=is_causal)
                    if zero_padding and src_key_padding_mask is not None:
                        if mask is not None or src_key_padding_mask.dtype != torch.bool:
                            raise ValueError("Only native MB boolean suffix padding is tested")
                        # PyTorch's original NestedTensor path calls
                        # to_padded_tensor(0.0, src.size()) after all encoder layers.
                        output = output.masked_fill(src_key_padding_mask[..., None], 0)
                    return output

                module.use_nested_tensor = False
                module.forward = dense_encoder
    native_root, native_pose = root.forward, pose.forward
    root_config = None
    root_call = StaticCudaCall(lambda *inputs: native_root(*inputs, config=root_config))
    pose_call = StaticCudaCall(native_pose)

    def forward_root(global_root_values, has_global_root_values, local_root_values,
                     has_local_root_values, poses, has_poses, num_tokens,
                     text_embeddings=None, has_text_embeddings=None, groundtruth_num_tokens=None,
                     allowed_pred_num_tokens=None, config=None):
        nonlocal root_config
        if any(value is not None for value in (text_embeddings, has_text_embeddings,
                                               groundtruth_num_tokens, allowed_pred_num_tokens)):
            raise ValueError("Captured Root supports only the tested goal-free control signature")
        current_config = dict(config or {})
        if root_config is None:
            root_config = current_config
        elif root_config != current_config:
            raise ValueError("Captured Root configuration changed")
        return root_call(global_root_values, has_global_root_values, local_root_values,
                         has_local_root_values, poses, has_poses, num_tokens)

    def forward_pose(pose_tokens, local_root_values, pose_cond, has_pose_cond, num_tokens,
                     text_embeddings=None, has_text_embeddings=None):
        if text_embeddings is not None or has_text_embeddings is not None:
            raise ValueError("Captured Pose supports only the tested goal-free control signature")
        return pose_call(pose_tokens, local_root_values, pose_cond, has_pose_cond, num_tokens)

    root.forward, pose.forward = forward_root, forward_pose
    return root_call, pose_call
