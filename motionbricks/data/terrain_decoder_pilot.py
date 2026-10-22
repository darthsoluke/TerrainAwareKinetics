"""Isolated decoder-only terrain/feedback experiment using native MB inference.

Root, Pose, the motion encoder and the codebook remain frozen. Terrain enters
before the native decoder's temporal convolutions through a zero-gated adapter.
The experiment does not alter the production model factory or live controller.
"""

from copy import deepcopy

import torch
from torch import nn

from motionbricks.data.native_mb_online_live import predict_planar_terrain
from motionbricks.data.native_mb_vertical_frame import (
    TERRAIN_VERTICAL_FRAME, observed_terrain_origin, shift_global_motion,
)
from motionbricks.motion_backbone.neural_modules.local_terrain_encoder import (
    LocalTerrainEncoder, TerrainCrossAttention,
)
from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel
from scripts.review_parc_decoder_height_response import capture_decoder_call


class TerrainLatentCondition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = LocalTerrainEncoder(width)
        self.attention = TerrainCrossAttention(width, 4)

    def forward(self, latent, observation):
        tokens, valid = self.encoder(observation, len(latent), latent.device, latent.dtype)
        return self.attention(latent.transpose(1, 2), tokens, valid).transpose(1, 2)


def vertical_origin(inference, history, observation):
    rep = inference.global_motion_rep
    frame = inference._pose_model.args.get('vertical_frame')
    if frame == TERRAIN_VERTICAL_FRAME:
        return observed_terrain_origin(history[:, 3, 1], observation)
    if frame == 'observed-pelvis-y-v1':
        return (history[:, 3, 1] - rep.stats.mean[1].to(history)).detach()
    if frame is not None:
        raise ValueError('Unsupported decoder experiment vertical frame')
    return history.new_zeros(len(history))


def training_targets(inference, source, history, observation, frames=32):
    """Keep source futures and actual history in one frame; do not transport truth."""
    if source.shape[1] != frames + 1 or history.shape != source[:, :4].shape:
        raise ValueError('Need source interval endpoints and exactly four observed frames')
    rep = inference.global_motion_rep
    origin = vertical_origin(inference, history, observation)
    target = shift_global_motion(source.detach(), rep, -origin)
    observed = shift_global_motion(history.detach(), rep, -origin)
    offset = history[:, :1, [0, 2]].detach()
    target[..., [0, 2]] -= offset
    observed[..., [0, 2]] -= offset
    local = inference.motion_rep.dual_rep.global_to_local(target, False, True,
        lengths=torch.full((len(target),), frames+1, device=target.device, dtype=torch.long))
    positions = rep.inverse(observed, False, return_all=False,
                            joint_positions_from='global_rot_data')['posed_joints']
    frame_mask = torch.arange(frames, device=target.device)[None] >= 4
    interval_mask = torch.arange(frames-1, device=target.device)[None] >= 3
    return {'local_motions': local[:, :frames].detach(),
            'global_motions': rep.normalize(target[:, :frames]).detach(),
            'executed_history_positions': positions.detach(),
            'motion_frame_mask': frame_mask.expand(len(target), -1),
            'motion_interval_mask': interval_mask.expand(len(target), -1)}


class DecoderExperiment:
    def __init__(self, inference, bundle, terrain=False):
        self.inference, self.terrain = inference, terrain
        self.codec = inference._vqvae_pose_model
        for model in (inference._root_model, inference._pose_model, self.codec):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.codec.decoder.requires_grad_(True)
        self.adapter = TerrainLatentCondition(self.codec.code_dim).to(inference.device)
        self.observation = None
        self.last_observation = None
        self.native_predict = inference.predict

        def predict(*args, **kwargs):
            observation = kwargs.get('terrain_observation')
            if observation is None:
                raise ValueError('Pilot needs the actual current terrain observation')
            self.observation = {k: observation[k].detach().clone() for k in ('points', 'valid')}
            return self.native_predict(*args, **kwargs)

        inference.predict = predict
        self.hook = self.codec.decoder.register_forward_pre_hook(self._condition)
        self.trainer = MotionVQVAEModel(self.codec, None, inference.motion_rep,
            optimizer=None, scheduler=None, args=deepcopy(bundle['tokenizer_config']['model']['args']))

    def _condition(self, _module, args):
        if not self.terrain:
            return args
        if self.observation is None:
            raise ValueError('Decoder called without a current observation')
        return (self.adapter(args[0], self.observation), *args[1:])

    def parameters(self):
        return list(self.codec.decoder.parameters()) + list(self.adapter.parameters())

    def request(self, history, goal, scene):
        with capture_decoder_call(self.codec) as calls:
            prediction, count = predict_planar_terrain(self.inference, history, goal, scene,
                num_tokens=8, config={'num_inference_step': 4, 'pose_token_sampling_use_argmax': True,
                                     'cache_terrain_encoder': False})
        if len(calls) != 1 or int(count.item()) != 8:
            raise ValueError('Unexpected native decoder call count or horizon')
        if calls[0]['kwargs']['has_target_cond'][:, 4:].any():
            raise ValueError('Future pose condition leaked into decoder request')
        calls[0]['observation'] = self.observation
        return prediction.detach(), calls[0]

    def replay(self, call):
        self.observation = call['observation']
        return self.codec.forward_decoder(*call['args'], **call['kwargs'])['recon_state']

    def restore_world(self, local_prediction, history, observation):
        frames = 32
        world = self.inference.motion_rep.dual_rep.local_to_global(
            local_prediction, True, False,
            lengths=torch.full((len(history),), frames, device=history.device, dtype=torch.long))
        offsets, heading, _ = self.inference._extract_initial_root_info(history[..., :5])
        world = self.inference._reapply_initial_root_info({
            'pred_global_poses': world, 'reference_start_root_global_offsets': offsets,
            'reference_start_root_global_heading': heading})
        return shift_global_motion(world, self.inference.global_motion_rep,
                                   vertical_origin(self.inference, history, observation))

    def loss(self, source, history, call):
        prediction = self.replay(call)[:, :32]
        target = training_targets(self.inference, source, history, call['observation'])
        output = {'recon_state': prediction, 'l_commit': prediction.sum()*0., 'perplexity': 0.}
        return self.trainer.loss(target, output, None), prediction

    def close(self):
        self.hook.remove()
        self.inference.predict = self.native_predict
