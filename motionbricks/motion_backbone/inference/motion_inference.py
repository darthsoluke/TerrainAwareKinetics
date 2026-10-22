from typing import Dict
import torch as t
import numpy as np
from motionbricks.motion_backbone.models.sampling import gumbel_sample
from motionbricks.motion_backbone.models.pose_model import MotionModel as pose_model_cls
from motionbricks.motion_backbone.models.root_model import MotionModel as root_model_cls
from motionbricks.vqvae.neural_modules.vqvae import VQVAE as vqvae

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

import copy

class motion_inference(t.nn.Module):
    """ @brief: For simplicity, we are mostly likely ONLY CONSIDER batch_size=1 cases
    """
    BATCH_SIZE = 1  # batch size > 1 supported as well

    # the feature type of `local_pose` provided to the `predict` function (externally provided)
    EXTERNAL_POSE_FEATURE_MODE = "joint_positions_and_rotations"
    # the feature type of `global_root` and `local_root` provided to the `predict` function (externally provided)
    EXTERNAL_ROOT_FEATURE_MODE = "root"
    # this is the feature of `local_pose` used by all three modes: vqvae, pose, root
    INTERNAL_POSE_FEATURE_MODE = "joint_positions_and_rotations_and_hip_height"
    EPS = 1e-5

    def __init__(self, models: list, args: Dict, device: str = 'cuda'):
        super(motion_inference, self).__init__()
        self._pose_model: pose_model_cls = models['pose'].eval().to(device)
        self._root_model: root_model_cls = models['root'].eval().to(device)
        self._vqvae_pose_model: vqvae = self._pose_model.supporting_nets['pose_net'].eval().to(device)

        self.global_motion_rep = self._pose_model.global_motion_rep
        self.local_motion_rep = self._pose_model.local_motion_rep
        self.motion_rep = self._pose_model.motion_rep  # the dual representation

        self._args = args
        self._device = device
        self._IS_ROOT_MODEL_TOKENIZED = self._root_model.backbone_net.IS_MODEL_TOKENIZED

        assert self._pose_model.backbone_net.initted and self._pose_model.vqvae_model_loaded \
            and self._root_model.vqvae_model_loaded, "The model should be initialized before inference."
        assert not self._IS_ROOT_MODEL_TOKENIZED, "The root model is not tokenized."

    def predict(self,
                global_root_values: t.Tensor, has_global_root_values: t.Tensor,
                local_root_values: t.Tensor, has_local_root_values: t.Tensor,
                local_poses: t.Tensor, has_local_poses: t.Tensor,
                num_tokens: t.Tensor,
                text_embeddings: t.Tensor = None, has_text_embeddings: t.Tensor = None,
                allowed_pred_num_tokens: t.Tensor = None,
                config: dict = {}, info: dict= {}, terrain_observation: dict = None,
                global_root_component_mask: t.Tensor = None, local_root_component_mask: t.Tensor = None):

        """ @param input_data: a dictionary containing the following keys:
        All these values are in unnormalized form. Note the root/pose information of the first 4 frames are always
        assumed to be given

        @input global_root_values: [batch, numConstrainFrames=8, 5]        # required
        @input has_global_root_values: [batch, numConstrainFrames=8]       # required

        @input local_root_values: [batch, numConstrainFrames=8, 4]         # required
        @input has_local_root_values: [batch, numConstrainFrames=8]        # required

        @input local_poses: [batch, numConstrainFrames=8, featdim]         # required
        @input has_local_pose: [batch, numConstrainFrames=8]               # required

        @input num_tokens: [batch, 1]                                      # optional

        """
        batch_size, device, dtype = has_global_root_values.shape[0], has_global_root_values.device, local_poses.dtype
        terrain_enabled = bool(self._root_model.backbone_net._args.get('terrain_conditioning', False))
        pose_terrain_enabled = bool(self._pose_model.backbone_net._args.get('terrain_conditioning', False))
        if terrain_enabled != pose_terrain_enabled:
            raise ValueError('Root and Pose must use the same terrain conditioning contract')
        if terrain_enabled != (terrain_observation is not None):
            raise ValueError('Terrain models require a current local observation; flat models do not consume it')
        online_root = self._root_model.args.get('online_conditioning')
        online_pose = self._pose_model.args.get('online_conditioning')
        if online_root != online_pose:
            raise ValueError('Root and Pose must use the same online condition version')
        if online_root:
            from motionbricks.data.native_mb_online_conditions import validate_online_inputs
            validate_online_inputs(global_root_values, has_global_root_values, global_root_component_mask,
                                   local_root_values, has_local_root_values, local_poses, has_local_poses, config,
                                   version=online_root, local_component_mask=local_root_component_mask)
        from motionbricks.data.native_mb_vertical_frame import validate_vertical_frame, center_sparse_conditions
        vertical_frame = validate_vertical_frame(self._pose_model.args.get('vertical_frame'))
        if vertical_frame != self._root_model.args.get('vertical_frame'):
            raise ValueError('Root and Pose vertical frames differ')
        vertical_origin = None
        if vertical_frame:
            if not online_root:
                raise ValueError('Observed vertical frame requires the causal online contract')
            global_root_values, local_poses, vertical_origin = center_sparse_conditions(
                global_root_values, local_poses, self.global_motion_rep, self.local_motion_rep,
                pose_feature_mode=self.EXTERNAL_POSE_FEATURE_MODE,
                vertical_frame=vertical_frame, terrain_observation=terrain_observation)
            local_root_values = local_root_values.clone()
            local_root_values[..., self.local_motion_rep.indices['global_root_y']] -= vertical_origin[:, None, None]
        if global_root_component_mask is not None:
            if not self._root_model.backbone_net._args.get('root_component_conditioning', False):
                raise ValueError('Component masks require a component-conditioned Root checkpoint')
            if (global_root_component_mask.shape != global_root_values.shape
                    or global_root_component_mask.dtype != t.bool):
                raise ValueError('Root component mask must be boolean and match the root values')
            global_root_values = t.where(global_root_component_mask, global_root_values,
                                         t.zeros_like(global_root_values))
            local_root_values = t.where(has_local_root_values[..., None], local_root_values,
                                        t.zeros_like(local_root_values))
            local_poses = t.where(has_local_poses[..., None], local_poses, t.zeros_like(local_poses))
        if local_root_component_mask is not None:
            local_root_values = t.where(local_root_component_mask, local_root_values, t.zeros_like(local_root_values))
        num_frames_per_token = self._pose_model.backbone_net.get_num_frames_per_token()
        if not (t.all(has_global_root_values[:, :num_frames_per_token]) and
                t.all(has_local_root_values[:, :num_frames_per_token]) and
                t.all(has_local_poses[:, :num_frames_per_token])):
            if not getattr(self, 'WARNING_PRINTED', False):
                print("WARNING: you are advised to provide all first 4 frames.")
                self.WARNING_PRINTED = True  # print warning only once
        if type(num_tokens) == int:
            num_tokens = t.full([batch_size, 1], num_tokens, dtype=t.int, device=device)
        elif num_tokens is None:  # indicate the length of the motion is not provided; needs to be predicted
            num_tokens = \
                self._root_model.backbone_net.MASKED_NUM_TOKENS * t.ones([batch_size, 1], dtype=t.int, device=device)

        # step 1: recenter the global root values so that the motions always start from (0.0f, 0.0f)
        batch = {}
        if terrain_enabled:
            # One bounded observation at the last history frame, reused across
            # denoising iterations. No future scene queries enter this call.
            if config.get('cache_terrain_encoder', False):
                if t.is_grad_enabled() or self._root_model.training or self._pose_model.training:
                    raise ValueError('Terrain encoder caching is inference-only')
                terrain_observation = dict(terrain_observation)
                terrain_observation['_inference_token_cache'] = {}
            batch['terrain_observation'] = terrain_observation

        batch['reference_start_root_global_offsets'], batch['reference_start_root_global_heading'], \
            recentered_global_root_values = self._extract_initial_root_info(global_root_values)

        batch['global_root_values'] = self.global_motion_rep.normalize(recentered_global_root_values)
        batch['local_root_values'] = self.local_motion_rep.normalize(local_root_values)
        if local_root_component_mask is not None:
            batch['local_root_component_mask'] = local_root_component_mask
            batch['local_root_values'] = t.where(local_root_component_mask, batch['local_root_values'],
                                                t.zeros_like(batch['local_root_values']))
        if global_root_component_mask is not None:
            batch['global_root_component_mask'] = global_root_component_mask
            batch['global_root_values'] = t.where(global_root_component_mask, batch['global_root_values'],
                                                   t.zeros_like(batch['global_root_values']))
            batch['local_root_values'] = t.where(has_local_root_values[..., None], batch['local_root_values'],
                                                  t.zeros_like(batch['local_root_values']))

        local_pose_feat_idx = extract_feature_from_motion_rep(
            t.zeros([1, 1, len(self.local_motion_rep.indices['all'])]),
            self.local_motion_rep, self.INTERNAL_POSE_FEATURE_MODE, fetch_feat_idx=True
        )
        global_height_values = \
            recentered_global_root_values[:, :, self.global_motion_rep.indices['global_root_pos'][[1]]]
        mean = self.local_motion_rep.stats.mean[None, None, local_pose_feat_idx].to(device=device, dtype=dtype)
        std = self.local_motion_rep.stats.std[None, None, local_pose_feat_idx].to(device=device, dtype=dtype)
        batch['local_poses'] = \
            (t.concat([global_height_values, local_poses], dim=-1) - mean) / t.sqrt(std ** 2 + self.EPS)
        if global_root_component_mask is not None:
            batch['local_poses'] = t.where(has_local_poses[..., None], batch['local_poses'],
                                            t.zeros_like(batch['local_poses']))

        batch['has_global_root_values'] = has_global_root_values
        batch['has_local_root_values'] = has_local_root_values
        batch['has_local_poses'] = has_local_poses
        batch['text_embeddings'] = text_embeddings
        batch['has_text_embeddings'] = has_text_embeddings
        batch['num_tokens'] = num_tokens
        batch['allowed_pred_num_tokens'] = allowed_pred_num_tokens

        # step 2: run the root model to predict the number of tokens and the root tokens
        batch['pred_num_tokens'], batch['pred_global_root_values'], batch['pred_local_root_values'] = \
            self._predict_root_trajectories(batch, config)

        # step 3: pred the pose tokens (1 iteration for now but also support multiple iterations)
        batch['pred_pose_tokens'], batch['pred_pose_cond'], batch['pred_has_pose_cond'] = \
              self._predict_pose_tokens(batch, config, info)

        # step 4: decode the pose tokens and root prediction to reconstruct the poses
        batch['pred_local_poses'], batch['pred_global_poses'] = \
            self._decode_motions_from_predicted_root_and_pose_tokens(batch, config, info)

        # step 5: apply the global root transforms to restore into the original world coordinates
        batch['pred_global_poses'] = self._reapply_initial_root_info(batch)
        if vertical_origin is not None:
            from motionbricks.data.native_mb_vertical_frame import shift_global_motion
            batch['pred_global_poses'] = shift_global_motion(
                batch['pred_global_poses'], self.global_motion_rep, vertical_origin)

        return batch['pred_global_poses'], batch['pred_num_tokens']

    def _sample_tokens_with_highest_prob(self, pose_tokens: t.Tensor, pose_tokens_prob: t.Tensor,
                                         pred_num_tokens: t.Tensor, step: int, num_pose_inference_steps: int):

        batch_size, device = pose_tokens.shape[0], pose_tokens.device
        rand_mask_prob = np.cos(float(step) / num_pose_inference_steps * np.pi * 0.5)
        num_pose_heads = self._pose_model.backbone_net.get_num_heads()[0]
        num_tokens = self._args['max_tokens'] * self._pose_model.backbone_net.get_num_heads()[0]

        num_pose_token_masked = t.clip((rand_mask_prob * pred_num_tokens *
                                        self._pose_model.backbone_net.get_num_heads()[0]).int(), min=1).view(batch_size, 1)

        # the padded tokens is not part of the masking process; give them +inf prob so that they are never masked
        pose_tokens_prob = t.where(t.arange(self._args['max_tokens'], device=device).view([1, -1, 1]) <
                                   pred_num_tokens.view([batch_size, 1, 1]).repeat([1, 1, num_pose_heads]),
                                   pose_tokens_prob, t.full_like(pose_tokens_prob, t.inf))

        # remove pose tokens with the least prob from pose_tokens_prob (sort)
        indices = pose_tokens_prob.view([batch_size, -1]).sort(descending=False)[1]
        tokens_to_be_masked = t.arange(num_tokens, device=device).tile([batch_size, 1]) < num_pose_token_masked
        indices = indices * tokens_to_be_masked + indices[:, :1] * (~tokens_to_be_masked)

        pose_tokens = pose_tokens.view([batch_size, -1]).clone()
        pose_tokens.scatter_(dim=-1, index=indices, value=self._pose_model.backbone_net.POSE_MASK_ID)
        return pose_tokens.view([batch_size, self._args['max_tokens'], -1])

    @property
    def device(self):
        return self._device

    def _extract_initial_root_info(self, global_root_values: t.Tensor):
        """ @brief: save the initial global root offsets and global heading information here so that we could
        cannonicalize the input to the network and de-canonicalize the output.
        NOTE: Since we are using a model where features are not relative to the root rotation transform,
        for the input, we don't rotate the root heading to 0.0f; we only move the root position to (0.0f, 0.0f).

        But for the output, since the `local_to_global` function does not have the initial heading, we log the initial
        heading and add back the heading to the output, despite headings are not used to reconstruct the character's
        joint transforms.
        """
        reference_start_root_global_offsets = \
            global_root_values[:, :1, self.global_motion_rep.indices['global_root_pos_2d']].clone()
        _, reference_start_root_global_heading = \
            self.global_motion_rep.compute_root_pos_and_rot(global_root_values[:, :1, :],
                                                            return_angle=True, return_quat=False)

        recentered_global_root_values = global_root_values.clone()
        recentered_global_root_values[:, :, self.global_motion_rep.indices['global_root_pos_2d']] -= \
            reference_start_root_global_offsets

        return reference_start_root_global_offsets, reference_start_root_global_heading, recentered_global_root_values

    def _reapply_initial_root_info(self, batch: dict):
        """ @brief: The follow-up function of `_extract_initial_root_info` to recover the global information.
        """
        _, root_rot_angle = self.global_motion_rep.compute_root_pos_and_rot(batch['pred_global_poses'],
                                                                            return_angle=True, return_quat=False)

        corrective_angle = batch['reference_start_root_global_heading'].reshape(root_rot_angle[..., 0].shape)
        new_angles = root_rot_angle + corrective_angle[..., None]  # [Batch, T]

        new_heading = t.stack([t.cos(new_angles), t.sin(new_angles)], dim=-1)

        batch['pred_global_poses'][:, :, self.global_motion_rep.indices['global_root_pos_2d']] += \
            batch['reference_start_root_global_offsets']
        batch['pred_global_poses'][:, :, self.global_motion_rep.indices['global_root_heading']] = new_heading
        return batch['pred_global_poses']

    def _predict_root_trajectories(self, batch: dict = {}, config: dict = {}):
        """ @brief: run the root module here to pred how many frames are between the keyframes and predict the local
        and global root trajectories.
        """
        # NOTE: the root module's output is the global root values; it has to be "root" in feature mode
        assert self._root_model.args['local_root_feature'] == self.EXTERNAL_ROOT_FEATURE_MODE, \
            self._root_model.args['global_root_feature'] == self.EXTERNAL_ROOT_FEATURE_MODE
        assert self._root_model.args['local_pose_feature'] == self.INTERNAL_POSE_FEATURE_MODE, \
            f"All submodules should only use {self.INTERNAL_POSE_FEATURE_MODE} as the local pose feature."
        assert self.EXTERNAL_ROOT_FEATURE_MODE == 'root', "Only support full root features for root model."

        num_frames_per_token = self._pose_model.backbone_net.get_num_frames_per_token()
        assert num_frames_per_token == self._root_model.backbone_net.get_num_frames_per_token()

        root_model_outputs = self._root_model.backbone_net(
            batch['global_root_values'], batch['has_global_root_values'],
            batch['local_root_values'], batch['has_local_root_values'],
            batch['local_poses'], batch['has_local_poses'], batch['num_tokens'],
            text_embeddings=batch['text_embeddings'], has_text_embeddings=batch['has_text_embeddings'],
            allowed_pred_num_tokens=batch['allowed_pred_num_tokens'],
            config=config,
            **({'terrain_observation': batch['terrain_observation']}
               if 'terrain_observation' in batch else {}),
            **({'global_root_component_mask': batch['global_root_component_mask']}
               if 'global_root_component_mask' in batch else {}),
            **({'local_root_component_mask': batch['local_root_component_mask']}
               if 'local_root_component_mask' in batch else {})
        )

        pred_num_tokens = root_model_outputs['pred_num_tokens']
        pred_global_root_values = root_model_outputs['pred_global_root_values']
        if config.get('debug_ground_truth_root_trajectories', None) is not None:
            pred_num_tokens[:] = config['debug_ground_truth_root_trajectories'].shape[1] // num_frames_per_token
            pred_global_root_values[:, :config['debug_ground_truth_root_trajectories'].shape[1]] = \
                config['debug_ground_truth_root_trajectories']  # both normalized
        pred_local_root_values = \
            self.motion_rep.dual_rep.global_to_local(pred_global_root_values, is_normalized=True, to_normalize=True,
                                                     lengths=pred_num_tokens * num_frames_per_token)
        if 'local_root_component_mask' in batch:
            from motionbricks.data.native_mb_online_conditions import apply_terminal_velocity
            pred_local_root_values = apply_terminal_velocity(pred_local_root_values, batch['local_root_values'],
                batch['local_root_component_mask'], pred_num_tokens * num_frames_per_token)
            return pred_num_tokens, pred_global_root_values, pred_local_root_values

        # now be very careful with the final root motion; since the raw calculation could be very incorrect
        estimated_final_local_root_motion = \
            t.gather(pred_local_root_values, 1,
                     pred_num_tokens.long().repeat([1, 4])[:, None, :] * num_frames_per_token - 1)  # why was this -2?
        local_root_feat_dim = len(self.local_motion_rep.indices['root'])
        final_local_root_motion = \
            batch['has_local_root_values'][:, -1:, None].expand([-1, -1, local_root_feat_dim]).float() * \
            batch['local_root_values'][:, -1:] + \
            (1 - batch['has_local_root_values'][:, -1:, None].expand([-1, -1, local_root_feat_dim]).float()) * \
            estimated_final_local_root_motion
        pred_local_root_values = pred_local_root_values.scatter(
            1, pred_num_tokens.long().repeat([1, 4])[:, None, :] * num_frames_per_token - 1, final_local_root_motion
        )  # replace the last velocity with the second last velocity

        return pred_num_tokens, pred_global_root_values, pred_local_root_values

    def _predict_pose_tokens(self, batch: dict = {}, config: dict = {}, info: dict = {}):
        """ @brief: run the pose module here to pred the pose tokens."""
        # NOTE: the root module's output is the global root values; it has to be "root" in feature mode
        assert self._pose_model.args['local_pose_feature'] == self.INTERNAL_POSE_FEATURE_MODE
        assert self.INTERNAL_POSE_FEATURE_MODE == "joint_positions_and_rotations_and_hip_height"
        assert (self._pose_model.args['cond_root_feature_is_from_motion_rep'] == 'local' and
                self._pose_model.args['cond_root_feature'] == 'root_without_hip_height_without_heading') or \
            (self._pose_model.args['cond_root_feature_is_from_motion_rep'] == 'global' and
             self._pose_model.args['cond_root_feature'] == 'root_without_hip_height'), \
            "These are the only two root cond feature combination supported."

        # collect the configs and construct initial input data
        num_pose_heads, num_frames_per_token = \
            self._pose_model.backbone_net.get_num_heads()[0], self._pose_model.backbone_net.get_num_frames_per_token()
        batch_size, device = batch['local_poses'].shape[0], batch['local_poses'].device
        num_pose_inference_steps = config.get('num_inference_step', 1)

        pose_tokens = t.full([batch_size, self._args['max_tokens'], num_pose_heads],
                              self._pose_model.backbone_net.POSE_MASK_ID, device=device)
        chosen_pose_tokens_prob = \
            t.full([batch_size, self._args['max_tokens'], num_pose_heads], 1.0 / num_pose_heads, device=device)

        pose_cond = t.concat([batch['local_poses'][:, :num_frames_per_token],
                              t.zeros([batch_size, (self._args['max_tokens'] - 1) * num_frames_per_token,
                                       batch['local_poses'].shape[-1]], device=device)], dim=1)
        has_pose_cond = \
            t.concat([batch['has_local_poses'][:, :num_frames_per_token],
                      t.zeros([batch_size, (self._args['max_tokens'] - 1) * num_frames_per_token],
                               dtype=bool, device=device)], dim=1)

        for i in range(num_frames_per_token):  # assign the target pose information dynamically
            onehot_idx = t.nn.functional.one_hot(
                batch['pred_num_tokens'].view([-1]).long() * num_frames_per_token - num_frames_per_token + i,
                num_classes=self._args['max_tokens'] * num_frames_per_token
            )
            pose_cond = pose_cond + \
                onehot_idx.float().view([batch_size, -1, 1]) * \
                batch['local_poses'][:, -num_frames_per_token + i].view([batch_size, 1, -1])
            has_pose_cond = t.logical_or(
                has_pose_cond, (onehot_idx.float().view([batch_size, -1]) *
                                batch['has_local_poses'][:, -num_frames_per_token + i].view([batch_size, 1])).bool()
            )

        for step in range(num_pose_inference_steps):
            pose_tokens = self._sample_tokens_with_highest_prob(pose_tokens, chosen_pose_tokens_prob,
                                                                batch['pred_num_tokens'],
                                                                step, num_pose_inference_steps)
            assert self._args['cond_root_feature_is_from_motion_rep'] in ['local', 'global']
            if self._args['cond_root_feature_is_from_motion_rep'] == 'local':
                pose_root_cond = \
                    extract_feature_from_motion_rep(batch['pred_local_root_values'], self.local_motion_rep,
                                                   self._pose_model.args['cond_root_feature'])
            else:
                assert self._args['cond_root_feature_is_from_motion_rep'] == 'global'
                pose_root_cond = \
                    extract_feature_from_motion_rep(batch['pred_global_root_values'], self.global_motion_rep,
                                                   self._pose_model.args['cond_root_feature'])

            terrain_kwargs = {}
            if 'terrain_observation' in batch:
                from motionbricks.data.native_mb_terrain_batch import root_height_condition
                terrain_kwargs['terrain_observation'] = batch['terrain_observation']
                terrain_kwargs['root_height_trajectory'] = root_height_condition(
                    batch['pred_global_root_values'], self.global_motion_rep,
                    context_frames=num_frames_per_token)
            pose_model_output = self._pose_model.backbone_net(
                pose_tokens, pose_root_cond, pose_cond, has_pose_cond,
                batch['pred_num_tokens'], batch['text_embeddings'], batch['has_text_embeddings'],
                **terrain_kwargs
            )
            if config.get('pose_token_sampling_use_argmax', False):
                pose_tokens = pose_model_output['pose_logits'].argmax(dim=-1)
            else:
                pose_tokens = gumbel_sample(pose_model_output['pose_logits'], temperature=1.0)
            pose_tokens_prob = pose_model_output['pose_logits'].softmax(dim=-1)
            chosen_pose_tokens_prob = pose_tokens_prob.gather(dim=-1, index=pose_tokens.unsqueeze(-1)).squeeze(-1)

        if config.get('debug_ground_truth_pose_tokens', None) is not None:
            pose_tokens[:, :config['debug_ground_truth_pose_tokens'].shape[1]] = \
                config['debug_ground_truth_pose_tokens'][:, :]

        return pose_tokens, pose_cond, has_pose_cond  # pose cond and has_pose cond are re-used in the decoder

    def _decode_motions_from_predicted_root_and_pose_tokens(self, batch: dict = {}, config: dict = {}, info: dict = {}):
        """ @brief: decode the pose tokens and root prediction to reconstruct the poses"""
        if getattr(self._pose_model.args, 'pose_vqvae_motion_rep', 'local'):
            assert self._vqvae_pose_model.motion_rep.name == 'local'
            external_mode = self._vqvae_pose_model.decoder_external_cond_feature_mode
            if self._pose_model.args.get('codec_height_conditioning'):
                from motionbricks.data.native_mb_codec_height import CODEC_HEIGHT_CONDITIONING
                if (self._pose_model.args['codec_height_conditioning'] != CODEC_HEIGHT_CONDITIONING
                        or not self._pose_model.args.get('online_conditioning')
                        or not self._pose_model.backbone_net._args.get('terrain_conditioning')):
                    raise ValueError('Dense codec height requires a declared matched online terrain pipeline')
                assert external_mode == 'root_without_heading'
            else:
                assert external_mode == 'root_without_hip_height_without_heading'
        else:
            assert self._vqvae_pose_model.motion_rep.name == 'global'
            assert self._vqvae_pose_model.decoder_external_cond_feature_mode == \
                "root_without_hip_height_without_heading_with_mask"

        assert self._vqvae_pose_model.decoder_target_cond_feature_mode == self.INTERNAL_POSE_FEATURE_MODE

        pose_external_root_cond = extract_feature_from_motion_rep(
            batch['pred_local_root_values'] if self._vqvae_pose_model.motion_rep.name == 'local' \
                else batch['pred_global_root_values'],
            self._vqvae_pose_model.motion_rep, self._vqvae_pose_model.decoder_external_cond_feature_mode
        )
        if self._vqvae_pose_model.motion_rep.name == 'global':
            raise NotImplementedError("The global root feature is not supported yet.")

        batch_size, device = batch['local_poses'].shape[0], batch['local_poses'].device
        pose_token_mask = t.arange(self._args['max_tokens'], device=device)[None, :].repeat([batch_size, 1]) < \
            batch['pred_num_tokens'].view([batch_size, 1])

        if type(config.get("use_target_cond_in_decoder", True)) == t.Tensor:
            # if a False tensor is provided, mask out all target condition here
            global_target_cond = config["use_target_cond_in_decoder"].bool().expand_as(batch['pred_has_pose_cond'])
            has_target_cond = t.logical_and(batch['pred_has_pose_cond'], global_target_cond)
        else:
            assert type(config.get("use_target_cond_in_decoder", True)) == bool
            has_target_cond = batch['pred_has_pose_cond'] \
                if config.get("use_target_cond_in_decoder", True) else t.zeros_like(batch['pred_has_pose_cond']).bool()
        if not config.get('use_constraints_at_decoder', True):
            has_target_cond = t.zeros_like(has_target_cond).bool()  # disable the target cond
        if config.get('skip_ending_target_cond', False):
            # skip the ending target cond; useful for inprecise ending target cond; can improve the foot steps
            num_frames_per_token = self._pose_model.backbone_net.get_num_frames_per_token()
            has_target_cond[:, num_frames_per_token:] = False

        if self._pose_model.args.get('online_conditioning'):
            # Old codecs saw at most two arbitrary keyframes. Only artifacts
            # explicitly trained on played context may consume all four here.
            context_policy = self._pose_model.args.get('codec_context_policy', 'native_random')
            if context_policy not in {'native_random', 'played_context4'}:
                raise ValueError('Unknown decoder context policy')
            current_pose_mask = t.zeros_like(has_target_cond)
            context_frames = self._pose_model.backbone_net.get_num_frames_per_token()
            if context_policy == 'played_context4':
                if not (self._pose_model.args.get('codec_height_conditioning')
                        or self._pose_model.args.get('matched_stage_adaptation')):
                    raise ValueError('Played decoder context requires a bound adapted codec')
                current_pose_mask[:, :context_frames] = True
            else:
                current_pose_mask[:, context_frames - 1] = True
            has_target_cond = has_target_cond & current_pose_mask

        pred_poses = self._vqvae_pose_model.forward_decoder(batch['pred_pose_tokens'],
                                                            target_cond=batch['pred_pose_cond'],
                                                            has_target_cond=has_target_cond,
                                                            external_cond=pose_external_root_cond,
                                                            use_overall_indices=False,
                                                            token_mask=pose_token_mask)['recon_state']

        num_pred_frames = batch['pred_num_tokens'] * self._pose_model.backbone_net.get_num_frames_per_token()
        if self._vqvae_pose_model.motion_rep.name == 'local':
            # NOTE: the pred_global_poses has incorrect heading since the accumlation function of `local_to_global`
            # does not have initial heading information; luckily the inverse function which reconstruct the final
            # results will not use the heading feature at all so it won't affect the final pose outputs
            pred_global_poses = self.motion_rep.dual_rep.local_to_global(pred_poses, is_normalized=True,
                                                                         to_normalize=False, lengths=num_pred_frames)
            pred_local_poses = self.local_motion_rep.unnormalize(pred_poses)
        else:
            pred_local_poses = self.motion_rep.dual_rep.global_to_local(pred_poses, is_normalized=True,
                                                                        to_normalize=False, lengths=num_pred_frames)
            pred_global_poses = self.global_motion_rep.unnormalize(pred_poses)

        if config.get('final_root_pred_mode', 'from_pose_module') == 'from_pose_module':
            # use the root prediction from the pose module; do nothing here
            pass
        elif config['final_root_pred_mode'] == 'from_root_module':
            # use the direct root prediction from the root module
            local_root_index, global_root_index = \
                self.local_motion_rep.indices['root'], self.global_motion_rep.indices['root']

            pred_global_poses[:, :, global_root_index] = \
                self.global_motion_rep.unnormalize(batch['pred_global_root_values'])
            pred_local_poses[:, :, local_root_index] = \
                self.local_motion_rep.unnormalize(batch['pred_local_root_values'])
        else:
            raise NotImplementedError(f"Not supported yet {config['final_root_pred_mode']}.")
        return pred_local_poses, pred_global_poses
