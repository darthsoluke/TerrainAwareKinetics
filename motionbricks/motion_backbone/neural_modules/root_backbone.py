import torch as t
import torch
from torch import nn
from typing import Dict, Optional
import numpy as np
from motionbricks.motion_backbone.neural_modules.position_embedding import PositionEmbedding
from motionbricks.motionlib.core.motion_reps import MotionRepBase
from motionbricks.motion_backbone.neural_modules.mlp import FCBlock as mlp
from functools import cached_property
from motionbricks.vqvae.neural_modules.encdec_double_cond import DoubleCondDecoder
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.neural_modules.local_terrain_encoder import LocalTerrainEncoder, TerrainCrossAttention

class root_backbone_network(nn.Module):
    def __init__(self, args: Dict, motion_rep: MotionRepBase):
        """ @brief: root backbone network for motion generation.
        """
        super().__init__()
        self._args = args

        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep

        self.IS_MODEL_TOKENIZED = False
        self._build_backbone()
        self._build_input_embeddings()
        if self._args.get('terrain_conditioning', False):
            self._terrain_encoder = LocalTerrainEncoder(self._args['n_embd'])
            self._terrain_cross_attention = TerrainCrossAttention(self._args['n_embd'], self._args['n_head'])

    def _build_backbone(self):
        """ @brief: the transformer backbone for the lp. The input is the embeddings of the start and end frames and
        the target root transform
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=self._args['n_embd'],
                                                   nhead=self._args['n_head'], batch_first=True)
        self._shared_transformer_model = nn.TransformerEncoder(encoder_layer, num_layers=self._args['n_layers_shared'])

        if self._args['n_layers_root_token'] > 0:
            self._root_token_transformer_model = \
                nn.TransformerEncoder(encoder_layer, num_layers=self._args['n_layers_root_token'])
        else:
            self._root_token_transformer_model = None
        self._args['local_root_dim'] = extract_feature_from_motion_rep(t.zeros([1, 1, 1000]), self.local_motion_rep,
                                                                      self._args['local_root_feature']).shape[2]
        self._args['global_root_dim'] = extract_feature_from_motion_rep(t.zeros([1, 1, 1000]), self.global_motion_rep,
                                                                       self._args['global_root_feature']).shape[2]
        self._args['local_pose_dim'] = extract_feature_from_motion_rep(t.zeros([1, 1, 1000]), self.local_motion_rep,
                                                                      self._args['local_pose_feature']).shape[2]

        self._conv_output = DoubleCondDecoder(
            self._args['global_root_dim'], self._args['n_embd'],
            down_t=self._args['down_t'], width=self._args['width'], depth=self._args['depth'],
            dilation_growth_rate=self._args['dilation_growth_rate'],
            activation=self._args['activation'], norm=self._args['norm'],
            target_cond_dim=self._args['global_root_dim'],  # global root information
            external_cond_dim=self._args['n_embd'],  # frame emb from root & pose
            target_component_conditioning=self._args.get('root_component_conditioning', False)
        )

    def _build_input_embeddings(self):
        """ @brief:
        @input global_root_values: [batch, numConstrainFrames=[0, 8], 5]   # optional, also used in root decoder
        @input has_global_root_values: [batch, 8]                          # required

        @input local_root_values: [batch, numConstrainFrames=[0, 8], 4]    # optional
        @input has_local_root_values: [batch, 8]                          # required

        @input start_pose: [batch, numConstrainFrames=4, featdim]          # required
        @input has_start_pose: [batch, 4] (bool)                           # required; whether start_pose is provided

        @input target_pose: [batch, numConstrainFrames=[0, 4], featdim]    # optional
        @input has_target_pose: [batch, 4] (bool)                          # required; whether target_pose is provided

        @input num_tokens: [batch, 1]                                      # optional
        """
        num_frames_per_token = self.get_num_frames_per_token()

        # input emb projection
        self._proj_local_pose = nn.Linear(self._args['local_pose_dim'], self._args['pose_feat_dim'])
        self._proj_local_root_value = nn.Linear(self.args['local_root_dim'],
                                                self._args['local_root_feat_dim'])
        self._proj_global_root_value = nn.Linear(self.args['global_root_dim'],
                                                 self._args['global_root_feat_dim'])
        if self._args.get('root_component_conditioning', False):
            self._proj_global_root_component_mask = nn.Linear(
                self._args['global_root_dim'], self._args['global_root_feat_dim'], bias=False
            )
            nn.init.zeros_(self._proj_global_root_component_mask.weight)
        if self._args.get('local_component_conditioning', False):
            self._proj_local_root_component_mask = nn.Linear(
                self._args['local_root_dim'], self._args['local_root_feat_dim'], bias=False)
            nn.init.zeros_(self._proj_local_root_component_mask.weight)

        # emb when the constraints are not given
        self._no_local_pose_emb = nn.Parameter(torch.randn([self._args['pose_feat_dim']]))
        self._no_local_root_emb = nn.Parameter(torch.randn([self._args['local_root_feat_dim']]))
        self._no_global_root_emb = nn.Parameter(torch.randn([self._args['global_root_feat_dim']]))

        self._conv_no_frame_emb = nn.Parameter(torch.randn([self._args['n_embd']]))

        proj_input_dim = self._args['pose_feat_dim'] + \
            self._args['local_root_feat_dim'] + self._args['global_root_feat_dim']
        self._proj_start_input = mlp(num_layers=self._args['input_feat_mlp_num_layers'],
                                     layer_width=self._args['n_embd'],
                                     size_in=proj_input_dim, size_out=self._args['n_embd'])
        self._proj_end_input = mlp(num_layers=self._args['input_feat_mlp_num_layers'],
                                   layer_width=self._args['n_embd'],
                                   size_in=proj_input_dim, size_out=self._args['n_embd'])
        self._input_position_emb = t.nn.Embedding(num_frames_per_token * 2, self._args['n_embd'])

        # the num_token emd
        self._proj_input_num_tokens = t.nn.Embedding(
            self._args['max_tokens'] - self._args['min_tokens'] + 1 + 1 + 1, self._args['n_embd']
        )  # the num_token input include [min_tokens, max_tokens] and [out of reach (max_token + 1)] and [mask]
        self._position_emb = PositionEmbedding(seq_length=self._args['max_tokens'],
                                               dim=self._args['n_embd'])  # the std for this fixed position emb is 0.5

        # output projection; the num_token input include [min_tokens, max_tokens] and [out of reach (max_token + 1)]
        self._proj_num_token_output_logit = t.nn.Linear(self._args['n_embd'],
                                                        self._args['max_tokens'] - self._args['min_tokens'] + 1 + 1)

        # step 4: (optional)
        if self.ACCEPT_TEXT_EMB_INPUT:
            self._proj_text_embeddings = mlp(num_layers=self._args['input_feat_mlp_num_layers'],
                                             layer_width=self._args['n_embd'],
                                             size_in=self._args['text_emb_dim'], size_out=self._args['n_embd'])
        if self.USE_HARD_NUM_TOKEN_EMB_FOR_ROOT_PREDICTION and self._root_token_transformer_model is not None:
            self._middle_token_emb = t.nn.Embedding(self._args['max_tokens'] - self._args['min_tokens'] + 1 + 1,
                                                    self._args['n_embd'])

    def forward(self, global_root_values: t.Tensor, has_global_root_values: t.Tensor,
                local_root_values: t.Tensor, has_local_root_values: t.Tensor,
                poses: t.Tensor, has_poses: t.Tensor, num_tokens: t.Tensor,
                text_embeddings: Optional[t.Tensor] = None, has_text_embeddings: Optional[t.Tensor] = None,
                groundtruth_num_tokens: t.Tensor = None,
                allowed_pred_num_tokens: t.Tensor = None, config: dict = {}, terrain_observation=None,
                global_root_component_mask: Optional[t.Tensor] = None,
                local_root_component_mask: Optional[t.Tensor] = None):
        """ @brief:
        @input global_root_values: [batch, numConstrainFrames=[0, 8], 5]   # required, also used in root decoder
        @input has_global_root_values: [batch, 8]                          # required

        @input local_root_values: [batch, numConstrainFrames=[0, 8], 4]    # required
        @input has_local_root_values: [batch, 8]                           # required

        @input pose: [batch, numConstrainFrames=8, featdim]                # required
        @input has_pose: [batch, 8] (bool)                                 # required; whether start_pose is provided

        @input num_tokens: [batch, 1]                                      # optional

        groundtruth_num_tokens: [batch, 1]                                 # optional; only valid in training
        """
        if terrain_observation is not None and not self._args.get('terrain_conditioning', False):
            raise ValueError("terrain_observation requires terrain_conditioning=True.")
        if global_root_component_mask is not None:
            if not self._args.get('root_component_conditioning', False):
                raise ValueError("global_root_component_mask requires root_component_conditioning=True.")
            if global_root_component_mask.dtype != torch.bool:
                raise ValueError("global_root_component_mask must have bool dtype.")
            expected_shape = (global_root_values.shape[0], 8, self._args['global_root_dim'])
            if tuple(global_root_component_mask.shape) != expected_shape:
                raise ValueError(f"global_root_component_mask must have shape {expected_shape}.")
            if tuple(global_root_values.shape) != expected_shape:
                raise ValueError(f"global_root_values must have shape {expected_shape} when component masking is used.")
            if tuple(has_global_root_values.shape) != expected_shape[:2]:
                raise ValueError(f"has_global_root_values must have shape {expected_shape[:2]}.")
            if has_global_root_values.dtype != torch.bool:
                raise ValueError("has_global_root_values must have bool dtype when component masking is used.")
            component_presence = global_root_component_mask.any(dim=-1)
            if not torch.equal(has_global_root_values, component_presence):
                raise ValueError("has_global_root_values must equal global_root_component_mask.any(dim=-1).")
            global_root_values = torch.where(
                global_root_component_mask, global_root_values, torch.zeros_like(global_root_values)
            )
        batch_size, device = poses.shape[0], poses.device
        if local_root_component_mask is not None:
            if (not self._args.get('local_component_conditioning', False)
                    or local_root_component_mask.dtype != t.bool
                    or local_root_component_mask.shape != local_root_values.shape
                    or not t.equal(has_local_root_values, local_root_component_mask.any(-1))):
                raise ValueError('Invalid partial local Root conditions')
            local_root_values = t.where(local_root_component_mask, local_root_values, t.zeros_like(local_root_values))
        num_frames_per_token = self.get_num_frames_per_token()

        # step 1: construct the frame embedding and initial time emb
        local_pose_emb = self._proj_local_pose(poses)                          # [batch, numFrames, dim]
        local_root_emb = self._proj_local_root_value(local_root_values)        # [batch, numFrames, dim]
        if local_root_component_mask is not None:
            local_root_emb = local_root_emb + self._proj_local_root_component_mask(
                (~local_root_component_mask).to(local_root_values.dtype))
        global_root_emb = self._proj_global_root_value(global_root_values)     # [batch, numFrames, dim]
        if global_root_component_mask is not None:
            global_root_emb = global_root_emb + self._proj_global_root_component_mask(
                (~global_root_component_mask).to(dtype=global_root_values.dtype)
            )

        local_pose_emb = local_pose_emb * has_poses[:, :, None].float() + \
            self._no_local_pose_emb[None, None, :] * (1 - has_poses[:, :, None].float())
        local_root_emb = local_root_emb * has_local_root_values[:, :, None].float() + \
            self._no_local_root_emb[None, None, :] * (1 - has_local_root_values[:, :, None].float())
        global_root_emb = global_root_emb * has_global_root_values[:, :, None].float() + \
            self._no_global_root_emb[None, None, :] * (1 - has_global_root_values[:, :, None].float())

        start_frame_emb = torch.cat([local_pose_emb[:, :num_frames_per_token, :],
                                     local_root_emb[:, :num_frames_per_token, :],
                                     global_root_emb[:, :num_frames_per_token, :]], dim=-1)
        start_frame_emb = self._proj_start_input(start_frame_emb)
        end_frame_emb = torch.cat([local_pose_emb[:, -num_frames_per_token:, :],
                                   local_root_emb[:, -num_frames_per_token:, :],
                                   global_root_emb[:, -num_frames_per_token:, :]], dim=-1)
        end_frame_emb = self._proj_end_input(end_frame_emb)
        frame_emb = torch.cat([start_frame_emb, end_frame_emb], dim=1)
        positioned_frame_emb = frame_emb + self._input_position_emb.weight[None, :, :]

        first_stage_time_emb = self._proj_input_num_tokens(num_tokens - self._args['min_tokens'])  # [batch, dim]
        first_stage_time_emb = first_stage_time_emb.view([batch_size, 1, self._args['n_embd']])

        # step 2: first stage transformer forward for the num token prediction
        first_stage_input_emb = t.concat([first_stage_time_emb, positioned_frame_emb], dim=1)

        terrain_tokens = terrain_valid = None
        if self._args.get('terrain_conditioning', False) and terrain_observation is not None:
            terrain_tokens, terrain_valid = self._terrain_encoder(
                terrain_observation, batch_size, device, first_stage_input_emb.dtype
            )
            first_stage_input_emb = self._terrain_cross_attention(
                first_stage_input_emb, terrain_tokens, terrain_valid
            )

        if self.ACCEPT_TEXT_EMB_INPUT and text_embeddings is not None:
            text_embeddings = self._proj_text_embeddings(text_embeddings)[:, None, :]
            first_stage_input_emb = \
                t.concat([text_embeddings, first_stage_input_emb], dim=1)  # [batch, 1+num_positions, n_embd]
            first_stage_mask = t.zeros([batch_size, 1 + 1 + num_frames_per_token * 2],
                                       device=device).bool()  # zeros mean allowing
            first_stage_mask[:, :1] = ~has_text_embeddings.bool()
        else:
            first_stage_mask = None

        first_stage_output_emb = self._shared_transformer_model(first_stage_input_emb,
                                                                src_key_padding_mask=first_stage_mask)
        if terrain_tokens is not None:
            first_stage_output_emb = self._terrain_cross_attention(
                first_stage_output_emb, terrain_tokens, terrain_valid
            )
        num_token_logits = \
            self._proj_num_token_output_logit(first_stage_output_emb[:, self.TRANSFORMER_TIME_LOGIT_ID, :])

        assert self.USE_HARD_NUM_TOKEN_EMB_FOR_ROOT_PREDICTION, "Only support hard num token emb."
        if groundtruth_num_tokens is not None:
            chosen_token = groundtruth_num_tokens.view([batch_size]) - self._args['min_tokens']
        else:
            assert not self.training, "groundtruth_num_tokens is required in training."
            if not config.get('allow_pred_out_of_reach_num_tokens', True):
                # erase the prob for predicting out of reach token (OOR token)
                num_token_logits = num_token_logits.clone()
                num_token_logits[:, self.OUT_OF_REACH_NUM_TOKENS - self._args['min_tokens']] = -t.inf
            if allowed_pred_num_tokens is not None:
                # only the chosen token is allowed to be predicted
                num_time_tokens = self._args['max_tokens'] - self._args['min_tokens'] + 1
                modified_num_token_logits = t.where(allowed_pred_num_tokens == 1,
                                                    num_token_logits[:, :num_time_tokens],
                                                    t.full([batch_size, num_time_tokens], -t.inf).to(device))

                num_token_logits = t.cat([modified_num_token_logits, num_token_logits[:, num_time_tokens:]], dim=-1)

            chosen_token = torch.argmax(num_token_logits, dim=-1).int()
            chosen_token = t.where(num_tokens.view([batch_size]) == self.MASKED_NUM_TOKENS,
                                   chosen_token, num_tokens.view([batch_size]) - self._args['min_tokens'])

        # step 3: the root global value transformer
        position_ids = t.arange(self._args['max_tokens'], device=device)
        position_emb = self._position_emb.embed[position_ids].view([1, self._args['max_tokens'], self._args['n_embd']])
        position_emb = position_emb.expand([batch_size, -1, -1])
        token_mask = t.arange(self._args['max_tokens'], device=device).view([1, -1]) < \
            chosen_token.view([-1, 1]) + self._args['min_tokens']

        if self._root_token_transformer_model is None:
            second_stage_output_emb = position_emb
        else:
            second_stage_token_emb = self._middle_token_emb(chosen_token)
            second_stage_input_emb = t.concat([second_stage_token_emb[:, None, :],
                                               positioned_frame_emb, position_emb], dim=1)
            second_stage_mask = t.cat([t.zeros([batch_size, 1 + num_frames_per_token * 2],
                                                device=device).bool(),  # num of token emb, frame emb
                                       ~token_mask], dim=1)
            second_stage_output_emb = \
                self._root_token_transformer_model(second_stage_input_emb, src_key_padding_mask=second_stage_mask)
            second_stage_output_emb = \
                second_stage_output_emb[:, -self._args['max_tokens']:]  # remove frame and num_token emb

        if terrain_tokens is not None:
            second_stage_output_emb = self._terrain_cross_attention(
                second_stage_output_emb, terrain_tokens, terrain_valid
            )

        # step 4: the root global value conv output
        pred_num_tokens = chosen_token + self._args['min_tokens']
        batch = {}
        keys = ['frame', 'global_root']
        num_total_frames = self._args['max_tokens'] * num_frames_per_token
        batch['has_frame'] = t.logical_or(t.logical_or(has_poses, has_global_root_values), has_local_root_values)
        batch['frame'] = t.where(batch['has_frame'][:, :, None], frame_emb, self._conv_no_frame_emb[None, None, :])
        batch['dense_frame'] = t.cat([batch['frame'][:, :num_frames_per_token],
                                      self._conv_no_frame_emb[None, None, :].repeat([batch_size, num_total_frames -
                                                                                     num_frames_per_token, 1])], dim=1)

        batch['global_root'], batch['has_global_root'] = global_root_values, has_global_root_values
        batch['dense_global_root'] = t.cat([batch['global_root'][:, :num_frames_per_token],
                                            t.zeros([batch_size, num_total_frames - num_frames_per_token,
                                                     self._args['global_root_dim']], device=device)], dim=1)
        if global_root_component_mask is not None:
            batch['dense_global_root_component_mask'] = t.cat([
                global_root_component_mask[:, :num_frames_per_token],
                t.zeros([batch_size, num_total_frames - num_frames_per_token,
                         self._args['global_root_dim']], device=device, dtype=torch.bool)
            ], dim=1)

        for key in keys:  # construct the full batch from sparse ones
            batch['dense_has_' + key] = \
                t.cat([batch['has_' + key][:, :num_frames_per_token],
                       t.zeros([batch_size, num_total_frames - num_frames_per_token], device=device).bool()], dim=1)
            for i in range(num_frames_per_token):
                offsets = (pred_num_tokens[:, None] * num_frames_per_token - num_frames_per_token + i).long()
                batch['dense_' + key] = \
                    batch['dense_' + key].scatter(1, offsets[:, :, None].repeat([1, 1, batch[key].shape[-1]]),
                                                  batch[key][:, -num_frames_per_token + i][:, None, :])
                batch['dense_has_' + key] = \
                    batch['dense_has_' + key].scatter(1, offsets[:, :],
                                                      batch['has_' + key][:, -num_frames_per_token + i][:, None])
                if key == 'global_root' and global_root_component_mask is not None:
                    batch['dense_global_root_component_mask'] = \
                        batch['dense_global_root_component_mask'].scatter(
                            1, offsets[:, :, None].expand(-1, -1, self._args['global_root_dim']),
                            global_root_component_mask[:, -num_frames_per_token + i][:, None, :]
                        )

        if global_root_component_mask is not None:
            batch['dense_has_global_root'] = batch['dense_global_root_component_mask'].any(dim=-1)

        decoder_kwargs = {}
        if global_root_component_mask is not None:
            decoder_kwargs['target_component_mask'] = batch['dense_global_root_component_mask']
        pred_global_root_values = self._conv_output(
            second_stage_output_emb.transpose(1, 2), external_cond=batch['dense_frame'],
            target_cond=batch['dense_global_root'],
            has_target_cond=batch['dense_has_global_root'], token_mask=token_mask, **decoder_kwargs
        ).transpose(1, 2)  # NOTE: only dense_has_global_root is used for no-show cond. Frame-cond uses non-emb instead

        return {'num_token_logits': num_token_logits, 'pred_num_tokens': pred_num_tokens,
                'pred_global_root_values': pred_global_root_values}

    def get_num_frames_per_token(self):
        return 2 ** self._args['down_t']

    @property
    def args(self):
        return self._args

    def get_num_heads(self):
        """ brief: num heads for the tokenizers """
        return self._args['pose_vqvae']['num_heads'], self._args['root_vqvae']['num_heads']

    def get_num_codes(self, include_aug_tokens: bool):
        num_codes_per_head = {}

        for vqvae in ['pose_vqvae', 'root_vqvae']:  # calculate nb_code per head from overall nb_codes
            nb_code = self._args[vqvae]['nb_code']
            num_heads = self._args[vqvae]['num_heads']
            code_dim = self._args[vqvae]['code_dim']
            assert code_dim % num_heads == 0, "code dim cannot be divided by the number of heads."
            num_codes_per_head[vqvae] = int(round(2 ** (np.log2(nb_code) / num_heads)))
            assert num_codes_per_head[vqvae] ** num_heads == nb_code, \
                "the specified number of code is not compatible with the number of heads."

        if include_aug_tokens:  # include a [MASK] token
            return num_codes_per_head['pose_vqvae'] + 1, num_codes_per_head['root_vqvae'] + 1
        else:
            return num_codes_per_head['pose_vqvae'], num_codes_per_head['root_vqvae']

    @property
    def OUT_OF_REACH_NUM_TOKENS(self):
        return self._args['max_tokens'] + 1

    @property
    def MASKED_NUM_TOKENS(self):
        return self._args['max_tokens'] + 2

    @cached_property
    def ACCEPT_TEXT_EMB_INPUT(self):
        return True if self._args.get('text_embeddings', None) is not None else False

    @property
    def TRANSFORMER_TIME_LOGIT_ID(self):
        return 1 if self.ACCEPT_TEXT_EMB_INPUT else 0

    @cached_property
    def USE_HARD_NUM_TOKEN_EMB_FOR_ROOT_PREDICTION(self):
        return self._args.get('use_hard_num_token_emb_for_root_prediction', False)
