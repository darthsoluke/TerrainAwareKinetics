from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from test_native_mb_stage_training import _motion_and_representation
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_codec_height import build_codec_height_tokenizer, rigid_vertical_pair
from scripts import train_parc_codec_height as codec_training
from scripts.train_native_mb_control import state_digest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _build_model_and_batch(frame_count=65):
    rep, batch = _motion_and_representation(frame_count=frame_count)
    config = small_config('tokenizer')
    network = config.model.pose_vqvae_network
    network.nb_code = 256
    network.code_dim = 32
    network.output_emb_width = 32
    network.width = 32
    network.depth = 1
    network.num_heads = 4
    network.dilation_growth_rate = 2
    source = build_native_model('tokenizer', rep, config).eval()
    model = build_codec_height_tokenizer(rep, source, config)
    model.train()
    model.pose_net.encoder.eval()
    model.pose_net.quantizer.eval()
    model._trainer = SimpleNamespace(global_step=50000)
    return rep, batch, model


def _fake_keyframes(motions, frame):
    known = torch.zeros(motions.shape[:2], dtype=torch.bool, device=motions.device)
    known[:, frame] = True
    return known, motions * known[:, :, None]


def test_native_random_default_keeps_old_keyframe_path_gradient_and_frozen_state(monkeypatch):
    np.random.seed(2333)
    torch.manual_seed(2333)
    _, batch, model = _build_model_and_batch()
    net = model.pose_net
    before = [state_digest(net.encoder), state_digest(net.quantizer)]
    captured = {}
    original_forward = net.forward_decoder
    original_loss = model.loss

    def sample_keyframes(motions, max_num_keyframes, probability):
        known, target = _fake_keyframes(motions, frame=8)
        captured['sampled_known'] = known.clone()
        captured['sampled_target'] = target.clone()
        return known, target

    def forward_decoder(indices, *args, **kwargs):
        captured['known'] = kwargs['has_target_cond'].detach().clone()
        captured['target'] = kwargs['target_cond'].detach().clone()
        return original_forward(indices, *args, **kwargs)

    def loss(loss_batch, prediction, root_output):
        captured['loss_batch'] = loss_batch
        return original_loss(loss_batch, prediction, root_output)

    monkeypatch.setattr(codec_training, 'sample_keyframes', sample_keyframes)
    monkeypatch.setattr(net, 'forward_decoder', forward_decoder)
    monkeypatch.setattr(model, 'loss', loss)

    # Omitting context_policy must retain the original random-keyframe path.
    losses = codec_training.decoder_step(model, batch)
    assert torch.equal(captured['known'], captured['sampled_known'])
    assert torch.equal(captured['target'], captured['sampled_target'])
    assert not captured['known'][:, :4].any()
    assert 'executed_history_positions' not in captured['loss_batch']
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    optimizer.step()
    assert before == [state_digest(net.encoder), state_digest(net.quantizer)]
    assert all(parameter.grad is None for parameter in net.encoder.parameters())


def test_played_context4_passes_true_prefix_and_detached_source_fk(monkeypatch):
    np.random.seed(3317)
    torch.manual_seed(3317)
    rep, batch, model = _build_model_and_batch()
    net = model.pose_net
    captured = {}
    original_forward = net.forward_decoder
    original_loss = model.loss

    def sample_keyframes(motions, max_num_keyframes, probability):
        return _fake_keyframes(motions, frame=9)

    def forward_decoder(indices, *args, **kwargs):
        captured['known'] = kwargs['has_target_cond'].detach().clone()
        captured['target'] = kwargs['target_cond'].detach().clone()
        return original_forward(indices, *args, **kwargs)

    def loss(loss_batch, prediction, root_output):
        local = loss_batch['local_motions']
        global_motion = loss_batch['global_motions']
        history = loss_batch['executed_history_positions']
        with torch.no_grad():
            source_fk = model.local_motion_rep.inverse(
                local, is_normalized=True, joint_positions_from='global_rot_data',
                return_all=True,
            )['posed_joints']
            global_target_ric = model.global_motion_rep.inverse(
                global_motion, is_normalized=True, joint_positions_from='ric_data',
                return_all=True,
            )['posed_joints']
            global_target_fk = model.global_motion_rep.inverse(
                global_motion, is_normalized=True, joint_positions_from='global_rot_data',
                return_all=True,
            )['posed_joints']
        assert history.shape == (local.shape[0], 4, rep.skeleton.nbjoints, 3)
        assert not history.requires_grad and history.grad_fn is None
        assert torch.equal(history, source_fk[:, :4].detach())
        # Source local FK, global target FK, and loss-target RIC share this
        # shifted world frame for the native representation/fixture.
        torch.testing.assert_close(source_fk, global_target_ric, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(source_fk, global_target_fk, rtol=1e-5, atol=2e-5)
        captured['loss_batch'] = loss_batch
        captured['source_fk'] = source_fk
        return original_loss(loss_batch, prediction, root_output)

    monkeypatch.setattr(codec_training, 'sample_keyframes', sample_keyframes)
    monkeypatch.setattr(net, 'forward_decoder', forward_decoder)
    monkeypatch.setattr(model, 'loss', loss)

    losses = codec_training.decoder_step(
        model, batch, context_policy='played_context4'
    )
    assert captured['known'][:, :4].all()
    assert captured['known'][:, 9].all()
    assert torch.equal(
        captured['target'][:, :4], captured['loss_batch']['local_motions'][:, :4]
    )
    assert torch.isfinite(losses['loss'])
    assert 'executed_history_positions' in captured['loss_batch']


def test_cli_context_policy_defaults_to_native_random():
    parser = codec_training.build_arg_parser()
    required = ['--bundle', 'bundle', '--dataset', 'dataset', '--output', 'output', '--gpu', '0']
    assert parser.parse_args(required).context_policy == 'native_random'
    assert parser.parse_args(required + ['--context-policy', 'played_context4']).context_policy == (
        'played_context4'
    )


@pytest.mark.parametrize('context_policy', ['native_random', 'played_context4'])
def test_context_policy_survives_tokenizer_config_artifact_roundtrip(context_policy):
    config = small_config('tokenizer')
    model = SimpleNamespace(args={'codec_context_policy': context_policy})
    artifact_config = codec_training.enable_config(
        model, {'tokenizer_config': config}
    )
    serialized = OmegaConf.to_container(artifact_config, resolve=False)
    restored = OmegaConf.create(serialized)
    contract_context = codec_training._context_policy_metadata(model)

    assert restored.model.args.codec_context_policy == contract_context['context_policy']
    assert contract_context['observed_context_frames'] == (
        4 if context_policy == 'played_context4' else 0
    )
    assert restored.model.args.codec_height_conditioning == 'native-dense-root-height-v1'
    assert restored.model.pose_vqvae_network.feature_mode[3] == 'root_without_heading'
    assert 'codec_context_policy' not in config.model.args


def test_executed_history_exposes_constant_height_boundary_and_finite_gradient():
    torch.manual_seed(517)
    rep, source_batch, model = _build_model_and_batch(frame_count=25)
    global_motion = source_batch['motion'][:, :24]
    local_motion = rep.dual_rep.global_to_local(
        global_motion, True, True, lengths=torch.tensor([24])
    )
    shifted_global, shifted_local = rigid_vertical_pair(
        global_motion, local_motion, rep, 0.35
    )
    history = model.local_motion_rep.inverse(
        shifted_local,
        is_normalized=True,
        joint_positions_from='global_rot_data',
        return_all=True,
    )['posed_joints'][:, :4].detach()
    prediction_base = shifted_local.detach()
    prediction_seed = shifted_local.detach().clone().requires_grad_(True)
    _, prediction_offset = rigid_vertical_pair(
        shifted_global, prediction_seed, rep, 0.25
    )

    def joint_velocity_loss(prediction, executed_history=None):
        loss_batch = {
            'local_motions': shifted_local,
            'global_motions': shifted_global,
        }
        if executed_history is not None:
            loss_batch['executed_history_positions'] = executed_history
        output = {
            'recon_state': prediction,
            'l_commit': prediction.new_zeros(()),
            'perplexity': prediction.new_zeros(()),
        }
        return model.loss(loss_batch, output, None)['l_joint_vel']

    native_base = joint_velocity_loss(prediction_base)
    native_offset = joint_velocity_loss(prediction_offset)
    torch.testing.assert_close(native_offset, native_base, rtol=1e-5, atol=1e-6)

    played_base = joint_velocity_loss(prediction_base, history)
    played_offset = joint_velocity_loss(prediction_offset, history)
    assert played_offset > played_base + 1e-5
    played_offset.backward()
    assert prediction_seed.grad is not None
    assert torch.isfinite(prediction_seed.grad).all()
    assert prediction_seed.grad.abs().sum() > 0
