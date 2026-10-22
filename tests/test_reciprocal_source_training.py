import torch

from test_joint_sdf_controller import make_pair
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable
from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction


def check_arm(arm):
    _, model, source, history, intent, scene = make_pair()
    configure_trainable(model, arm)
    with torch.no_grad():
        before = model(history, intent, [scene])['world'].clone()
        altered = source.clone()
        altered[:, 4:, :3] += 3.
        changed = model(history, intent, [scene], source=altered)['world']
    torch.testing.assert_close(before, changed, atol=0, rtol=0)
    result = model(history, intent, [scene], source=source, straight_through=True)
    torch.testing.assert_close(before, result['world'], atol=3e-5, rtol=1e-5)
    native = model.native_losses(result, source)
    auxiliary = native_codec_reconstruction(model, result)['loss']
    prefix = decoded_factual_history_loss(model.inference, result['local_prediction'], history,
                                         result['observation'], root_coefficient=1.)
    (native['native_total']+auxiliary+prefix).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.codec.decoder.parameters())
    for module in (model.root, model.pose):
        active = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert active == (arm == 'coordinated')
    for module in (model.codec.encoder, model.codec.quantizer, model.sdf_encoder, model.sdf_root, model.sdf_pose, model.sdf_decoder):
        assert all(p.grad is None for p in module.parameters())
    torch.testing.assert_close(before, result['world'], atol=3e-5, rtol=1e-5)


def test_decoder_control_preserves_frozen_predictors_and_future_isolation():
    check_arm('decoder')


def test_coordinated_supervision_reaches_all_three_components_not_vocabulary():
    check_arm('coordinated')


def test_native_source_optimizer_and_rng_resume_matches_next_cpu_step():
    from copy import deepcopy
    _, first, source, history, intent, scene = make_pair()
    configure_trainable(first, 'coordinated')
    opt = torch.optim.Adam([p for p in first.parameters() if p.requires_grad], lr=5e-5)
    def step(model, optimizer):
        result = model(history, intent, [scene], source=source, straight_through=True)
        loss = model.native_losses(result, source)['native_total'] + native_codec_reconstruction(model, result)['loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .5)
        optimizer.step()
    step(first, opt)
    snapshot = dict(model=deepcopy(first.state_dict()), optimizer=deepcopy(opt.state_dict()), rng=torch.get_rng_state())
    _, second, _, _, _, _ = make_pair()
    configure_trainable(second, 'coordinated')
    second.load_state_dict(snapshot['model'])
    resumed = torch.optim.Adam([p for p in second.parameters() if p.requires_grad], lr=5e-5)
    resumed.load_state_dict(snapshot['optimizer'])
    torch.set_rng_state(snapshot['rng'])
    step(first, opt)
    torch.set_rng_state(snapshot['rng'])
    step(second, resumed)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    for index, value in opt.state_dict()['state'].items():
        for key, tensor in value.items():
            torch.testing.assert_close(tensor, resumed.state_dict()['state'][index][key], atol=0, rtol=0)
