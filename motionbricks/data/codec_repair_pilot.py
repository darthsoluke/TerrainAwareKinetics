"""Explicitly bounded decoder-only repair experiment, separate from full training."""

import hashlib
import json
import math

import torch
import torch.nn.functional as F


SCHEMA = 'motionbricks.codec_repair_pilot.v1'
PARENT_SHA = 'b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19'
PREFIXES = ('codec.decoder.', 'adaptation.decoder.', 'adaptation.height_encoder.')


def validate_plan(plan):
    expected = dict(schema=SCHEMA, status='proposal_not_authorized', parent_sha256=PARENT_SHA,
        arms=dict(native=0., native_fk=.02), updates_per_arm=250, global_batch=1024,
        micro_batch=512, reference_batch=256, first_reference_step=50000,
        exposures_per_arm=256000, total_exposures=512000,
        mix=dict(pfnn=.5, parc=.25, bones=.25), seed=2333,
        learning_rate=5e-5, minimum_learning_rate=5e-7, gradient_clip=.5,
        cuda_memory_fraction=.9, fk_scale_m=.02, played_frame_start=4, played_frame_end=28,
        checkpoint_every_updates=50, trainable_prefixes=list(PREFIXES),
        optimizer_policy='fresh_adam_per_arm_no_resume', runtime_changed=False,
        quality_approved=False, delivery_approved=False)
    if plan != expected:
        raise ValueError('Pilot plan differs from the reviewed bounded proposal')
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def accounting(plan, updates):
    if type(updates) is not int or not 0 <= updates <= plan['updates_per_arm']:
        raise ValueError('Pilot update budget exceeded')
    exposures = updates*plan['global_batch']
    return dict(actual_updates=updates, actual_exposures=exposures,
        equivalent_reference_updates=exposures//plan['reference_batch'],
        next_reference_step=plan['first_reference_step']+exposures//plan['reference_batch'])


def learning_rate(plan, updates):
    progress = accounting(plan, updates)['actual_exposures']/plan['exposures_per_arm']
    low, high = plan['minimum_learning_rate'], plan['learning_rate']
    return low+(high-low)*(1+math.cos(math.pi*progress))/2


def configure_scope(stage):
    controller = stage.controller
    controller.eval().requires_grad_(False)
    active = [controller.codec.decoder, controller.adaptation['decoder'], controller.adaptation['height_encoder']]
    for module in active:
        module.requires_grad_(True)
    object.__setattr__(stage, '_active_modules', active)
    stage.train()
    names = [name for name, parameter in controller.named_parameters() if parameter.requires_grad]
    if not names or any(not name.startswith(PREFIXES) for name in names):
        raise ValueError('Unexpected trainable pilot parameter')
    if any(not any(name.startswith(prefix) for name in names) for prefix in PREFIXES):
        raise ValueError('Missing required decoder or terrain-adapter parameters')
    assert_frozen_scope(stage)
    return names


def assert_frozen_scope(stage):
    controller = stage.controller
    if controller.codec.training or controller.codec.encoder.training or controller.codec.quantizer.training:
        raise ValueError('Frozen encoder/quantizer must remain in eval mode; EMA updates are forbidden')
    for name, parameter in controller.named_parameters():
        expected = name.startswith(PREFIXES)
        if parameter.requires_grad != expected or (not expected and parameter.grad is not None):
            raise ValueError('Pilot touched a protected parameter: '+name)


def fk_reconstruction_loss(controller, reconstruction, target, plan):
    rep = controller.inference.local_motion_rep
    names = rep.skeleton.bone_order_names
    ids = sorted(set([names.index(side+joint) for side in ('Left', 'Right')
                      for joint in ('Leg', 'Shin', 'Foot')]+[int(i) for i in rep.skeleton.foot_joint_idx]))
    pred = rep.inverse(reconstruction, True, return_all=False,
        joint_positions_from='global_rot_data')['posed_joints']
    with torch.no_grad():
        truth = rep.inverse(target, True, return_all=False,
            joint_positions_from='global_rot_data')['posed_joints']
    pred, truth = pred-pred[:, :, :1], truth-truth[:, :, :1]
    first, last = plan['played_frame_start'], plan['played_frame_end']
    return F.smooth_l1_loss(pred[:, first:last, ids]/plan['fk_scale_m'],
                          truth[:, first:last, ids]/plan['fk_scale_m'])


def window_metrics(controller, local, source, observation):
    from .native_mb_vertical_frame import shift_global_motion
    from .terrain_decoder_pilot import vertical_origin
    inference = controller.inference
    rep = inference.global_motion_rep
    world = inference.motion_rep.dual_rep.local_to_global(local, True, False,
        lengths=torch.full((len(local),), 32, device=local.device, dtype=torch.long))
    offsets, heading, _ = inference._extract_initial_root_info(source[:, :4, :5])
    world = inference._reapply_initial_root_info(dict(pred_global_poses=world,
        reference_start_root_global_offsets=offsets, reference_start_root_global_heading=heading))
    world = shift_global_motion(world, rep, vertical_origin(inference, source[:, :4], observation))
    pred = rep.inverse(world, False, return_all=False,
        joint_positions_from='global_rot_data')['posed_joints']-world[:, :, None, :3]
    truth = rep.inverse(source[:, :32], False, return_all=False,
        joint_positions_from='global_rot_data')['posed_joints']-source[:, :32, None, :3]
    names = rep.skeleton.bone_order_names
    ids = [names.index(side+joint) for side in ('Left', 'Right') for joint in ('Leg', 'Shin', 'Foot')]
    leg = (pred[:, 4:28, ids]-truth[:, 4:28, ids]).norm(dim=-1).mean((1, 2))
    root = (world[:, 4:28, :3]-source[:, 4:28, :3]).square().mean((1, 2)).sqrt()
    return dict(leg_mpjpe_mean_m=float(leg.mean()), leg_mpjpe_p95_m=float(torch.quantile(leg, .95)),
        root_xyz_rms_mean_m=float(root.mean()), root_xyz_rms_p95_m=float(torch.quantile(root, .95)),
        per_window_leg_mpjpe_m=leg.cpu().tolist(), per_window_root_xyz_rms_m=root.cpu().tolist(),
        partial_window_gate_pass_count=int(((leg <= .015) & (root <= .02)).sum()),
        geometry_evaluated=False, quality_approved=False)


class PilotObjective:
    def __init__(self, controller, plan, arm):
        if arm not in plan['arms']:
            raise ValueError('Unknown experiment arm')
        self.controller, self.plan, self.arm = controller, plan, arm
        self.native = controller.loss_model.loss
        self.last = None

    def __call__(self, target, output, root):
        native = self.native(target, output, root)
        weight = self.plan['arms'][self.arm]
        auxiliary = (fk_reconstruction_loss(self.controller, output['recon_state'],
            target['local_motions'], self.plan) if weight else native['loss'].new_zeros(()))
        total = native['loss']+weight*auxiliary
        self.last = dict(native_loss=native['loss'].detach(), fk_loss=auxiliary.detach(),
            reconstruction=output['recon_state'].detach())
        return dict(native, loss=total)
