"""Causal command-event phase for learned braking and restart, never motion edits."""

import torch
from torch import nn


class CommandTransitionState:
    def __init__(self):
        self.was_moving = None
        self.restarting = False
        self.resume_plans = 0

    def update(self, requested_velocity, observed_speed):
        moving = float(torch.as_tensor(requested_velocity).norm()) > 1e-6
        if not moving:
            self.restarting = False
            self.resume_plans = 0
        elif self.was_moving is False or (self.was_moving is None and observed_speed <= .15):
            self.restarting = True
            self.resume_plans = 0
        if moving and self.restarting:
            if self.resume_plans >= 2 and observed_speed >= .3:
                self.restarting = False
            self.resume_plans += 1
        self.was_moving = moving
        return (not moving) or self.restarting


class TransitionRootCommand(nn.Module):
    def __init__(self, previous):
        super().__init__()
        self.base = previous.base
        self.stop = previous.stop
        object.__setattr__(self, 'origin', previous)

    def forward(self, controls):
        gate = (controls[:,2:4].abs().amax(-1) <= 1e-6) if self.origin.active is None else self.origin.active
        return self.base(controls)+self.stop(controls)*gate[:,None]


class TransitionStageCondition(nn.Module):
    def __init__(self, previous, root):
        super().__init__()
        self.update = previous.update
        object.__setattr__(self, 'root', root)

    def forward(self, query, controls):
        features = controls[:,2:7].unsqueeze(1).expand(-1,query.shape[1],-1)
        gate = (controls[:,2:4].abs().amax(-1) <= 1e-6) if self.root.active is None else self.root.active
        return query+self.update(torch.cat((query,features),-1))*gate[:,None,None]


def enable_transition_gate(model):
    model.control.active = None
    model.control.forward = TransitionRootCommand(model.control).forward
    for stage, target, transpose in [('pose',model.pose._transformer_model,False),('decoder',model.codec.decoder,True)]:
        previous = getattr(model,'stop_'+stage+'_condition')
        # Existing hooks close over previous, so replace its forward behavior in place.
        transition = TransitionStageCondition(previous,model.control)
        previous.forward = transition.forward


def set_transition_active(model, active, batch=1):
    model.control.active = torch.full((batch,),bool(active),device=model.inference.device,dtype=torch.bool)
