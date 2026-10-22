"""Explicit sample-equivalent accounting, separate from Adam update counts."""

from dataclasses import dataclass
import math


POLICY = 'motionbricks.reference-exposure.v1'
REFERENCE_BATCH = 256
REFERENCE_STEPS = 50000
TARGET_EXPOSURES = REFERENCE_BATCH * REFERENCE_STEPS


class ReferenceExposureBatches:
    def __init__(self, schedule, dataset, global_batch, start_step, end_step=50000):
        if global_batch % 256:
            raise ValueError('A compute batch must group whole reference batches')
        self.schedule,self.dataset=schedule,dataset
        self.global_batch,self.start_step,self.end_step=global_batch,start_step,end_step

    def __iter__(self):
        from .exposure_codec_training import original_order_indices
        for step in range(self.start_step,self.end_step,self.global_batch//256):
            count=min(self.global_batch,(self.end_step-step)*256)
            yield original_order_indices(self.schedule,self.dataset,step,count)

    def __len__(self):
        return math.ceil((self.end_step-self.start_step)*256/self.global_batch)


@dataclass
class ExposureBudget:
    initial_updates: int
    initial_exposures: int
    global_batch: int
    actual_updates: int = None
    actual_exposures: int = None

    def __post_init__(self):
        if (any(type(value) is not int for value in (self.initial_updates,self.initial_exposures,self.global_batch))
                or self.initial_updates < 0 or self.initial_exposures < 0
                or self.initial_updates > self.initial_exposures // REFERENCE_BATCH
                or self.initial_exposures > TARGET_EXPOSURES
                or self.initial_exposures % REFERENCE_BATCH
                or self.global_batch < REFERENCE_BATCH
                or self.global_batch % REFERENCE_BATCH):
            raise ValueError('Invalid reference-exposure budget')
        if self.actual_updates is None:
            self.actual_updates = self.initial_updates
        if self.actual_exposures is None:
            self.actual_exposures = self.initial_exposures
        self.validate()

    @property
    def equivalent_steps(self):
        return self.actual_exposures // REFERENCE_BATCH

    @property
    def next_batch(self):
        return min(self.global_batch, TARGET_EXPOSURES - self.actual_exposures)

    def validate(self):
        if type(self.actual_updates) is not int or type(self.actual_exposures) is not int:
            raise ValueError('Progress counts must be integers')
        delta = self.actual_exposures - self.initial_exposures
        updates = self.actual_updates - self.initial_updates
        if (delta < 0 or self.actual_exposures > TARGET_EXPOSURES or delta % REFERENCE_BATCH
                or updates != math.ceil(delta / self.global_batch)
                or (self.actual_exposures != TARGET_EXPOSURES and delta % self.global_batch)):
            raise ValueError('Adam updates and sample exposures disagree')

    def advance(self, examples):
        if type(examples) is not int or examples != self.next_batch or examples == 0:
            raise ValueError('Advance exactly the next admitted exposure batch')
        self.actual_updates += 1
        self.actual_exposures += examples
        self.validate()

    def learning_rate(self, base=5e-5, minimum=5e-7):
        fraction = self.actual_exposures / TARGET_EXPOSURES
        return minimum + (base - minimum) * (1 + math.cos(math.pi * fraction)) / 2

    def contract(self):
        return dict(policy=POLICY, reference_batch=REFERENCE_BATCH,
                    reference_steps=REFERENCE_STEPS, target_exposures=TARGET_EXPOSURES,
                    initial_updates=self.initial_updates, initial_exposures=self.initial_exposures,
                    global_batch=self.global_batch, learning_rate_unit='sample_exposures',
                    trajectory_equivalent=False)


def validate_completion_counts(contract, actual_updates, actual_exposures):
    policy = contract.get('exposure_budget')
    if policy is None:
        if actual_updates != contract['steps'] or actual_exposures != actual_updates * contract['batch']:
            raise ValueError('Incomplete legacy optimizer budget')
        return
    if (policy.get('policy') != POLICY or policy.get('reference_batch') != REFERENCE_BATCH
            or policy.get('reference_steps') != REFERENCE_STEPS
            or policy.get('target_exposures') != TARGET_EXPOSURES
            or policy.get('global_batch') != contract['batch']
            or policy.get('learning_rate_unit') != 'sample_exposures'
            or policy.get('trajectory_equivalent') is not False
            or contract['steps'] != REFERENCE_STEPS or actual_exposures != TARGET_EXPOSURES):
        raise ValueError('Incomplete or incompatible reference-exposure budget')
    ExposureBudget(policy['initial_updates'], policy['initial_exposures'], contract['batch'],
                   actual_updates, actual_exposures)
