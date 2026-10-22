"""Explicit sample-budget accounting; sample exposure is not optimizer equivalence."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TerrainSampleBudget:
    reference_batch: int
    reference_updates: int
    retained_examples: int
    global_batch: int

    def __post_init__(self):
        if min(self.reference_batch,self.reference_updates,self.global_batch)<1 or self.retained_examples<0:
            raise ValueError('Positive batch/budget and nonnegative retained example count required')
        if self.retained_examples>self.target_examples:
            raise ValueError('The retained model already exceeds this sample budget')

    @property
    def target_examples(self):return self.reference_batch*self.reference_updates

    @property
    def remaining_updates(self):
        return math.ceil((self.target_examples-self.retained_examples)/self.global_batch)

    def examples_after(self,updates):
        if not 0<=updates<=self.remaining_updates:raise ValueError('Update count is outside this run budget')
        return self.retained_examples+updates*self.global_batch

    def equivalent_updates_after(self,updates):
        return self.examples_after(updates)/self.reference_batch

    def learning_clock_after(self,updates,inherited_reference_updates):
        return self.equivalent_updates_after(updates)-inherited_reference_updates

    def learning_rate_after(self,updates,inherited_reference_updates,warmup):
        clock=self.learning_clock_after(updates,inherited_reference_updates)
        horizon=self.reference_updates-inherited_reference_updates
        if clock<warmup:return 5e-5*min(1.,max(0.,clock)/max(1,warmup))
        progress=min(1.,max(0.,(clock-warmup)/max(1,horizon-warmup)))
        return 2e-6+(5e-5-2e-6)*(.5+.5*math.cos(math.pi*progress))

    def report(self):
        end=self.examples_after(self.remaining_updates)
        return {'reference_batch':self.reference_batch,'reference_updates':self.reference_updates,
            'target_primitive_examples':self.target_examples,'retained_primitive_examples':self.retained_examples,
            'global_batch':self.global_batch,'remaining_actual_updates':self.remaining_updates,
            'final_primitive_examples':end,'rounding_extra_examples':end-self.target_examples,
            'final_reference_batch_equivalent_updates':end/self.reference_batch,
            'meaning':'Repeated primitive examples, not unique clips or an equal-convergence guarantee'}
