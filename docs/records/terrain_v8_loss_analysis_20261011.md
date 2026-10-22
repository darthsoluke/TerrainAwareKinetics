# V8 Loss and Training-Dynamics Analysis

This is a read-only CPU audit requested by the user. No running training code,
weights, optimizer or evaluation schedule was changed. Recorded training rows,
the actual checkpoint and accepted source metadata were inspected. Selected
short source shards were opened to verify their real frame counts.

## Confirmed Feedback-Length Defect

At checkpoint4700, replaying the exact seeded sampler from the saved draw offset
recovered163 global batches.154 had a shortest source of65 frames,7 of66 frames,
1 of67 and1 of70. The trainer takes the minimum source length across every row
and every DDP rank before choosing one common rollout depth for the entire batch.

Resulting episode depths:112 batches of5 primitives,15 of2,10 of4,6 of6,15 of1,
5 of3. Mean4.270 primitives; zero batches reached the configured maximum13.
Emitting8 frames per primitive at30Hz gives0.27-1.60 seconds per episode, instead
of the intended3.47 seconds for13 primitives. The first primitive is initialized
from factual history, so the uninterrupted generated-feedback portion is shorter.

This defect already exists in the global-minimum batching logic; expanding the
batch makes its effect more consistent. About85.67% of source sampling mass has
at least129 frames, sufficient for13 primitives with the common32-frame prediction
horizon. The probability that every draw qualifies is about.00708 at batch32 and
2.51e-52 at batch768. These probabilities concern independently drawn source
lengths; the exact replay above is the evidence for the current run.

The next training revision should group compatible source lengths or independently
advance valid row episodes. Preserve source sampling probabilities, keep short
clips in their own eligible batches, never pad futures and never discard short
sources solely to inflate rollout statistics. Verify realized per-row history
lengths and source-time alignment before claiming13-primitive training.

## Loss Comparability

The log records one rank every25 updates. It is not an all-step/all-rank mean.
Prediction horizons vary24-64 frames and samples occupy different generated-history
positions. Larger windows of logged rows give total-loss means1.756,1.927,1.759,
1.726; this is not persistent divergence. The last block remains incomplete.

Matching horizon and episode position leaves only1-2 logged batches per period.
Those matched total losses decreased slightly, but generated-history FK error
did not consistently improve. The evidence is too sparse and uses different
source examples to establish convergence or motion-quality improvement.

Token cross-entropy contributes about85% of the scalar total. Physical quantities
have different units and gradient paths. Scalar shares do not establish gradient
dominance and do not justify increasing physical weights without measurements.

## Sample Budget Is Not Optimizer Equivalence

Batch768 is24 times the reference batch32. Equal primitive-example exposure needs
about24 times fewer Adam updates. LR decay and feedback scheduling now use sample
exposure, but parameter updates and Adam moments still evolve per optimizer step.
Adam betas remain.9/.999 and the LR peak stays5e-5. The remaining transition from
feedback probability.75 to1 now spans42 updates instead of1000. This is a plausible
optimization limitation, not a demonstrated cause of the measured FK trend.

Blindly multiplying LR by24 would not be justified for this Adam, straight-through
token and generated-history system. Test schedule/batch changes on fixed train
cases under equal sample budgets before drawing a causal conclusion. No such
new model experiment was run during this analysis.

## What Was Ruled Out

Sample exposure remains approximately50/50 BONES and PARC. No unknown-history
resets were recorded in the inspected logs. Recorded loss and gradients are
finite; clipping is active, which alone is not proof of instability. The batched
geometry change already passed value/gradient equivalence on actual CUDA inputs
(3.7253e-9 maximum absolute difference).

The current authorized run continues to its bounded sample budget and its single
final review. It must not be described as having delivered13-primitive feedback
training. Implementation diagnosis does not replace the final source/model visual
comparison or free-WASD evaluation.

Evidence: out/model_terrain_v3_20261011/sample_budget_dynamics_audit.json.
Reproduction: scripts/audit_terrain_sample_budget_dynamics.py, CPU only.
