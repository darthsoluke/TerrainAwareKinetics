# Drift Mechanisms and Evidence-Based Next Experiments

This turn performs code review and a read-only probe. It does not start training,
change the running A100 experiment, deploy a model, generate more PARC data,
or modify output roots/feet. The immediate question is why drift persists and
which small experiment can distinguish a real fix from another plausible idea.

## New Measured Evidence

Artifact: `out/terrain_decoder_pilot_20261010/drift_mechanisms_v1.json`.
The probe uses the same genuine held-out BONES rest seed as the local viewer,
and the same original A100 first-arm bundle and combined500-update pilot.
It executes113 replans, eight emitted frames each (30.13s), with a zero planar
velocity command. One fixed seed limits generalization of these findings.

| Measurement | Original model | Terrain + feedback pilot |
| --- | ---: | ---: |
| Mean Root condition speed over the executed prefix |0.04976m/s|0.10717m/s|
| Maximum commanded final-frame speed in decoder conditions |1.78e-15m/s|1.78e-15m/s|
| Final emitted net displacement X/Y/Z |[-1.433,0.190,-0.442]m|[-1.384,0.372,-3.027]m|
| Mean absolute redecoded last-context Y error |1.04mm|20.86mm|
| Mean absolute emitted-end Y difference from Root plan |9.87mm|50.44mm|

For every replan the probe checks the exact identity:

`emitted[11] - observed[3] = (decoded[3] - observed[3]) + (decoded[11] - decoded[3])`.

The first term is reconstructed-context error; the second is the predicted
in-plan motion. In the original model the signed context X/Z contribution sums
to[-0.0606,-0.0425]m, substantially less than total[-1.433,-0.442]m. Thus ordinary
horizontal drift in this case is not explained primarily by context rebuilding.
In the combined pilot the signed context Y contribution sums to+2.348m while
the in-plan Y contribution sums to-1.976m. These cancel to the actual+0.372m;
neither sum is a separate physical trajectory. This exposes repeated boundary
error in the failed decoder adaptation without pretending that one component
alone accounts for the observed path.

For body-height interventions, the fixed terrain stays unchanged and all body
world-Y channels shift together. Moving the observed body by+0.1m changes the
original model's next-block mean height by+0.09917m; the-0.1m intervention changes
it by-0.10003m. The combined pilot behaves similarly. This is weak one-step
height-restoring behavior near this seed, not proof of a global unstable
Jacobian, an exact gain greater than1, or an instruction to snap a body to ground.

## Four Different Mechanisms

### Control Objective and Receding Horizon

Our current `VELOCITY_PROTOCOL` constrains a terminal planar velocity, not a
near-term position or a complete desired velocity trajectory. Runtime plans
32 frames but consumes frames4..11, then plans again. A valid request to be
stopped at the end of the horizon does not by itself require that the executed
prefix be stationary. The measured condition satisfies the final-frame zero
velocity while its prefix still moves about5cm/s on average.

`apply_terminal_velocity` also supplies that explicit terminal condition after
global-to-local conversion. Consequently the final condition's near-zero value
is not proof that the model learned an adequate braking trajectory. This is a
control-task definition weakness, not evidence of a unit conversion fault.

MB section6.1 uses smart primitives: command-driven planar target positions,
timing and proxy pose constraints guide neural Root and body generation. Using
the official networks/losses while substantially weakening these conditions
does not reproduce the same runtime problem.

### Root and Decoder Can Disagree

The decoder receives planar Root velocity conditions and pose/history tokens,
and generates its own root/body features. Root-Y affects Pose, while the current
baseline decoder has no direct terrain or dense Root-Y input. Official MB
intentionally permits root refinement (section6.1); banning refinement or
copying Root output into the final skeleton is not the appropriate default fix.

Measured disagreement shows why a good Root output alone cannot certify good
body output. The tiny decoder-only trial trained the decoder while freezing
Root and Pose; this moved their learned interface without giving Pose a chance
to adapt. Its observed failure motivates matched downstream training, not a
claim that all decoder-only adaptation or generated-history training must fail.

### Short-Window Supervision Is Not Long-Horizon Recovery

Current A100 training samples source histories and short windows. Frozen Root
predictions as Pose conditions address one train/runtime mismatch, but do not
train the system on many consecutive generated FK histories. Pose's native loss
is token cross-entropy, not a direct long-horizon contact or drift objective.

DART's paper Appendix D.2 and official `train_mvae.py`/`train_mld.py` progressively
introduce generated histories to both representation and generation training.
Our500-step pilot adapted only the decoder, using two primitives and a short
curriculum. It is a deliberately limited diagnostic, not a completed test of
DART's full principle. Strongly deviated histories may also make unchanged
source futures difficult to recover within one short horizon; re-labeling those
targets by moving them with the prediction would hide error instead of solving it.

### Terrain Presence and Contact Loss Are Not Stable Support

The added decoder branch had nearly zero functional response to changed terrain
after500 updates. Opening the entire decoder while a zero-gated new branch is
still learning allows the established path to absorb the objective. Nonzero
gradients or a nonzero gate do not establish a useful terrain dependency.

The native contact-related velocity term is weighted by predicted contact
channels. It is not a measured-force constraint or a scene-aware support test.
An almost stationary foot above the floor can have low foot speed; low penetration
can also mean the whole body is floating. Neither quantity establishes contact.
Our older sole-region/physics diagnostics should not be described as active
native training losses unless the actual entry point uses them.

## Principles and Prioritized Strategy

1. Preserve the MB in-betweening architecture, tokenizer vocabulary and legal
   Root refinement while establishing a matched small control-interface test.
   Compare terminal-velocity-only conditions with explicit command-derived
   planar goal/timing constraints or a declared near-term intent condition.
   A target XZ derived from the command is an instruction, not a future measured
   body pose. Future height and terrain support remain predictions. Define
   braking/start/idle semantics once and use them identically in labels and runtime.
   Do not reassign a new distant stop deadline indefinitely. Compare zero-command
   rest and start/stop before terrain traversal; do not repair the output afterward.
2. Train continuity and generated-history recovery in the actual Root -> Pose ->
   decoder -> FK-history path. Preserve source world/clock, re-query only the
   current local observation, and explicitly handle the observed-to-predicted
   boundary. Introduce bounded consecutive generated segments as in DART, with
   an explicit curriculum and matched budget. For discrete tokens, begin with
   detached on-policy histories and native token targets; do not pretend argmax
   supplies differentiable end-to-end gradients. Any new decoded-motion objective
   requires an explicit gradient mechanism and a separate ablation.
3. Make the environment branch demonstrably functional before scaling it. A
   TokenHSI-inspired frozen-base/adapter-only warmup is a justified experiment;
   keep the motion/terrain encoders separate and fuse where motion is generated.
   Require deterministic nontrivial observation sensitivity plus better held-out
   paired motion, then train matching Pose/decoder dependencies if the decoder
   behavior changes. Sensitivity alone is not correctness. Do not infer that a
   shared VQ dictionary or an SDF will automatically stabilize a controller.
4. Restore contact-aware supervision only through the verified SOMA geometry
   contracts, known source support evidence and valid terrain queries. Distinguish
   support, swing, flight and unknown; do not demand both feet touch at all times.
   Compare actual foot/surface gaps and same-material-point motion with visual
   support timing. Predicted contact probabilities must not erase evidence from
   the scoring mask. Any extra objective is a declared MB extension, not an
   unchanged official loss.
5. Fix model selection for future runs: a regression against the initial
   tokenizer must remain a candidate rejection, since the current selector only
   compares positive-step checkpoints. Preserve matching tokenizer/Pose bindings;
   do not swap a better old tokenizer into an already trained new Pose blindly.
   Evaluate whole-pipeline fixed replay cases and visualized long rollouts,
   not only individually optimal native validation losses.

Perform these in bounded, isolated experiments on the existing corpus; do not
launch four large ablations or restart data generation. Finish the already
authorized A100 vertical-frame comparison to retain its evidence. A100/B's
coordinate change may improve height representation but cannot alone repair
terminal-command semantics, a drifting decoder or untrained feedback recovery.

## References and What They Do Not Establish

- [MotionBricks sections3,4.2,6.1](https://arxiv.org/html/2604.24833v1): partial
  keyframe constraints, conditional decoding and intentional root refinement.
  These do not promise stable zero-command infinite rollout after removing
  the original smart-primitive condition design.
- [DART section3 and Appendix D.2](https://arxiv.org/html/2410.05260v3): generated
  history training. This is a mechanism to adapt and test, not evidence that
  our decoder-only500-update pilot is sufficient.
- [PARC section5.3](https://arxiv.org/html/2505.04002v1): history dominance and
  terrain compliance. Its physics-tracked output relies on more than conditioning.
- [TokenHSI sections3.3-3.4](https://arxiv.org/html/2503.19901v1): separate
  observation tokens and efficient policy adaptation. Its physical actions and
  simulator differ from our kinematic decoder.
- [CLoSD section4.3](https://arxiv.org/html/2410.03441v1): actual executed-state
  feedback and tracker adaptation. Offline physically corrected data does not
  automatically confer runtime force/contact feedback on our model.
