# Locomotion learning revision, 2026-10-05

## Question and frozen first experiment

V3 still has 56.54 mm/s measured support slip, despite correct source masks and
official skinning. Determine which representation/control mechanism limits it
before changing loss weights again. The first experiment is read-only on the
same 57 validation windows and the completed Region checkpoint.

1. Compare discrete decoding with continuous encoder-latent decoding. A large
   reduction in both pose error and slip supports studying the quantizer. A null
   result argues against adding codebook capacity now. Continuous inputs are an
   out-of-distribution probe, not a fair new model or a guaranteed upper bound.
2. Decompose measured material-point velocity error into a common horizontal
   translation and the orthogonal residual. This estimates what root translation
   could compensate, but cannot establish that root prediction caused the error.
   Record the unconstrained displacement required, including endpoint drift.
3. Measure disagreement between decoded XYZ features and rotation-based FK;
   compare error across the four token interval phases and verify the source
   velocity target agrees with source FK. Use these to distinguish representation
   inconsistency, temporal artifacts, and a coordinate error.

No data ground, labels, scoring masks, or validation membership changes in this
probe. Subsequent training must state which prediction it tests and keep a
same-budget control from the identical checkpoint.

## Literature and implementation lessons

### MotionBricks (2026)

Paper: https://arxiv.org/html/2604.24833v1 (sections 4.2, 6.1, 7.4).
Local implementation: `motionbricks/vqvae/neural_modules/vqvae.py` and
`motionbricks/vqvae/models/motion_vqvae.py`.

The root trajectory conditions each decoder upsampling level. The decoder can
refine root motion, and smart locomotion blends commanded and neural trajectories.
Our hard root-rate replacement is a project-specific deviation. It may restrict
pose/root coordination; simply removing it from already trained weights is not
a valid comparison because those output channels did not receive reconstruction
gradients. First measure, then train any root-refinement alternative with explicit
trajectory fidelity constraints. The paper's velocity loss already supervises FK
motion in our path; adding a duplicate velocity penalty needs separate justification.

### DartControl (ICLR 2025)

Paper: https://arxiv.org/html/2410.05260v3
Code: https://github.com/zkf1997/DART/blob/main/mld/train_mvae.py

`calc_loss` couples body-model joints to predicted joints and temporal deltas to
finite differences. `train` gradually replaces real history with generated history.
The applicable lessons are coherent redundant representations and training on
self-generated history. SOMA FK replaces SMPL here; copying SMPL-specific weights
or assuming its short primitive schedule fits MB would be unjustified. Rollout
training matters once the Pose/Root generation path exists, and cannot be validated
with whole-window source reconstruction.

### ARDY (SIGGRAPH 2026)

Paper: https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf
Official source reviewed:
https://github.com/nv-tlabs/ardy/blob/main/ardy/model/auto_latent_twostage_denoiser.py
https://github.com/nv-tlabs/ardy/blob/main/ardy/model/autoencoder/transformer.py

The denoiser predicts global root first, converts that prediction to local root
features, and conditions body generation on it inside each denoising step. The
body stage therefore learns with predicted root inputs, not only perfect source
trajectories. In the released implementation the conversion is detached during
training; this is staged conditioning, not unrestricted end-to-end body gradients
through root prediction. The tokenizer uses causal patch attention and local
root velocity conditioning. Its ablation reports worse skating without the
global-to-local conversion. Our MB decoder already receives local root rates, so
that lesson alone does not justify changing our representation again.

Transfer the train/runtime conditioning principle, not the entire diffusion
architecture, proprietary data scale or loss coefficient. The paper's contact
loss uses predicted contact labels; retain our source evidence masks for the
supervised losses to avoid reducing supervision by changing predicted contacts.
The paper explicitly lists residual skating and jitter as limitations. Its
reported latency is not an RTX 3080 end-to-end measurement. Released checkpoints
currently list Core and G1; a SOMA model is described as forthcoming, so this is
not a drop-in SOMA replacement.

### No foot-locking fallback

The user explicitly rejected the PFNN/Learned Motion Matching foot-IK route.
Do not add a foot anchor, IK foot lock, root compensation after decoding, or
test-time motion correction as the solution to this task. Evaluate the raw
generated root and local rotations through official SOMA FK/skinning. Existing
third-party correction modules are references only and remain unused.

The target is learned coordination: history and command produce coherent root,
body and contact evolution. A low skating score alone is insufficient: standing
still, slowing down, changing contact masks, or losing the requested trajectory
must not count as success.

## Next decision

Prioritize an evidence-supported change to representation consistency or root/pose
coordination. Preserve the MB control architecture. Evaluate pose, source-masked
slip, penetration, trajectory fidelity and temporal errors together. After the
tokenizer is usable, train Root/Pose with history feedback and test continuous
start/stop/turn/backward control. RTX 3080 latency requires an actual runtime
measurement; training throughput cannot establish it.

## First probe and the next frozen comparison

The initial read-only probe found that continuous latent decoding worsened pose
error and skating. Four token interval phases had similar velocity errors. The
source FK and velocity target agree to approximately 1.6e-5 m/s. Redundant XYZ
and FK differ by approximately 14 mm. This supports testing consistency first;
it does not establish causality or prove that root refinement is unnecessary.

The common-translation component covers approximately 76% of measured velocity
error energy, but correct source root motion is already supplied. Root changes
could compensate for pose error; this is not evidence that the root is wrong.
The first probe additionally recanonicalized yaw; a second probe removes that
redundant step to retain exactly the same source masks as the V3 evaluator.

Next experiment: warm both Control and Consistent from the V3 Region final
checkpoint; 152 training clips, identical seed and loaded parameters, 10,000
updates each. All region/source losses stay fixed. Consistent adds position
coupling 5/m and velocity coupling 0.5 s/m, motivated by the existing FK position
scale and a 0.1-second temporal scale. Architecture and inference cost are unchanged.
Predictions versus Control: representation disagreement <=70%, slip <=80%, and
MPJPE <=105%. All 57 validation windows remain fixed. A failed prediction is
reported as a negative result; it cannot be replaced by a favorable showcase.

Protocol files and hashes are saved before launching. Local Git protocol commit
was attempted but the object database rejected a temporary-file write with
`Invalid argument`; no unrelated changes were staged or reverted.

## Completed comparison and handoff

Both arms completed 10,000 updates; paired initialization and data hashes match.
The exact-window probe retains all 57 validation windows and 7,615 measured
intervals. Quantized decoding has 35.689 mm MPJPE and 56.540 mm/s slip; continuous
latent decoding worsens these to 43.734 mm and 65.059 mm/s. The exact common-mode
velocity error energy fraction is 77.23%, not a causal attribution to root.

Against the same-budget Control, Consistent reduces XYZ/FK disagreement from
13.117 to 6.167 mm and acceleration error from 4.065 to 3.572 m/s^2. However,
slip changes only from 50.242 to 49.487 mm/s (ratio 0.985, target <=0.8), MPJPE
worsens from 35.499 to 36.541 mm, and peak slip worsens from 0.989 to 1.227 m/s.
The main skating prediction failed. Do not infer a significant skating gain from
one seed or enlarge this loss in another blind sweep.

Control is frozen for the next bounded Root/Pose experiment, not approved for UE
quality. Both retain substantial raw motion artifacts. The next executor's plan
is `sol61_no_ik_locomotion_execution.md`; it explicitly rejects foot-locking and
requires learned continuous generation with generated-history feedback.
