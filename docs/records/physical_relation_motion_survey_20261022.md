# Physical relations in real-time motion generation

2026-10-22. Research requested by the user: can motion generation directly model
body/object relations so generated kinematics satisfy physics? In this context,
"kinetics" is interpreted as the project's kinematic motion data and actor.
No model changes, training, remote GPU work, or deployment were performed.

## Finding

The closest direct precedents are PhysPT and InterPhys. They jointly represent
motion and physical quantities and train against dynamics, contact and force
consistency. InterDiff and CHOIS emphasize human/object contact relationships.
ContactNets shows a stronger structural approach: learn contact representations
and use a mechanics-based transition solve. PARC and InterMimic use physics to
produce/refine executable motion and supervise learned models.

These approaches have different output contracts. Soft physics losses improve
learned predictions; contact/dynamics solving enforces the chosen mathematical
model to numerical tolerance. A positive normal-force parameterization can
enforce that one inequality, but cannot alone enforce whole-body dynamics,
collision-free geometry and friction/contact-mode consistency.

## Data premise

A real human capture originates in a physical event. The training representation
contains reconstructed poses, estimated geometry and possibly retargeting errors.
Its physical feasibility is conditional on body proportions, inertia, environment
geometry, friction and timing. InterMimic explicitly discusses imperfect MoCap
contacts; PhysPT explicitly identifies missing physical properties/force labels
in existing motion datasets.

Our PFNN-to-SOMA data uses retargeting and author patch/RBF scene pairing. Source
admission checks geometry and source contact behavior, not measured joint torque
or contact force. The original geometry/contact receipts remain useful but do
not provide a dynamics certificate. Changing body proportions or a support
surface changes the relation that must hold for the same motion.

Even a fully admitted physical dataset requires a suitable model: regression
between different support modes can create an invalid intermediate state, and
autoregressive errors move subsequent inputs away from the recorded states.
Explicit contact modes and closed-loop training address these mechanisms.

## Primary works read

The abstract, method and relevant experiment/implementation sections below were
read from their primary arXiv HTML pages in this turn. Downloaded HTML/text,
URLs, retrieval timestamps and hashes are in
`out/physical_motion_survey_20261022/sources.json`.

### PhysPT, CVPR 2024

[Physics-aware Pretrained Transformer for Estimating Human Dynamics from Monocular Videos](https://arxiv.org/html/2404.04430v1)

- Sections3.2.1--3.2.4: Transformer encoder-decoder reconstructs/refines motion
  and predicts contact forces/internal actuation. Phys-SMPL supplies segment
  mass and inertia from body geometry.
- Section3.2.4 Eq.5: constrained quadratic optimization estimates force-model
  parameters from motion and assumed body properties. These are inferred labels,
  not measured force-plate ground truth.
- Training combines motion reconstruction, force, contact and Euler-Lagrange
  residual losses. This is a direct precedent for learning physical structure
  in a transformer without requiring a runtime physics engine for its estimates.
- Original contact model mainly assumes ground contact. Its encoder reads an
  input motion-estimate sequence; an autoregressive decoder alone does not make
  the full pipeline a zero-lookahead streaming joystick generator.
- Section4 implementation: six layers, eight heads, embedding1024, sequence16.
  Our proposed lightweight128-wide actor should borrow supervision/representation
  ideas rather than copy this network size or claim its latency.

### InterPhys, 2026 preprint

[Physics-aware Human Motion Synthesis in a Dynamic Scene](https://arxiv.org/html/2605.01036v1)

- Sections3.1--3.3: model human dynamics, dynamic-object dynamics and equal/opposite
  human/object reaction forces. Contact normals follow arbitrary3D surfaces;
  tangential friction is coupled to normal force.
- First-stage transformer diffusion predicts hand trajectories, joint torques
  and continuous contact-model coefficients. Second-stage diffusion generates
  human motion conditioned on those quantities and scene/object features.
- Eq.17 adds a soft dynamics-residual loss to motion reconstruction. Section4.2
  derives force parameters by optimization using body/object mass and inertia.
- The studied synthesis is conditioned on object motion. Our causal controller
  must predict or plan object evolution from available state, rather than read
  a future object trajectory that runtime does not possess.
- Evaluated on OMOMO and a dynamic-object subset of TRUMANS. The reported object
  collision threshold is4cm and hand contact threshold5cm; keep our2cm full-skin
  gate independently. Tables still report nonzero collisions/scene penetration.
- The inspected text does not establish a30Hz causal per-frame execution budget.
  Its useful contribution here is the force/contact/body/object coupling.

### InterDiff, ICCV 2023

[Generating 3D Human-Object Interactions with Physics-Informed Diffusion](https://arxiv.org/html/2308.16905v1)

- Sections3.1--3.2: jointly forecast human and object motion. A contact-based
  coordinate frame makes relative object motion simpler to predict.
- A learned STGNN interaction predictor corrects intermediate denoised motion;
  correction is scheduled from contact/penetration evidence inside diffusion.
- This is an interaction-representation and geometric-consistency approach,
  not a full force/torque dynamics solve. It demonstrates why relative contact
  coordinates are more informative than independent world-coordinate trajectories.
- Original BEHAVE setup uses10 past frames and25 predicted frames; later output
  can be generated autoregressively. Its schedule is not our single-frame actor.

### CHOIS, ECCV 2024

[Controllable Human-Object Interaction Synthesis](https://arxiv.org/html/2312.03913v2)

- Jointly generates human and object motion from language, initial states and
  sparse object waypoints. Object geometry loss improves trajectory alignment.
- Hand/object and foot/floor contact guidance is applied during the last10
  denoising steps. The method explicitly models interaction geometry, while its
  inference guidance adds work beyond a single feed-forward next-frame model.

### ContactNets, CoRL 2020

[Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations](https://arxiv.org/html/2009.11193v2)

- Sections2--4: learn inter-body distance/contact-frame Jacobians, then use the
  structured contact dynamics formulation for transitions.
- Eq.4 models nonpenetration, nonnegative normal impulses and complementarity;
  Eqs.5--7 encode friction/maximal dissipation.
- Figure2 explicitly uses a contact simulation/optimization formulation at
  prediction time. It is not an unconstrained neural pose decoder.
- Its demonstrations include simpler rigid-body systems, rather than a complete
  human whole-body animation system. Experiments still show numerical/geometric
  penetration errors, so "structured mechanics" must specify its model/tolerance.
- For our known scenes, reuse the existing geometry and analytical FK Jacobians;
  learning an approximate replacement SDF is not required by this idea.

### PhysDiff, ICCV 2023

[Physics-Guided Human Motion Diffusion Model](https://arxiv.org/html/2212.02500v3)

- Physics-based motion imitation projects intermediate denoised sequences to
  physically plausible motion; those results feed subsequent diffusion steps.
- It provides a useful physics-guided teacher/reference. The reported1000-step
  setting takes51.6s for one motion versus19.6s for MDM; this setup is not a30Hz
  next-frame implementation. No latency from it is assigned to our actor.

### PARC, SIGGRAPH 2025; InterMimic, CVPR 2025

[PARC](https://arxiv.org/html/2505.04002v1) connects terrain-conditioned motion
generation, physics-based tracking/refinement and data improvement. Its contact
and geometry losses and train/execute distribution matching are relevant to
an offline teacher for a fast kinematic actor.

[InterMimic](https://arxiv.org/html/2502.20390v2) trains subject-specific physics
teachers on imperfect human/object MoCap, distills them into a shared controller
and fine-tunes through RL. Section4.1 also trains kinematic HOI generators on
teacher-refined data and evaluates them with the physical controller. The
physics-executed result and the kinematic generator output are distinct artifacts.

### Further matches

- [SceneDiffuser, CVPR2023](https://arxiv.org/html/2301.06015v1): integrates3D
  scene-conditioned generation and geometry/goal optimization during diffusion.
  Useful for scene-conditioned constraints; the objectives vary by task and do
  not automatically constitute full human dynamics.
- [CLoSD, ICLR2025](https://arxiv.org/html/2410.03441v1): real-time diffusion
  planner plus physics controller in a feedback loop. Shows that physics-based
  execution can be real time, with a different system boundary from pure kinematics.
- [SMR, submitted2026-10-21](https://arxiv.org/html/2610.29169v1): causal transformer
  retargeting learned from semantic-conditioned physics experts for G1. Relevant
  to distilling physics-derived trajectories into a streaming network. It is a
  new preprint, uses a downstream controller and reports an embodiment/controller
  transfer limitation; it is not an established all-terrain human generator.

## Recommended modeling for this project

Use the next-step state of an interacting system as the modeling unit:

`body history + object/scene state + joystick -> contacts + motion + force variables`.

Retain our1536-point3D observation,24 spatial tokens and predicted-Root corridor
resampling. Add stable surface/object identity and object-local contact anchors.
An egocentric token index is not a persistent material point: the sampling grid
moves with the body, whereas a supporting foot relates to a fixed surface point
or to a point moving with its object.

For each candidate body/surface interaction, encode body region, object/surface
ID, surface-local anchor, normal, relative velocity, contact mode and time phase.
Separate approaching, sticking/supporting, sliding/rolling and releasing. Sliding
contacts cannot use a universal zero-velocity target. Start with feet and floor/
step surfaces, while allowing hands/knees/torso in the same representation.

The small causal transformer can predict the contact transition and next pose
jointly, with auxiliary contact-force/centroidal-dynamics heads during training.
Use a compatible human body model with explicit segment mass/inertia estimates
and scene friction parameters. Original source/derived labels retain provenance.
For moving objects, store pose, velocity and mass/inertia; geometry alone does
not determine whether an object can support or be moved by the character.

The central physical relationship is

`M(q) qdd + h(q, qd) = S^T tau + sum_i J_i(q)^T lambda_i`.

Here `S` restricts internal actuation to actuated joints; there is no arbitrary
external force/torque applied to the free Root. Human/object contact forces are
equal and opposite. Normal contact forces are nonnegative, friction depends on
normal force, and inactive/separated contacts cannot provide support. Actuation
limits and estimated physical parameters must prevent unconstrained force heads
from merely absorbing any motion residual.

For legged motion, a centroidal force/momentum formulation is a useful smaller
starting objective. A standing or walking body needs achievable support forces;
flight instead requires ballistic motion. Do not apply a static
"CoM projection inside support polygon" rule to all dynamic motions.

## Two implementation choices

**Fast kinematic deployment:** use dynamics/contact equations, inferred physical
labels and optional offline physics teachers during training. Deploy a single
causal next-frame network. This preserves the simple30Hz runtime and targets
lower violation rates through learning and structured representation. Preserve
our current long-rollout/contact/skin gates and closed-loop training.

**Constrained decoder:** predict contact choices and motion preferences, then
compute the next state with an embedded kinematic/dynamic constraint solve or a
parameterization that satisfies a specified subset by construction. This puts
enforcement inside generation. It changes the current pure neural-forward
execution design; any guarantee refers to the chosen collision/body/material
model, feasible initial conditions and numerical tolerance. It needs its own
local latency measurement and should be compared against the fast version.

For the immediate project, prioritize PhysPT-style physical training and
InterPhys-style body/object/scene relations in the lightweight causal model.
Use contact-relative representation from InterDiff, and retain a small structured
decoder prototype as a separate experiment when strict enforcement is required.
Do not claim that the training losses or transformer architecture enforce every
constraint at runtime. Evaluate which violations remain and their failure modes.

## Data additions and comparison

- Preserve the original motion, paired geometry and admitted source contact data.
- Add stable surface anchors, contact-mode evidence and physical parameter estimates.
- Infer force labels with a documented model, or generate them with an offline
  human-compatible physics teacher; keep estimates distinct from measurements.
- Reuse P2 generated-history training. On counterfactual commands/scenes, use
  task and geometry/physics objectives rather than inconsistent source future poses.
- Compare the same small transformer with pose-only supervision, contact-relative
  supervision, and contact plus physical-dynamics supervision under matched data
  exposure and inference budgets. Only then test an embedded solver variant.
- Test normal locomotion, braking, turning, rough transitions, support changes and
  later hand/knee interactions separately. Record full-skin penetration, relative
  surface slip, contact timing, force residuals, task success and latency.

This survey is an architectural recommendation. It does not modify the current
P2 budget, authorize a new training run, or reinstate the previous G1 runtime.
