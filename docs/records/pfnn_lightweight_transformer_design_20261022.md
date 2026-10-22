# Lightweight transformer for terrain-aware frame generation

2026-10-22. Architecture discussion requested by the user. This note proposes a
comparison with the current GRU actor; it does not launch training or replace
the current P2 plan, model, perception code, or user-facing page.

## Recommendation

Use a small causal temporal transformer with spatial cross-attention and a
continuous next-frame output. Temporal autoregression advances one actual frame
at30Hz. All joints for that frame are predicted together. A short Root-path
proposal supplies geometric lookahead; it does not commit a motion playback queue.

Existing MotionBricks Root/Pose models already contain transformers. Their
four-frame discrete tokens and sampling/decoding schedule differ from this
framewise continuous design. The current P1 actor instead uses a256-wide GRU.

## Preserve perception and its spatial tokens

Keep VolumetricEgoSampler, CompositeSceneField and path_observation, including
the known/unknown masks and existing feature meanings:

- Three8x8x8 regions: body, requested direction and actual velocity;1536 points.
- Signed distance, normal, relative positions, heading and region identity.
- SharedTerrainEncoder already returns24 spatial tokens, each128-dimensional:
  eight spatial groups per region. The current P1 `_terrain()` averages them.
- Give the24 tokens directly to attention. Preserve token availability and
  spatial/region identity; do not expand1536 raw points into temporal attention.
- Predict eight Root XZ path stations at0.1--0.8s, then requery the512 corridor
  points along that prediction. Body/actual-velocity samples are reused only
  within the same frame, as in the current implementation. The fine observation
  still contains24 tokens and three-dimensional geometric information.

The encoder weights and statistics can initialize from the existing saved
checkpoint. New temporal/attention/action heads require training. Original
encoder feature definitions must remain unchanged when reusing those weights.

## Temporal and output structure

Initial prototype: width128, three temporal blocks, four heads, feed-forward
width384. Start with a16-frame motion window and benchmark8-frame context too.
This is a design choice to profile, not a measured model latency or size.

Each historical frame becomes one motion token, encoding executed Root motion,
local joint rotations, FK-derived velocities and predicted contact state. The
transformer sees only already emitted frames. The four factual startup frames
are valid; additional slots are masked until real history accumulates. No
repeated idle frames or predicted future frames are inserted as history.

The motion window and perception history have separate lengths: temporal
attention can read16 frames while the existing sampler still uses the last four
Root frames. Relative time embeddings identify age. Root-relative positions and
consistent heading transforms align motion, command and spatial token positions.
Changing the encoder's native observation conventions requires retraining that
interface rather than silently reinterpreting the pretrained features.

1. Encode the executed motion window once.
2. A current-command query attends to history and coarse24-token geometry and
   predicts the eight-station Root path. Movement velocity and desired facing
   remain independent, matching the left/right sticks.
3. Requery the predicted path corridor using the existing algorithm.
4. Six parallel queries (Root, torso/head, left/right legs, left/right arms)
   attend to history, the current command, the Root proposal and fine geometry.
   Jointly emit the next Root displacement,30 joint-local6D rotations and four
   foot-contact logits. Define disjoint joint groups; output coordinates are
   converted through the same declared Root/world transform.
5. Official SOMA FK constructs positions/velocities and the emitted frame is
   appended to history. The next frame immediately uses the latest input.

Only the path proposal precedes the full motion output; limb queries are
parallel. Next Root motion and pose share the fine-conditioned decode so the
same geometry can influence body placement and joint configuration. Contact
outputs are learned kinematic labels, not a runtime foot-locking rule.

Begin with a full recomputation of this small temporal window each frame.
Current-Root recentering changes historical token coordinates and fresh geometry
changes conditioning; reusing stale KV tensors would need an explicit coordinate
and conditioning invariant. Reuse history activations within a frame instead.
Static shapes, SDPA and verified CUDA graphs are the first execution optimizations.

## Training

Retain the existing family split, full PFNN coverage and separate exposure ledger.
Use the corrected P2 displacement and boundary-velocity semantics in both GRU
and transformer comparisons. Architecture is the experimental variable.

- Supervise next Root displacement, next local rotations, FK pose, actual
  frame-to-frame velocity, source contact state and valid future path stations.
- Progress from factual history to generated history with4/8/16-frame rollout,
  retaining factual-history supervision. Match runtime history masks and window
  growth. Each rollout remains within its genuine paired source interval.
- For the longer motion window, use the available factual prefix and variable
  history masks. A33-frame cache row does not supply arbitrary16-frame prefixes
  plus16 future controls and a complete0.8s path. Mask unavailable labels and use
  longer intervals only where the original same-scene source actually provides them.
- Evaluate and train temporal/contact objectives separately: source-certified
  stance velocity and support height, preserved swing amplitude, full-body
  geometry, idle drift and braking. Learned contact logits cannot suppress masks.
- For changed commands or changed geometry, use explicit control/geometry task
  objectives; conflicting source future poses cease to be imitation targets.
  PFNN locomotion does not supply the full-body interaction examples needed for
  crawling/climbing. Those need admitted, paired motion and3D scene data.

The current P1 foot review shows teacher foot-span median1.003 but source-stance
sole speed8.75cm/s, rising to42.36cm/s with generated history. This makes velocity,
contact and rollout training central to the transformer design, alongside longer
visible motion history and spatial attention.

## Comparisons and deployment budget

Use three bounded comparisons after the common training objective is ready:

| Arm | Temporal backbone | Geometry consumption |
| --- | --- | --- |
| A | Current GRU | Current24-token mean |
| B | Same GRU | Spatial attention to24 tokens |
| C | Small causal transformer | Same spatial attention |

Keep data, input semantics, losses and exposure budgets matched; record parameter
counts, history lengths and actual compute separately. Arm B isolates the value
of spatial attention; B versus C evaluates temporal modeling. Avoid attributing
the joint effect of both changes to the transformer alone.

Target local complete-frame P99<=20ms, including fresh geometry queries, both
conditioning stages, motion prediction and FK; control30Hz, display>=30fps and
input-to-picture P95<=100ms. Current P1 measured2.81--3.00ms on PRO6000, while
the earlier local untrained frame-chain proof was7.02--10.91ms on RTX3080.
The proposed transformer needs its own local proof before training at scale.

Control, source stance, all18056 skin vertices/eight substeps/2cm and zero
unknown-uncertain remain independent acceptance gates. No runtime IK, output
filter, Root replacement or fabricated frames are part of this architecture.

## Basis

- PFNN's framewise control, persistent motion state and trajectory-conditioned
  terrain sensing: https://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf
- MotionBricks temporal-token/replanning analysis: https://arxiv.org/html/2604.24833v1
- DartControl's generated-history curriculum: https://arxiv.org/html/2410.05260v3
- PARC's geometry-sensitive learning and changed-scene target handling:
  https://arxiv.org/html/2505.04002v1

These references were reviewed in `pfnn_3d_literature_review_20261021.md`; this
note's dimensions, token organization and continuous output design are project
proposals. Code pointers: `SharedTerrainEncoder` in `unified_terrain.py`, current
`FrameActor._terrain()` in `pfnn_frame_actor.py`, and `path_observation()` in
`pfnn_3d_scene.py`.
