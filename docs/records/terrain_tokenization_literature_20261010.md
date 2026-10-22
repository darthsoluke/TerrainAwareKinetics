# Motion, Terrain and Execution: Literature Review

Read on 2026-10-10 in response to the question of jointly tokenizing the body
and terrain. This is a research review, not an implemented model change or a
new training launch. The current matched vertical-frame experiment is unchanged.

## Main Finding

Joint reasoning about motion and environment does not require a shared discrete
VQ codebook. PARC and TokenHSI encode different inputs separately, then fuse
their continuous embeddings inside a shared transformer. Their conditioning
reaches the network that produces motion or physical actions. Motion compression,
environment representation, and feedback training are separate design decisions.

The literature does not prove that a shared motion/terrain VQ codebook is bad,
or that adding terrain to our decoder alone solves drift. Our earlier proposed
decoder-only extension is a testable hypothesis, not an established diagnosis.

## Primary Sources and Actual Mechanisms

### PARC

Paper: [PARC, sections5-6 and Appendix D](https://arxiv.org/html/2505.04002v1).
Code inspected: local official checkout revision
`846ed98bd526c1dd93ddc1701b601036d553cd2f`, specifically
[`mdm_transformer.py`](https://github.com/mshoe/PARC/blob/846ed98bd526c1dd93ddc1701b601036d553cd2f/PARC/motion_generator/mdm_transformer.py).

- A character-local31x31 heightmap is encoded by a CNN into continuous tokens.
  Noisy motion frames and a planar target direction are embedded separately.
  These tokens, diffusion-time information, and optional two-frame history are
  processed jointly by a transformer to predict denoised motion.
- `forward()` concatenates the condition and motion tokens before the shared
  transformer; it discards condition-token output positions and maps motion
  outputs to continuous frame features. This is not a joint VQ motion/map codec.
- Section5.3 explicitly discusses overreliance on previous motion causing the
  generator to ignore terrain. The paper uses penetration supervision and a
  10% branch generating on random terrain with reconstruction loss disabled.
  It also blends history-conditioned and history-unconditioned denoising to
  trade temporal smoothness against terrain compliance.
- Its full method includes physics tracking and iterative generator/tracker/data
  improvement. High-quality simulated results must not be attributed to the
  kinematic generator or terrain tokens alone. The paper also describes batch
  selection and optional kinematic correction for collection; those are not
  requirements to copy into our local low-latency runtime.

Relevance: directly supports local terrain and motion feature fusion, and
testing whether history dominates the terrain condition. It does not prescribe
our two9x9 patches, command-dependent range, SOMA representation or MB losses.

### TokenHSI

Paper: [TokenHSI, CVPR2025, sections3.3-3.4 and4.2.3](https://arxiv.org/html/2503.19901v1).
Code inspected:
[`amp_network_builder_transformer_adapt.py`](https://github.com/liangpan99/TokenHSI/blob/57cdda998cd11e1c238e9cb5f5bb22e347f9a8e1/tokenhsi/learning/transformer/amp_network_builder_transformer_adapt.py),
especially the state/task encoders, token concatenation and shared transformer.

- Separate MLPs turn proprioception and each task observation into64-dimensional
  continuous tokens. A shared transformer fuses the active tokens using task
  masks; an action head produces control actions.
- Foundational skills include following, sitting, climbing and carrying.
  Terrain adaptation introduces a heightmap tokenizer while adapting the relevant
  task tokenizer and action-head adapters. The base proprioception tokenizer and
  transformer can remain frozen.
- In the implementation, the current body token, relevant task tokens and new
  observation token are concatenated and passed to the shared transformer.
  These "tokenizers" are feature encoders, not nearest-codeword VQ modules.
- The reported terrain experiment covers ascending/descending stairs, obstacles
  and flat terrain with physical policy training. Long task chains also use
  task switching; they are not evidence of unconstrained WASD parkour synthesis.
- Its ablation supports a separate shared proprioception encoder over duplicating
  body observations inside every task encoder for these tasks. It does not
  directly compare joint motion/terrain VQ against MB's factorization.

Relevance: evidence for separate modality encoders and joint downstream reasoning,
including an efficient frozen-backbone adaptation route. Its action-space,
simulation-based training differs from our kinematic motion reconstruction.

### DART / DartControl

Paper: [DartControl, ICLR2025, section3 and Appendices C-D/F-G](https://arxiv.org/html/2410.05260v3).
Code inspected at revision `06b8588429d4c85ea6418b1cceddee03cb4575b4`:
[`train_mvae.py`](https://github.com/zkf1997/DART/blob/06b8588429d4c85ea6418b1cceddee03cb4575b4/mld/train_mvae.py)
and [`train_mld.py`](https://github.com/zkf1997/DART/blob/06b8588429d4c85ea6418b1cceddee03cb4575b4/mld/train_mld.py).

- A history-conditioned continuous VAE represents short motion primitives.
  A latent diffusion model predicts the next primitive conditioned on history
  and text. The frozen decoder maps history and the latent to future motion.
  This is neither MB's discrete dictionary nor a motion/terrain joint VQ model.
- The paper and code progressively replace source history with generated
  history. `train_mvae.py` lines380-406 and `train_mld.py` lines413-423/521-525
  implement the rollout-history replacement and increasing probability.
  The previous generated primitive is detached; this is not automatically
  full differentiable backpropagation through an entire long rollout.
- The official training example explicitly enables full denoising rollouts and
  rollout history, while some CLI defaults use different settings. Copying
  defaults alone does not reproduce the published training recipe.
- DART's demonstrated 3D-scene interactions use SDF-based contact/collision
  objectives in latent-noise optimization. Appendix F reports a60-frame,
  100-update optimization example taking about74 seconds. Those scene demos
  must not be presented as the same real-time path as its online generation.
  Its demonstrated RL goal-reaching controller uses a simple flat scene.

Relevance: strong motivation to test generated-history exposure in both codec
and token generation. It does not establish a ready-made real-time controller
for our irregular terrain or prove that SDF must be our observation format.

### CLoSD

Paper: [CLoSD, sections4.1-4.3 and limitations](https://arxiv.org/html/2410.03441v1).
This review used the primary paper; no code-level parity claim is made here.

- DiP directly generates motion through autoregressive diffusion, conditioned on
  a clean motion prefix, text and target locations. Conditions enter transformer
  cross-attention. It does not first learn a shared terrain/body VQ dictionary.
- A PHC-derived controller tracks the plan in simulation. The actually simulated
  motion becomes the next planner prefix; the reference plan is not substituted
  for executed state. The tracker is fine-tuned in this closed loop while DiP
  remains frozen.
- The paper explicitly identifies adding exteroception such as vision or
  heightmaps as future work. CLoSD is therefore evidence for execution feedback,
  not evidence that its published planner already solves arbitrary terrain.

Relevance: separates desired motion from achieved motion and shows why simulated
results depend on an online execution system. Our FK playback feedback is not
equivalent to this force/contact-aware simulated feedback.

### MoConVQ

Paper: [MoConVQ, SIGGRAPH2024, sections3.1-3.4](https://arxiv.org/html/2310.10198v3).
[Official repository](https://github.com/heyuanYao-pku/MoConVQ).
The primary paper and repository overview were inspected; no detailed code
reproduction or terrain-controller parity claim is made here.

- It encodes motion with residual VQ: later quantizers represent residual error
  from earlier codewords. This differs from MB's parallel multihead quantization.
- Its decoder includes temporal upsampling, a policy conditioned on the current
  simulated state, and the simulator itself. Decoding a token therefore means
  controlling the character through successive state transitions.
- A learned world model approximates simulation for differentiable training,
  with real simulator rollouts used to train the world model. Its physical
  execution properties do not come from discrete symbols alone.

Relevance: when tokens are intended to represent executable skills, execution
and state feedback must participate in learning. This is a larger change than
adding terrain inputs to our existing kinematic decoder, not a drop-in VQ swap.

## Implications for This Project

The current MB extension already uses separate continuous local-terrain tokens
inside Root/Pose. What remains unverified is whether these conditions exert the
needed effect on the final decoded motion, and whether the model can recover
from the histories it generates. The completed first arm had Root-Y error
0.02925m with teacher history versus0.07732m with generated history in the fixed
short terrain review, and all four60-second rest/start-stop cases failed.
These results motivate feedback training; they do not isolate a unique cause.

Recommended experiments, not yet launched by this review:

1. Measure decoding with source action tokens versus predicted tokens under the
   same allowed conditions. Source future tokens are an offline diagnostic only.
   Determine whether errors originate in representation/decoding, token
   prediction, Root prediction, or their composition before replacing the codec.
2. Add generated-history exposure, following DART's principle with explicit
   coordinate reconstruction, correct future-target alignment and current-state
   terrain queries. Adapt this to MB's discrete objectives rather than claiming
   a literal reproduction of DART. Do not replace runtime observations with
   source future motion or unexecuted forecasts.
3. Independently test terrain-conditioned decoding. A frozen encoder/codebook
   can isolate whether direct scene access helps, but the existing codec's
   reconstruction regression means a frozen dictionary is not presumed optimal.
4. Compare unchanged baseline, feedback training only, decoder terrain only,
   and both, with matched source data, optimizer budgets and held-out rollouts.
   Start with bounded pilots before committing to another long training budget.
5. Use PARC's history-dominance finding to motivate terrain-sensitivity tests.
   Changing terrain under otherwise identical inputs should affect relevant
   future clearance and body motion. Such sensitivity alone is not correctness;
   evaluate valid paired scenes, contact evidence and complete traversals too.

Changing observation encoding does not supply collision geometry or physics for
free. Current heightfields cannot represent overhangs or multiple support surfaces
at one horizontal location. If geometry losses are introduced, they need the
verified SOMA surface/terrain coordinate contracts and valid observation masks.
Do not treat PARC's full-scene training collision checks as authorization to feed
unobserved future scene information into the online policy.

The immediate conclusion is to prioritize conditional joint reasoning and
generated-history consistency. A shared discrete motion/terrain dictionary
remains a research alternative whose extra capacity and generalization benefit
would need a controlled demonstration.
