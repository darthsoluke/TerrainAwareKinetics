# Small4090D Decoder Terrain/Feedback Experiment

The user authorized a small-sample comparison on4090D, not another large
training run. The A100 vertical-frame experiment remains unchanged. PARC data
generation remains stopped.

## Inputs and Budget

- Host: private-user@remote-host.example:32297. Only currently available GPU0/1 are used.
  An unrelated task occupied GPU1 after initial inspection; the runner queues
  on free cards and never stops or shares that task's GPU.
- Remote output base:
  `${PRIVATE_STORAGE}/motionbricks-parc-training-20261009/terrain_decoder_pilot_v1`.
- Source: the completed A100 `pelvis_baseline/pipeline.pt`, SHA256
  `1bc2312a5ee24c15cb0e175db2deafc76188d52a0fd7254d5b30abce6ced84cc`.
- Original mixed dataset manifest SHA256:
  `c250f57dfd13ee7907a6ae0239dfbba26f8655b2dd64c5ec8f84ae937cb75de7`.
-24 training clips: eight BONES, eight official PARC and eight generated PARC,
  selected deterministically with category coverage. Four fixed65-frame windows
  per clip give96 training windows.
-12 validation clips: four BONES, five official PARC and three generated PARC;
  three windows each give36 held-out windows. Original source/scene groups and
  train/validation splits remain unchanged; the test split is not used.
- Window package SHA256:
  `3b081e6a32a57cac5e5630118d03b4b8402d03f661c299ae29b4df47a83e3a8f`.
- The764MiB source bundle, data and code snapshot moved through private Google
  Drive. All three hashes were verified on4090D. No bulk public-IP copy occurred.
- Each arm runs500 updates with four sampled windows and two primitives per
  update. Seed2333, Adam5e-5, gradient clip0.5, FP32/highest matmul precision.
  This is bounded decoder adaptation with a fresh optimizer, not a paper-budget
  MB reproduction. No artificial10k warmup is applied to a500-update experiment.

## The Four Arms

| Arm | Generated second-primitive history | Decoder terrain access |
| --- | --- | --- |
| baseline | no | no |
| feedback | scheduled | no |
| terrain | no | yes |
| both | scheduled | yes |

All four train the native decoder; Root, Pose, encoder and quantizer remain
frozen and are checked by tensor hashes. The terrain branch encodes the actual
current local observation and applies zero-gated cross-attention to the decoder
latent before its temporal convolutions. It does not move output joints, roots
or terrain. The initial zero gate reproduces the unmodified model output.

For feedback arms, updates0..100 use source history, updates100..400 linearly
increase the probability of generated history, and subsequent updates use it
for every second primitive. Only the last four actually committed FK frames
are re-encoded, detached and passed as history. This is a decoder-only adaptation
of DART's feedback-training principle, not a complete DART/MB Pose rollout trainer.

The same source windows and random draws are used across arms and their digest
is checked. Frozen Root/Pose requests for teacher contexts can be cached exactly;
decoder outputs are always recomputed with current weights. Generated histories
trigger a fresh native prediction and terrain observation at their actual position.

## Supervision and Evaluation

Reuse the native reconstruction, contact-related foot velocity and joint-velocity
objective. Commitment loss is zero because the encoder/codebook are frozen.
Observed frames are excluded from reconstruction supervision; the actual-history
to first-predicted-frame velocity boundary is included. Source future targets
retain their original world coordinates and clock under the SAME reference
transform as history. They are not moved to hide generated drift.

The initial source model is evaluated separately from equally budgeted baseline
finetuning. An offline source/model token and source/model Root-condition crossing
diagnoses decoder reconstruction capability; oracle futures never enter the
online comparison. Each arm then receives identical held-out teacher-history
and generated-history tests, plus30-second rest/start-stop tests from a genuine
held-out quiet BONES seed. These are distinct from the A10060-second benchmark.

Contact speed uses source-selected, fixed material vertices, with coverage and
invalid predicted terrain queries reported. Such source geometry masks are not
measured contact forces. A single seed and overlapping short windows do not
establish broad terrain traversal quality. No live weights are automatically
replaced, and no IK, foot locking, smoothing or output-height projection is used.

## Verification and Outputs

Eleven focused tests passed before deployment, including actual native request
replay, all supported vertical frames, zero-gate parity, gradient isolation and
fixed-world future targets. The later teacher-request cache check passed all
five decoder tests, including parity after changing decoder weights.

A real two-update4090D mechanics smoke run also completed its full evaluation
without changing frozen dependencies. It is separate from the500-update arms.
The production coordinator PID at launch was765973. It runs the four arms under
`runs/`, with a follow-on report/render/private-Drive exporter.

Final artifacts are `comparison/comparison.json`, `comparison/REPORT.md`,
`comparison/comparison.png` and planned source/baseline/both SOMA videos for two
preselected held-out examples. The report is written before optional rendering,
so a visualization failure cannot erase the numerical comparison.

## Results and Interpretation Audit

All four arms completed500 updates; each feedback arm actually used1010
generated-history second primitives and990 teacher-history ones. Window/draw
hashes matched. The small adaptation did not deliver an acceptable controller.
Training times were about87-164 seconds per arm, excluding evaluation and I/O.

The new terrain adapter's response is effectively at numerical noise level:
raising the decoder-only terrain input by0.1m changed predicted Root Y by roughly
0.00000007-0.00000010m on six diagnostic windows. Disabling its learned gate
changed output by roughly0.00000015-0.00000017m. Therefore this trial did NOT
establish an effective terrain-conditioned decoder, and cannot reject that
architectural idea. Root/Pose remain terrain-conditioned in all arms. Merely
adding the branch and passing gradient tests was insufficient evidence that the
branch learned a useful dependency.

The original-model oracle comparison over the first eight generated frames was:
model tokens/model Root60.34mm FK MPJPE; source tokens/model Root39.35mm;
source tokens/source Root30.22mm. This suggests token prediction matters, while
the codec still has reconstruction error. Crossed oracle conditions are not
runtime inputs and are not physically certified performance bounds.

The user asked whether metrics were correct and emphasized visualization. A
separate export reran the two30-second rest/start-stop sequences for the source
and all four arms. Root trajectories matched the saved evaluated trajectories
exactly. Official full-surface skinning reproduced the measured foot-region gaps
exactly. Recomputing FK MPJPE from saved short-sequence joints differed by at
most2.98e-8m. These checks establish arithmetic/render-geometry agreement for
those cases, not universal metric validity.

The earlier source-slow-vertex speed must NOT be described as measured contact
sliding: a prediction may legitimately change support phase, and the mask is
geometric evidence rather than measured force contact. MPJPE also penalizes
alternative plausible motion. Root height error against source motion alone
cannot prove floating. In static-flat tests, both feet's actual surface gaps
provide stronger evidence: the lower-foot median during rest was0.099m for the
source,0.793m for feedback-only, and0.944m for both. Both feet exceeded3cm clearance
for100% of the settled frames in the latter two cases. Conversely, baseline and
terrain-only decoders sank into the fixed plane. Zero penetration in the floating
cases is not success.

Visualization data `visual_data_v1` contains unchanged official SOMA77 rig
matrices for all five models, full30-second stationary/start-stop recordings,
and short flat/terrain examples with dataset references. Data moved to the local
PC through private Drive. The local viewer is `web/decoder_pilot/`, with a common
camera, frame stepping, playback speeds, root paths and optional transparent
ground. It displays recorded inference, not a new live controller or an approved
model. No joint, root or ground adjustment is applied for display.

Final local viewer checks passed at1500x960 and390x844. Browser skinned vertices
match official LBS samples within1.273e-7m, canvases contain visible moving
geometry, no page errors or horizontal overflow were observed. A local-browser
binary download interception returned HTTP204 for some `.bin` requests despite
the server/API returning the full file. JSON/base64 transport fixes that display
failure while preserving every float32 rig byte; frame lengths are checked.
Earlier failed viewer screenshots/check files are diagnostic artifacts, not
validation. The authoritative results are `viewer_checks_final/checks.json`.
The viewer runs at `http://127.0.0.1:28920/web/decoder_pilot/`.
