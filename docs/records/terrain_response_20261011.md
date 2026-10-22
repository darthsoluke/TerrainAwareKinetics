# Terrain Response Review and Online Corridor Experiments

The original A100-B service on port 28930 remains running. Port 28931 is the
interactive inference experiment. Its checkpoint SHA256 remains
0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d.
The authored world is test geometry; it is not new motion training data.

The later [flat-ground range repair](scan_range_flat_failure_20261011.md) supersedes
the expanded-observation behavior below: scans beyond the trained envelope now
remain explicit far-survey previews and are not fed into this old checkpoint.
This corrects severe flat-ground motion changes caused by raw out-of-range XYZ.

## Measured Failure Chain

`out/a100_live_20261010/terrain_response_chain_v1.json` contains 24 synthetic
observation interventions and 72 held-out source/predicted-token comparisons.
No synthetic future motion served as a target in these probes.

- At speed 1.6 m/s, eight tokens look ahead 1.893 m. A wall at 2 m was unobserved
  and produced exactly the same response as flat terrain.
- Twelve tokens extend the same trained adaptive sampler to 2.747 m; sixteen
  reach its original 3.4 m cap. This increases observation coverage, not skill.
- A 1.2 m wall one meter ahead with sixteen tokens produced a final Root rise
  of 0.960 m, while the decoded body's final rise was -0.015 m. Its executed
  early prefix still rose only millimeters. Both planning timing and the
  Root/Pose/decoder interface matter.
- The deployed decoder has planar external Root conditions and no direct scene
  collision objective. Lower native token/reconstruction loss cannot certify
  obstacle anticipation or nonpenetration of the final skinned body.

## First Training Ablation

Two 500-update A100 pilots and a zero-update baseline completed their evaluations
on the same existing split-preserving 96-train/36-validation window package.
GPUs 1 and 2 were checked free immediately before use. No new PARC generation,
bulk public transfer, or live weight replacement occurred.

The pilots introduce dense predicted Root-Y through MB's existing decoder fusion
layers. Only newly added height columns learn; Root/Pose, encoder/codebook and
all original decoder weights remain fixed and are checked after training.
The geometry arm additionally uses official SOMA surface points, PARC's inverted
air-box distance construction for wall/floor collision gradients, and conservative
source-geometry support evidence. Contact targets retain source gaps instead of
forcing the character onto an invented floor. Unknown terrain has explicit masks.
Geometry coefficients are calibrated from training gradient norms, not validation
success. The actor receives current local observations and predicted Root height;
full scene geometry participates only in target-side training losses.

| Variant | Played-history FK MPJPE, m | Mean sole penetration, m | 30 s rest drift, m |
| --- | ---: | ---: | ---: |
| Zero-update baseline | 0.17891 | 0.01819 | 0.76676 |
| Height columns | 0.17794 | 0.02130 | 0.78660 |
| Height + geometry | 0.17848 | 0.01830 | 1.04996 |

These pilots are rejected as motion improvements. They do not establish that the
small new interface acquired useful height control, nor that geometry losses are
ineffective with a properly adapted Pose/decoder. Training time was 48.4/65.6 s
excluding loading/evaluation. Metrics are under `out/terrain_response_20261011/`.
The remote task directory of the same name contains immutable contracts, weights,
fixed-scene short motion recordings and 30 s rest/start-stop trajectories.

The next learning experiment should adapt Pose and decoder together with actual
generated-history exposure, keeping source world/clock and vocabulary binding.
It is deferred while completing the user's requested online corridor sampler.

## Online Controls and Route Sampling

The interactive endpoint supports 6-16 motion tokens, 1-8 denoising passes,
0.3-12 m/s movement intent, 0.8-20 m lookahead and 0.3-6 m corridor half-width.
Automatic range is `base + gain * speed * prediction_seconds`, bounded to
0.8-20 m. Gain spans 0-4; base spans 0-10 m. Near-support sampling stays fixed
at +/-0.4 m, 0.1 m spacing. Changes apply only at a fully played block boundary;
changing a setting invalidates the previous cached continuation, not played motion.

Refinement follows a causal proposal/query/regeneration loop:

1. Generate a proposal from four actually played history frames, planar intent
   and the current near/intent terrain patches.
2. Take only the proposal's decoded XZ curve. Sample current scene geometry along
   its arc length, with lateral offsets from the curve's local tangents.
3. Regenerate Root/Pose/decoded motion from the same observed history and command
   using the measured corridor patch. The played history does not advance here.
4. Optionally repeat once with the refined proposal. Publish only the final result.

There is no future body-height target, pose observation, IK, foot lock or output
projection. A long scan extends beyond the finite proposal along its last tangent;
that extension defines observation locations, not a fabricated motion trajectory.
Five rows retain near-curve detail and four cover the farther range. Total model
input remains two 9x9 XYZ/valid patches; bigger distance reduces distant spatial
resolution and does not provide overhang/multiple-surface representation.

`Path refinement = 0` gives input-direction sampling; 1 is the default route pass;
2 performs two route refinements. The orange corridor shows actual model input.
Blue is the Root proposal; purple is the decoded trajectory to be played. The
scan proposal is the preceding refinement's decoded curve, so it may differ from
the final purple curve. Both provenance and coordinates are recorded in debug data.

These expanded and curved observations change the input distribution. They must
be included in matched training before claiming robust terrain traversal. The
existing straight-grid checkpoint remains a useful interactive comparison.

## Verification

Before route integration: 65 CPU tests passed, two optional tests skipped; real
browser sliders changed all applied settings, accepted 12 m/s commands, and passed
desktop/mobile rendering checks. Dynamic compiled horizons/ranges preserved tokens
and differed from native FK by at most 15.38 micrometers in ten startup cases.

Route sampling adds a regression in which straight input scanning misses an inside
turn wall while the decoded-curve corridor sees it. Near-support samples remain
identical, proposal gradients are detached, and an unavailable proposal falls back
to the original input scan. The extended CPU suite passes 66 tests, two skipped.
Final compiled/native corridor parity passed ten cases, including a curved proposal
and 20 m / 12 m/s settings, with identical tokens and maximum FK difference
15.44 micrometers. `runtime_corridor_review/checks.json` records eight actual
slider changes, route/straight switching, valid proposal/81-point model-input
diagnostics, 12 m/s control and desktop/mobile render checks. One default-route
browser snapshot measured 56 ms inference; an extra refinement costs another
native model pass. This is not a sustained 30 FPS performance approval.
