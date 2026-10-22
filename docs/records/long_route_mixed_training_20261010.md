# Long-Route Mixed Continuation

## Current Run

The user has stopped further PARC generation: the existing 5031.8667 seconds
(85 clips, about 1 h 24 min) of new long training motion are sufficient for this
round. No collection coordinator or generation worker remained at launch.

A100 base: `${PRIVATE_STORAGE}/motionbricks-20261010`.
Active supervisor: `long_route_mixed_10k_ab_v2`, recovery PID 2666648.
Code snapshot: `code_long_route_train_v3`; each arm also freezes its own code.
The first arm retains its original v2 training code and trained artifacts.
Physical GPU1 and GPU2 were free at dispatch and are the requested devices.
They are connected through PXB, not NVLink; the NVLink partners were occupied
at launch. GPU0 and GPU7 remain reserved for other users.

The first launch used an overlong nested TMPDIR, which prevented Python's
AF_UNIX data-loader socket from starting. Only this launch's descendant
processes were terminated; there was no optimizer progress. The v2 launch
inherits the short task TMPDIR, passed a real multiprocessing IPC smoke test,
and reuses the already completed dataset without rebuilding it.

Live startup verification reached Tokenizer step500 (loss0.025911) and Root
step200 (loss0.048570); these are different random batches, not a loss trend or
evidence of motion improvement. Native finite-loss/gradient checks are active.

## Immutable Inputs

PARC: `parc_long_route_complete_v1/parc_native`, manifest SHA256
`f0ed1d707eb4be576afe063e779b1b13865810c92f58ff068d6691f1449c708e`.
Its native artifact hashes, source whitelist and source/terrain split ownership
were verified before mixing. Data admission is permission to use a geometric
teacher, not proof of physical or skill quality.

Mixed data: `long_route_mixed_10k_ab_v1/mixed_dataset`, manifest SHA256
`c250f57dfd13ee7907a6ae0239dfbba26f8655b2dd64c5ec8f84ae937cb75de7`.
The existing BONES fixed-plane source screen passed all selected records.
Original splits and PARC training-only normalization are retained.

| Source | Train clips | Train seconds | Validation clips | Test clips |
| --- | ---: | ---: | ---: | ---: |
| BONES-SEED | 2003 | 16872.4000 | 19 | 20 |
| Official PARC | 437 | 1809.4667 | 48 | 49 |
| Generated PARC | 89 | 5096.4000 | 3 | 0 |

There are 2529 training clips and 23778.2667 seconds (about6.61h), counted as
`(frames_30hz - 1) / 30`, without overlap/window multiplication. Generated
training contains the85 new long clips plus four previous admitted clips.

Warmstart: `mixed_velocity_50k_v1/pipeline.pt`, SHA256
`de0aa8c351dea859339422acf151af83cbada3b86ab47babf7022243502c64a8`.
This bundle failed prior closed-loop tests; it is an initialization for the
matched experiment, not an approved live controller.

## Training and Automatic Evaluation

Both arms initialize from the exact same bundle: `pelvis_baseline` uses
`observed-pelvis-y-v1`, then `terrain_reference` uses `observed-terrain-y-v1`.
Each trains Tokenizer and Root independently on one GPU each for10000 updates,
then its matching Pose on both GPUs for10000 updates. This is a weights-only
warmstart with fresh optimizers, not an uninterrupted optimizer resume.

Keep native MB classes and loss coefficients; Adam5e-5, warmup10000,
cosine horizon2000000, clip0.5, seed2333. Tokenizer/Root areFP32 with input
batch256. Pose is BF16 with FP32 frozen teachers, input batch1024/GPU,
global2048. This is the measured throughput profile, not the paper's full
distributed training budget. A10k adaptation covers only the warmup portion;
it is a bounded comparison, not convergence training.

Sampling remains50% BONES, balanced across control categories, and50% PARC.
The new explicit `parc-windows-v1` option weights PARC clips by their eligible
65-frame starts instead of giving long and short clips equal weight. Actual
probabilities are50% BONES,7.686% official PARC and42.314% generated PARC.
Both arms share this change. Existing callers retain `clips-v1` by default.

Validate every2000 updates, retain periodic/final checkpoints, and select the
validation candidate using the existing fixed held-out benchmark. Pose binds
its own frozen selected Tokenizer/Root; vocabularies are not interchanged.
After export, run four60-second flat rest/start-stop cases plus24 fixed
teacher-history/played-history windows. The latter actually include17 official
PARC, three new long generated PARC and four BONES windows. Preserve the
source-relative benchmark; it does not imply complete parkour-course success.

No future height/pose is an observed runtime input, and no IK, foot locking,
height projection or output smoothing is introduced. No automatic live model
replacement is configured. Models/reports export through private Google Drive;
interactive inference remains local to the user's PC.

Verification:18 focused sampling, split-isolation and vertical-frame tests
passed, the real short-TMPDIR IPC test passed, and actual training updates were
observed. Final quality and the terrain-reference arm remain pending.

## First Arm Completed and Review Recovery

Tokenizer, Root and Pose each completed10000 optimizer updates. The held-out
selector chose step2000 for all three; it selects among trained checkpoints,
not the pre-adaptation initialization. Tokenizer's selected validation loss is
0.033540 versus0.032155 at initialization, Root0.076979 versus0.084572, and
Pose1.397066 versus1.724516. Lower Root/Pose loss did not produce stable rest.

All four60-second flat rest/start-stop cases failed the existing quality gate:
settled cumulative displacement was0.647-1.832m. Sole penetration P99 was
0-0.00879m, while the median higher-foot gap was0.0535-0.1482m. These metrics
measure different failure modes; low penetration is not stable contact.
No local controller was replaced.

The subsequent terrain review crashed because it accessed `dual_rep` on the
global sub-representation, where it is None. The corrected evaluation calls
the inference object's real dual representation, with explicit33-frame batch
lengths required by native velocity conversion. Four analytic velocity-command
cases cover all rollout block offsets. The training/runtime conversion path
already used the correct object and is unchanged.

The supervisor now supports recovery from a failed run. Before reusing the
completed first arm it checks the source/dataset hashes, coordinate contract,
batch/precision/update budget, selected checkpoints, exported bundle and flat
review bindings. Previous logs and partial review output remain available.
Recovery repeats the failed terrain review, then starts the second arm with the
same planned training configuration. No first-arm training is repeated.
Eleven focused evaluation/recovery regression tests passed.

The repaired real terrain review completed all24 windows in both history modes
(48 cases). Teacher-history versus generated-history mean Root-Y errors were
0.02925m versus0.07732m; FK MPJPE0.08243m versus0.20742m. Mean penetration was
0.00616m versus0.01329m. This is continued evidence of feedback degradation,
not a motion-quality pass. The second arm has now started actual Tokenizer and
Root optimizer updates; its first recorded losses were0.08988 and0.07962.
No data generation or first-arm retraining was started during recovery.
