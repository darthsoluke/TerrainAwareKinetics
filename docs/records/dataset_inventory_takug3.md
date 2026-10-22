# Training Asset Inventory: taku-g3

Audit date: 2026-10-04. SSH account: private-user. Target: a terrain-aware human locomotion model with a purely kinematic runtime.

Scope update: the agreed first-stage experiment is BONES-SEED root-trajectory-conditioned locomotion with supervised foot contacts, no heightmap input, and RTX 3080 60 Hz inference. See `bones_seed_locomotion_training_plan.md`. Terrain dataset acquisition and heightmap validation below remain later-stage requirements; they do not block this BONES-only V1. The inventory itself remains unchanged.

This is an inventory, not a training-readiness certificate. No training or large dataset download was started. Existing source data and other users' code were not modified.

## Verified Existing Data

| Asset | Observed quantity | Location | Current validation |
| --- | --- | --- | --- |
| BONES-SEED SOMA Uniform originals | 71,132 BVH files; 138.192 GiB logical file size | `${PRIVATE_STORAGE}/bones-seed/uniform_orig/soma_uniform/bvh` | All filenames/sizes counted; eight headers parsed; not all motion rows validated |
| MotionPersonaX release | 128 BVH archives, 128 SMPL archives, 128 shape NPZ files; release directory 200.640 GiB | `${PRIVATE_STORAGE}/release/MotionPersonaX` | Manifest parsed; both archive sets match all 128 body IDs; archive contents/integrity not fully verified |
| MotionPersona refined subset | 365 BVH files; 2.407 GiB | `${PRIVATE_STORAGE}/t7/motionpersona/100style++/refined/BVH` | All filenames/sizes counted; eight headers parsed |
| MotionPersona indexed source collection | 2,993 source paths, including raw/refined versions | `${PRIVATE_STORAGE}/t7/motionpersona/100style++` | 32 indexed paths sampled; all accessible; not a unique-clip count |
| Motion100 processed | 237 BVH files; 15.093 GiB | `${PRIVATE_STORAGE}/t7/motion100/motion100/processed_add_T_pose` | All filenames/sizes counted; eight headers parsed |
| LAFAN1-derived test excerpts | 29 BVH files; 109.826 MiB; 29,000 declared frames; approximately 16.11 minutes | `${PRIVATE_STORAGE}/code/motionfit/test_data/lafan1_*.bvh` | All 29 headers parsed; every file declares 1,000 frames; not the original full dataset |
| BEAT indexed source collection | 1,945 indexed source paths | `${PRIVATE_STORAGE}/t7/beat/beat_english_v0.2.1/beat_english_v0.2.1` and `${PRIVATE_STORAGE}/beat2/beat_english_v2.0.0` | 32 indexed paths sampled; all accessible; gesture-oriented data needs locomotion relevance filtering |

Logical file sizes above are not filesystem allocated-space measurements. Counts from source indexes must not be added to counts of their overlapping subsets.

The official BONES-SEED release has 71,132 originals plus 71,088 mirrored motions, not two equal halves. The observed original count matches the documented original set; full identity matching still requires metadata. Missing mirrors alone do not require downloading all originals again. Generate or recover mirrors only after metadata-based eligibility and train/test grouping are established. [Official dataset card](https://huggingface.co/datasets/bones-studio/seed)

## MotionPersonaX Manifest

The server's `manifest.csv` contains:

- 328,960 rows, 128 bodies and 2,570 distinct source clip names.
- 313,600 rows at 30 fps, 10,240 at 60 fps, and 5,120 at 15 fps.
- Approximately 4,112.31 expanded hours across body variants; approximately 32.13 hours per body.
- No missing or unexpected body archive names in either the BVH or SMPL archive directories.

Expanded body variants are not thousands of hours of independently captured behavior. Keep every body variant of a source recording in the same data split. Do not assume this server manifest is uniformly 30 fps or identical to every subsequent public release.

## Skeleton Differences

Header counts include named ROOT/JOINT entries but exclude BVH End Site blocks. They are not a declaration of the model's reduced joint subset.

| Observed source | Named joints | Header fps | Scope |
| --- | --- | --- | --- |
| BONES-SEED SOMA Uniform | 78 | Approximately 120 | Eight sampled headers; root names include `Root` and `Hips`; body, hands and face joints |
| MotionPersona raw refined BVH | 60 | Approximately 120 | Eight sampled headers |
| MotionPersona public example BVH | 23 | Approximately 30 | Eight sampled headers; pelvis-rooted SMPL-style body skeleton |
| Motion100 processed BVH | 98 | Approximately 30 | Eight sampled headers; includes auxiliary and marker joints |
| LAFAN1-derived test excerpts | 75 | Approximately 30 | All 29 headers; these are converted excerpts, not canonical Ubisoft BVHs |

Do not concatenate these arrays by joint index or infer that a 27-joint MotionBricks model consumes the full SOMA BVH directly. Verify the target checkpoint's exact ordered joints, parents, reference offsets/orientations, root convention and motion representation first. Every source needs an explicit, tested conversion into that contract.

BONES metadata, temporal labels, shared SOMA shape parameters and `soma_base_skel_minimal.bvh/.usd` were not found in the checked BONES/release/public-code directories. They may exist elsewhere or in inaccessible directories; their presence is currently unconfirmed. The official dataset documents these companions and requires authorized access for its files. [Official format and structure](https://huggingface.co/datasets/bones-studio/seed)

## Existing Methods and Weights

| Method | Verified accessible server assets | Limitation |
| --- | --- | --- |
| MotionPersona | Release-style code under `${PRIVATE_STORAGE}/code/persona_pub_data/code`; `checkpoints/codec.ckpt` is 84,109,655 bytes | No `prior.ckpt` in that checkpoint directory; official revision/hash not yet matched |
| MotionPersona research runs | `${PRIVATE_STORAGE}/code/persona_26/save/final_v5/latfm_full_v5_s2_t15/best.ckpt` and `vae_full_v5_s1_b2e4/best.ckpt`, with configs | Experimental prior/codec files exist; compatibility with public code has not been verified; checkpoints were not loaded |
| MotionBricks | Adapter/baseline under `${PRIVATE_STORAGE}/code/persona_26/baselines/motionbricks` | Adapter is not evidence of a complete official repository or human checkpoint |
| ARDY | No named project confirmed in scanned accessible directories | Official inference implementation exists; published checkpoint table lists Core/G1 and says SOMA is coming soon |
| PARC | No named project or release archive confirmed in scanned accessible directories | Official code and motion/terrain release available externally |
| PFNN | Old index lists 40 animation paths; sampled paths currently missing | No usable complete code/data bundle confirmed |
| Kimodo / DART | Complete official installations not confirmed | Do not count derived filenames or evaluation outputs as a ready installation |

The possible official MotionBricks checkout at `${PRIVATE_STORAGE}/code/GR00T-WholeBodyControl` is not accessible to `private-user`. The explicit paths `${PRIVATE_STORAGE}/code/GR00T-WholeBodyControl` and `${PRIVATE_STORAGE}/code/GR00T-WholeBodyControl` do not exist. The official local checkout is already available at `${UPSTREAM_ROOT}`; stage an audited revision in our own server directory when needed.

Official method references: [MotionPersona](https://github.com/AIGAnimation/MotionPersona), [ARDY](https://github.com/nv-tlabs/ardy), [MotionBricks](https://nvlabs.github.io/motionbricks/), [PARC](https://github.com/mshoe/PARC), [PFNN](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control).

## Stale Indexes and Processing Status

`${PRIVATE_STORAGE}/full_corpus/farm.tsv` contains LAFAN1, PFNN, AMASS, 100STYLE, MotionX, CMU and other entries. In each of these groups, all 32 sampled source paths were missing at their indexed locations. This does not prove the datasets are absent from every disk or private directory; it does mean this index cannot serve as an availability manifest.

Examples of stale roots:

- LAFAN1: `${PRIVATE_STORAGE}/t7/lafan1/ubisoft-laforge-animation-dataset/lafan1/lafan1`.
- Retargeted LAFAN1: `${PRIVATE_STORAGE}/p05/geno/lafan1-resolved/bvh`.
- PFNN: `${PRIVATE_STORAGE}/t7/pfnn/pfnn/data/animations`.
- 100STYLE: `${PRIVATE_STORAGE}/t7/100style/100style/100STYLE_raw`.
- AMASS BVHs: `${PRIVATE_STORAGE}/t7/amass/amass/amass_bvh`.

InterAct/ZeroEGGS paths under `${PRIVATE_STORAGE}/dataset` returned permission errors and are classified as inaccessible, not missing.

The existing `ingest_status.txt` records an initial BONES processing failure followed by a successful retry (`fc_bones_seed rc=0`). BEAT, MotionPersona and Motion100 also have successful exit records. Exit codes and ready flags do not establish compatibility with our future skeleton, contact and terrain representation. Existing caches must be verified before reuse.

## Acquisition Priorities

1. Recover the full official LAFAN1 BVH set: 77 sequences, 496,672 frames at 30 fps, approximately 4.6 hours. Verify the Git LFS ZIP is real data, not a pointer. The official license is CC BY-NC-ND 4.0; preserve the license and review intended use before redistribution/product use. [Official source](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
2. Add PARC's official motion/terrain pairs and preprocessing source. The current Hugging Face listing provides `dec_release.zip` (3.91 GB) and `smaller_model.zip` (32.4 MB). Released paired data can be reused for a kinematic training backend without making a physics simulator part of the gameplay runtime. [Official repository](https://github.com/mshoe/PARC), [official files](https://huggingface.co/datasets/mxucg/PARC/tree/main)
3. Recover PFNN's original Code & Data bundle, including animations, contact/phase annotations, terrain patches and preprocessing scripts. Its terrain-fitting procedure is a reference, not an excuse to attach arbitrary heightmaps to unrelated motions. Download availability and archive contents remain unverified. [Author's project](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control)
4. Complete BONES metadata/temporal labels/SOMA rig and shape companions through an already authorized account or an existing authorized copy. Do not download another full SOMA Uniform archive just to obtain small companion files.
5. Validate MotionPersonaX archives and stage its flat training format only if selected for style/body-conditioned locomotion. Its body variants must not leak across splits. Add full AMASS/100STYLE or other corpora later as justified by locomotion coverage and licensing, not solely raw clip count.

No missing assets in this list have been downloaded during this audit.

## Training Gate

Training remains disabled until the chosen first-stage corpus passes all of the following:

- A versioned availability manifest with source revision, provenance/license, content hashes, declared/actual frame counts, duration and archive integrity; deduplicate recordings and their converted variants.
- One pinned skeleton contract with names/order/parents, rest transforms, scale and coordinate conventions. Check FK before/after conversion and round-trip representative clips through UE retargeting.
- A documented target sampling rate; quaternion interpolation and appropriate downsampling; velocities recomputed from actual timestamps; no concatenation across clip boundaries.
- Locomotion selection and balanced coverage of idle, start/stop, walk/run, turns, strafing/backward movement, crouching and jumps where genuinely available. Record absent behaviors instead of assuming coverage.
- Contact validation using toe/heel geometry, motion and terrain height. Preserve airborne frames and reject severe penetration, foot sliding, teleportation, invalid rotations and nonfinite values.
- Terrain features computed in the same root-local frame, yaw and metric units during preprocessing and UE inference. Verify sampling positions, grid orientation, height origin, footprint and obstacle/clearance handling on controlled ramps/steps. Missing terrain is an explicit validity mask, not a fabricated flat-ground label; obstacle clips need compatible geometry.
- Splits grouped by original recording and actor as appropriate. Keep mirrors, body retargets, overlapping windows and alternate conversions together; hold out terrain layouts for terrain generalization evaluation. Normalization statistics use training data only.
- Visual checks and quantitative conversion/terrain tests passed before building final training shards. Successful historical ingestion alone is insufficient.

## Audit Artifacts

- Server report: `${PRIVATE_STORAGE}/motionbricks-preparation/audit/inventory_20261004.json`.
- Server script: `${PRIVATE_STORAGE}/motionbricks-preparation/audit/audit_remote_motion_assets.py`.
- Local machine-readable report: `${PROJECT_ROOT}\docs\dataset_inventory_takug3_20261004.json`.
- Local script: `${PROJECT_ROOT}\scripts\audit_remote_motion_assets.py`.

The script inventories files and parses headers/CSV indexes without loading pickle-based checkpoints or modifying source datasets. Its reuse option explicitly retains previous directory inventory timestamps while refreshing manifests and index probes. No personal storage quota has been confirmed; choose staging locations based on writable space and quota before acquisition.
