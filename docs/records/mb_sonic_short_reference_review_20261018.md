# SONIC Short G1 Reference Review

Reviewed through 2026-10-19 (Asia/Singapore); requested filename retained.
Scope: bounded, read-only configuration/source inspection.
Pinned HF revision: `6733128a3d8a523b1418b06bca3cdf61c8b0987f`.

## Decision

**PASS: the official low-latency variant has a genuinely shorter G1-mode future
window, not merely a shorter SMPL mode.** G1 still uses ten reference samples,
but step1 at 50 Hz means offsets **0,0.02,...,0.18 s**, versus original step5
offsets **0,0.10,...,0.90 s**. The training config independently declares
`num_future_frames=10`, `dt_future_ref_frames=0.02`, and target FPS 50.
[Pinned deployment config](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/low_latency/observation_config.yaml),
[pinned training config](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/low_latency/config.yaml).

One existing eight-token MB call supplies 0.8 s trusted future. A **separate
candidate** using a fixed 0.18+0.10=0.28 s consumer horizon would fit one call,
with 15 dense 50 Hz reference samples and the same declared 0.10 s replan guard.
It would remove the second autoregressive MB call from that candidate's required
generation path. The original matched pair, two-call baseline, 1.0 s consumer
contract and full-proposal failure evidence stay unchanged.

Candidate adapter, ONNX execution, timing, physics and behavior: **NOT_RUN**.
The shorter reference requirement is established; total-chain speed is unmeasured.

## Strict Comparison

| Contract | Original Baseline | Official `low_latency` |
|---|---|---|
| G1 modality | mode 0, 29 joint positions/velocities + root orientation | same G1 modality, mode 0 |
| G1 reference samples | 10, step5 | 10, step1 |
| Required future span | 0.90 s | 0.18 s |
| Relative root orientation | actual full-body orientation, not heading-only | same body-frame convention |
| Deployment encoder packing | 1762 floats | 1247 floats, different order |
| Token output | 64, `use_fp16=false` | same declared token size/precision |
| Decoder input | 994 = token 64 + proprioception 930 | same declared 994 |
| Actual proprioception history | 10 samples, step1, 0.18 s past span | unchanged |
| Policy/physics timing | 50 Hz; 0.005 s with decimation 4 | same training declaration |
| Actor running mean/std | false | false |
| Robot selector/default offset | `g1_model_12_dex`, true | same |
| Training action clipping | 20 | 20 |
| SMPL future count | 10 | 4; not used for this G1 decision |

The original training spacing is explicitly 0.1 s, independently confirming its
0.9 s G1 span. The low-latency mode's ten-frame G1 spacing is 0.02 s.
[Original pinned training config](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_release/config.yaml).

### Packing And Normalization

Low-latency G1 uses `motion_joint_positions_10frame_step1`,
`motion_joint_velocities_10frame_step1`, and
`motion_anchor_orientation_10frame_step1`. Its deployment YAML places the
multi-frame orientation **before** the single-frame orientation; the original
does the reverse and also includes two root-height entries. Low-latency removes
those height entries and reduces the inactive SMPL/wrist sections to four frames.
The enabled dimensions sum to 1247, with G1 slices positions [4:294], velocities
[294:584], orientation [584:644]. Non-G1 sections remain mode-masked; G1 mode 0
keeps the official all-zero four-element mode code.

The small resolved model config records training tokenizer dimension 1257;
deployment packing uses 1247. The arithmetic difference is compatible with
dropping 11 height inputs and expanding the mode field from 3 to 4, but actual
ONNX input names/shapes and export behavior still need verification against the
matched pair. Do not feed the training dimension directly to the deployment graph.
[Pinned resolved model config](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/low_latency/model_config.yaml).

The current source keeps reference joint angles/velocities in SI, measured
proprioceptive joint position relative to default angles, gyro and joint velocity
in rad/s, gravity in the body frame, and previous actions in policy order. Both
actor configs disable running mean/std; critic RMS is unrelated to deployment.
No external input normalizer was specified in these files. Embedded ONNX
operations were **NOT_RUN**, because model binaries were not fetched.

The checked C++ registry maps both G1 orientation names to orientation mode 0:
inverse actual base quaternion times heading-aligned reference quaternion,
first two rotation-matrix columns flattened row-wise. It is not the v1.1
heading-only/reference-heading variant. Current registry has the G1 step1
entries but lacks the YAML's SMPL four-frame entries: using the entire YAML with
that C++ binary requires registry completion/rebuild even when selecting G1.
The local Python builder deliberately accepts only the original schema today.
Sources: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
lines 612-687 and 1704-1764; `motionbricks/mb_sonic/sonic_policy.py` lines 107-172.

### Gains And Action Scale

Neither inspected training YAML embeds a distinct low-latency gain/scale override.
Both select the same robot configuration and default-position offset. Current
source maps that selector to the common model-12 actuator configuration and
`action_scale=0.25*effort_limit/stiffness`, with natural frequency 10 Hz and damping
ratio 2. Default offsets include hip pitch -0.312, knee 0.669, ankle pitch -0.363,
elbow 0.6, shoulder pitch 0.2 and shoulder roll +/-0.2 rad.

Current deployment constants, representative joint groups:

| Joint Group | Kp | Kd | Action Scale (rad) |
|---|---:|---:|---:|
| hip pitch/roll, knee | 99.09843 | 6.30880 | 0.3506615 |
| hip yaw, waist yaw | 40.17924 | 2.55789 | 0.5475465 |
| ankle pitch/roll, waist pitch/roll | 28.50125 | 1.81445 | 0.4385773 |
| shoulder, elbow, wrist roll | 14.25062 | 0.90722 | 0.4385773 |
| wrist pitch/yaw | 16.77833 | 1.06814 | 0.07450087 |

This is the shared source configuration, not a numerical actuator snapshot
embedded in the downloaded low-latency YAML. Candidate E0/E1 must bind the actual
XML, defaults, all 29 scales/gains and torque limits by hash and inspect the
matched graph. Sources: `gear_sonic/envs/manager_env/robots/g1.py` lines 9-27,
223-371; `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` lines 999-1017;
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`.

## Matched Pair And Cost

The pinned HF tree lists a matched low-latency encoder/decoder/config set.
Metadata only, no model bytes downloaded:

| File | Bytes | Declared LFS SHA256 |
|---|---:|---|
| `low_latency/model_encoder.onnx` | 45,933,505 | `60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b` |
| `low_latency/model_decoder.onnx` | 149,763,762 | `c4ac2e74045e7cbfb568f15e6bf47ea7ce023df7a94322af50be223e0a628bab` |

The original actual decoder is 40,900,688 bytes. Low-latency's decoder is 3.66x
larger, and its configured dynamic decoder hidden widths are
4096/4096/2048/2048/1024/1024/512/512 rather than original
2048/2048/1024/1024/512/512. Shorter lookahead does not establish faster ONNX
inference. The first MB call and the entire policy/physics pipeline still need
their own timing; removing the second call cannot inherit a real-time PASS.
[Pinned file tree](https://huggingface.co/nvidia/GEAR-SONIC/tree/6733128a3d8a523b1418b06bca3cdf61c8b0987f/low_latency).

## Bounded Next Work

1. Parent-approved separate variant assets and hashes; never mix original and
   low-latency encoders/decoders or silently replace the original baseline.
2. Separate exact 1247-field builder/mask and offsets 0..9 at 50 Hz; keep the
   real ten-sample decoder history and full-body orientation semantics. Existing
   original globals/fixtures remain frozen.
3. Separate one-call eight-token MB proposal and uniform 0.28 s candidate packet;
   retain raw 0.8 s trusted proposal, all joint-limit diagnostics, actual-history
   separation, stale-state/intent checks, and no padding/clipping/rebasing.
4. Independent B0/E1: graph I/O/dtypes, golden observations and heading/joint
   perturbations, policy outputs, gains/scales, actual contacts/no auxiliary
   forces, timestep convergence and full timing. Then one-call E3 measurements.

No implementation, GPU use, policy inference, physics/server run, model download,
training, SSH, or agreement acceptance occurred in this review.

## Configuration Receipts

Actual bytes read from the pinned resolve URLs; hashes were computed in memory:

| File | Bytes | SHA256 |
|---|---:|---|
| `low_latency/observation_config.yaml` | 3,258 | `582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab` |
| `low_latency/model_config.yaml` | 8,826 | `e8577de81e4d0098d5cacd1a58457060c0312bc7e5b1a9f0a63a83acbc39b65d` |
| `low_latency/config.yaml` | 29,063 | `f16ebaf1ec793cea1566121bbd034fe49a7e81500244791dd6ee6eb794108e2c` |
| `sonic_release/config.yaml` | 28,331 | `f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7` |
| original `observation_config.yaml` | 2,336 | `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b` |
