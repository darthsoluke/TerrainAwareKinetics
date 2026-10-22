# Scene Control V3: Untrained Capacity Probe

Status: **UNTRAINED capacity probe**, `architecture_capacity_only`.
Actor schema: `motionbricks.scene_control_v3.actor.v1` (final 16-D scene input).
This phase performs zero optimizer steps. Learned WASD usability, motion quality,
and quality response are unmeasured. Existing formal 38.4M exposures, v2 code,
services, pages, and prior diagnostic results are unchanged.

The proposed RTX3080 10GB target is sustained 30 fps, necessary inference P99
at most 20 ms, and command-associated display P95 at most 100 ms. These are
measurement targets, not results from the model CPU regressions.

## Public Actor API

```python
from motionbricks.scene_control_v3 import CausalSceneActor, SceneActorConfig

actor = CausalSceneActor(SceneActorConfig(hidden_dim=256, preview_frames=16))
hidden = actor.init_hidden(batch_size, device=None, dtype=None)
outputs = actor(history, command, contact_state, contact_valid,
                scene_features, scene_valid, hidden)
```

`init_hidden` defaults to the model parameter device/dtype. `H` is the configured
hidden width; capacity profiles 256, 512, and 1024 use the same public API.
One invocation advances one 30 Hz step. Units are metres and seconds, Y is up.
Final 16-D profile parameter counts are 1,059,694 (H=256), 2,808,686 (H=512),
and 8,665,966 (H=1024).

| Input | Shape | Meaning |
| --- | --- | --- |
| `history` | `[B,4,366]` | Observed native global366 history, chronological |
| `command` | `[B,3]` | Desired `vx`, `vz` in m/s and heading in radians |
| `contact_state` | `[B,15,16]` | Packed factual state and separate proposals |
| `contact_valid` | `[B,15]` bool | Factual observation validity, not head confidence |
| `scene_features` | `[B,512,16]` | Caller-supplied ego scene queries |
| `scene_valid` | `[B,512]` bool | Explicit geometry observation validity |
| `hidden` | `[B,H]` | Bounded caller-owned recurrent memory |

| Output | Shape | Meaning |
| --- | --- | --- |
| `root_delta` | `[B,3]` | World XYZ displacement in metres per 30 Hz step |
| `heading_delta` | `[B,1]` | Auxiliary predicted radians per step |
| `global_joint_rot6d` | `[B,30,6]` | World-global SOMA30 native column6D rotations |
| `contact_logits` | `[B,15]` | Predicted contact logits, not actual evidence |
| `release_logits` | `[B,15]` | Predicted release logits, not actual release |
| `root_preview` | `[B,N,3]` | Ego displacement from the current root at offsets 1..N |
| `hidden` | `[B,H]` | Next recurrent memory |
| `input_valid` | `[B]` bool | At least one known geometry query in that row |

Native column6D means the first matrix column XYZ followed by the second column
XYZ, as implemented by native `cont6d_to_matrix`. Native history heading is derived
from predicted global rotations; `heading_delta` remains an auxiliary head.
The prototype retains raw global366 conditioning; an ego-invariant history
representation is outside this zero-training phase.

The scene feature order is: ego query XYZ (0..2), distance (3), ego normal (4..6),
body-part scalar ID (7), query time offset seconds (8), ego closest XYZ (9..11),
ego surface velocity XYZ in m/s (12..14), `distance_signed_flag` (15).
The last flag is 1 for closed-volume signed distance semantics, 0 for open/unsigned
distance. It is not an inside/outside indicator. This is a single 16-D contract;
there is no legacy 15-D path. Body-part IDs and the 15 contact slots remain
explicit caller-bound identities, with no inferred mapping to four native feet.

## Causal Computation

Scene query features are encoded once and attention-pooled into eight 64-D tokens.
The same tokens feed a shared history/command/contact fusion and one GRUCell call.
Root, body, contact, release, and preview heads consume the same recurrent state.
There are no four-pass Pose calls or full-horizon transformer recomputations.

The preview head predicts eight XYZ coefficients, decoded against a fixed time
basis for all requested offsets in one matrix multiplication. Changing N changes
only this final decode, not model parameter count or backbone work. Preview basis
buffers are excluded from the state dict so the same weights load at every N.
Common preview prefixes match across horizons. These predictions are not committed
body frames or postprocessed motion.

Memory is a fixed `[B,H]` state, initially zero. The GRU's bounded candidate and
gated recurrence keep state in [-1,1] when started with `init_hidden`. The actor
neither caches nor mutates history, input state, model buffers, or parameters.

Invalid scene/contact feature slots can contain NaN. They are replaced before
linear computation, not multiplied by zero afterwards. All-unknown scene rows
use explicit learned unknown tokens and zero coverage; `input_valid=False`.
**Frame publication must reject these rows**, checking the output validity tensor
alongside the normal device-to-host publication transfer. Unknown tokens are not
terrain. The tensor-only forward performs no `.item()`, `.numpy()`, or Python
scalar decisions based on tensor values.

## Factual Contact Lifecycle

`ContactLifecycleState.init(B)` creates unknown observations and fixed-size state.
`observe_contacts(state, observation_time_s=..., active=..., valid=...,
object_id=..., material_id=..., anchor_ego_m=..., normal_ego=...,
surface_velocity_ego_mps=...)` admits CPU factual observations and returns new state.
Inputs have `[B,15]` scalar or `[B,15,3]` vector shapes, except time `[B]`.
Timestamps are retained as float64 to preserve 30 Hz intervals on long-running
clocks. CPU admission rejects nonfinite times, nonfinite known-active contact
geometry, and negative/non-int64 active binding IDs. Unknown geometry may be NaN.

`object_id` is the persistent object identity. **`material_id` is a persistent
material-point/contact-event anchor identity, not a material type.** Moving to
another bound point on the same object requires a different `material_id`.
An object or material-point change emits `REBOUND` and resets observed age.

`event` describes the newest accepted observation: `ACQUIRED`, `RELEASED`,
`REBOUND`, `OBSERVATION_LOST`, `OBSERVED_FREE`, or `NONE`. A continuous newer
observation emits `NONE`; `last_event_time_s` retains the latest non-NONE event
time. Duplicate/older observations leave the slot unchanged, including its
`observation_time_s`. Consumers process an event once per
`(batch, slot, observation_time_s)` so replay cannot retrigger it.

Unknown observations preserve last factual contact without asserting release,
freeze its observed age, and mark `observed_valid=False`. Re-observation starts a
new interval. Age advances from observation timestamps, not call count or predicted
probability. The state stores no growing event log or horizon cache.

`contact_features(state, proposed_contact_probability, predicted_release_probability)`
returns `(contact_state, contact_valid)` without modifying factual state. Its 16
channels are actual active (0), proposed probability (1), predicted release (2),
observed active age (3), observed inactive age (4), seconds since last event (5),
ego anchor XYZ (6..8), ego normal XYZ (9..11), ego surface velocity XYZ (12..14),
observation known (15). Unknown rows are masked, not supervised negatives.

## Scope And Verification

No optimizer, new training, old quality rerun, IK, foot locking, collision
projection, source Root substitution, smoothing, or fake frames are implemented.
GPU work and servers remain parent-allocated activities.

Focused tests use the supplied Python environment, `PYTHONPATH=.;out/vendor/kimodo`,
and two CPU threads. They cover tensor shapes, explicit validity, NaN masks,
unmodified inputs, command/history sensitivity, repeated causal state, bounded
120-step memory, CPU contact admission, point/object rebindings, observed timing,
fresh events, unknown observations, every tested preview horizon
(1/4/8/12/16/32/64), and hidden widths 256/512/1024. A strict `torch.export` graph
regression checks the tensor-only actor path.

Test files: `tests/test_scene_control_v3_model.py` and
`tests/test_scene_control_v3_state.py`. Result: **31 passed**, CPU two threads,
2026-10-18. The installed Torch export path emitted 14 `script_method`
deprecation warnings.
