"""Bounded v2 Pose training with explicit conditioning and immutable codec labels."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from .curriculum import HORIZONS, ITERATION_POLICY, remask_argmax


BATCH_SCHEMA = "motionbricks.scene_control_v2.training_batch.v1"
LINEAGE_SCHEMA = "motionbricks.scene_control_v2.final_codec_labels.v1"
CONFIG_SCHEMA = "motionbricks.scene_control_v2.training_config.v1"
SHA_FIELDS = ("codec_sha256", "vocabulary_id", "codebook_sha256", "encoder_sha256", "scene_encoder_sha256",
              "normalization_sha256", "label_manifest_sha256")


def validate_label_lineage(lineage):
    if lineage["schema"] != LINEAGE_SCHEMA or lineage["final"] is not True or lineage["frozen"] is not True:
        raise ValueError("Pose labels require a final, frozen v2 codec receipt")
    if lineage["data_kind"] not in ("engineering", "formal"):
        raise ValueError("label_lineage.data_kind must identify engineering or formal labels")
    for field in SHA_FIELDS:
        if not isinstance(lineage[field], str) or re.fullmatch("[0-9a-f]{64}", lineage[field]) is None:
            raise ValueError(f"label_lineage.{field} must be a SHA256")
    if lineage["frames_per_token"] != 4:
        raise ValueError("The v2 training/inference contract requires four frames per token")
    for name in ("heads", "entries_per_head"):
        if type(lineage[name]) is not int or lineage[name] < 1:
            raise ValueError(f"label_lineage.{name} must be positive")
    return dict(lineage)


def make_training_label_lineage(identity, *, data_kind, codec_sha256, scene_encoder_sha256,
                                normalization_sha256, label_manifest_sha256):
    """Map the codec worker's FrozenCodebookIdentity into a final-label receipt."""
    if normalization_sha256 != identity.normalization_sha256:
        raise ValueError("Final label normalization differs from the frozen codec identity")
    config = identity.quantizer_config
    lineage = {"schema": LINEAGE_SCHEMA, "final": True, "frozen": True, "data_kind": data_kind,
               "codec_sha256": codec_sha256, "vocabulary_id": identity.identity_id,
               "codebook_sha256": identity.codebook_sha256, "encoder_sha256": identity.encoder_sha256,
               "scene_encoder_sha256": scene_encoder_sha256, "normalization_sha256": normalization_sha256,
               "label_manifest_sha256": label_manifest_sha256, "frames_per_token": 4,
               "heads": config.num_heads, "entries_per_head": config.entries_per_head,
               "representation_id": identity.representation_id}
    return validate_label_lineage(lineage)


def attach_final_codec_labels(batch, cache, identity, *, lineage):
    """Read the codec worker's exact frozen-vocabulary cache, without re-encoding."""
    from .quantizer import read_token_cache

    checked = validate_label_lineage(lineage)
    if (checked["vocabulary_id"] != identity.identity_id or
            checked["codebook_sha256"] != identity.codebook_sha256 or
            checked["encoder_sha256"] != identity.encoder_sha256 or
            checked["normalization_sha256"] != identity.normalization_sha256 or
            checked["heads"] != identity.quantizer_config.num_heads or
            checked["entries_per_head"] != identity.quantizer_config.entries_per_head or
            checked["representation_id"] != identity.representation_id):
        raise ValueError("Final training receipt differs from codec frozen vocabulary")
    return dict(batch, pose_tokens=read_token_cache(cache, identity), label_lineage=checked)


def select_root_condition(source, predicted, *, root_source):
    """Select the declared teacher/predicted branch; both inputs remain unchanged."""
    size = len(root_source)
    if not size or set(root_source) - {"source", "predicted"}:
        raise ValueError("Root condition selection needs per-row source/predicted choices")
    selected = {"source": list(root_source)}
    for key in ("values", "height"):
        a, b = source[key], predicted[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape != b.shape or len(a) != size:
            raise ValueError(f"source/predicted root_condition.{key} shapes must match")
        choose = torch.tensor([x == "predicted" for x in root_source], device=a.device)
        selected[key] = torch.where(choose.reshape(size, *([1] * (a.ndim - 1))), b, a).detach()
    for key in ("target_id", "history_id", "condition_id"):
        _ids(source[key], size, "source." + key)
        _ids(predicted[key], size, "predicted." + key)
        selected[key] = [predicted[key][i] if branch == "predicted" else source[key][i]
                         for i, branch in enumerate(root_source)]
    return selected


def _tensor(value, shape, name, *, dtype=None, check_finite=True):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if check_finite and value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _ids(values, size, name):
    if not isinstance(values, (list, tuple)) or len(values) != size or any(
            not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{name} requires {size} nonempty string IDs")
    return tuple(values)


def dual_to_global_features(dual_motion, *, dual_rep):
    """Explicit raw dual-370 to native global-366 extraction, without padding."""
    if (not torch.is_tensor(dual_motion) or dual_motion.ndim != 3 or
            dual_motion.shape[-1] != 370 or not dual_motion.is_floating_point()):
        raise ValueError("dual_motion must be raw floating-point [B,T,370] dual features")
    if dual_rep.motion_rep_dim != 370 or dual_rep.global_motion_rep.motion_rep_dim != 366:
        raise ValueError("Expected native SOMA dual370/global366 representation indices")
    return dual_rep.get_feature_subset(dual_motion, "global")


def validate_training_batch(batch, *, lineage, policy_id):
    """Admission only: never edits source futures to fit a generated history."""
    if batch["schema"] != BATCH_SCHEMA:
        raise ValueError("Expected independent scene_control_v2 training batch schema")
    if batch["data_kind"] not in ("engineering", "formal"):
        raise ValueError("data_kind must identify engineering fixtures or formal data")
    expected = validate_label_lineage(lineage)
    if batch["label_lineage"] != expected:
        raise ValueError("Pose label lineage differs from the final frozen codec")
    world = batch["world"]
    if not torch.is_tensor(world) or world.ndim != 3 or world.shape[-1] != 366:
        raise ValueError("world must have native global shape [B,T+1,366]; convert dual370 explicitly")
    size, endpoints, _ = world.shape
    frames = endpoints - 1
    horizon = frames // 4
    if not size or frames != horizon * 4 or horizon not in HORIZONS:
        raise ValueError("world must contain exactly 33/49/65 frames for horizons 8/12/16")
    if not world.is_floating_point():
        raise ValueError("world/history must be raw floating-point global motion")
    _tensor(world, (size, frames + 1, 366), "world")
    _tensor(batch["history"], (size, 4, 366), "history", dtype=world.dtype)
    if not torch.equal(world[:, :4], batch["history"]):
        raise ValueError("world[:,:4] must equal actual history; provide a matched recovery target, not a source splice")
    ids = {name: _ids(batch[name], size, name) for name in
           ("source_id", "family_id", "scene_id", "policy_id", "target_id", "history_id",
            "root_source", "history_source")}
    if any(p != policy_id for p in ids["policy_id"]):
        raise ValueError("Batch observation policy differs from the configured inference policy")
    if set(ids["root_source"]) - {"source", "predicted"} or set(ids["history_source"]) - {"source", "generated"}:
        raise ValueError("Unknown Root/history source")
    root = batch["root_condition"]
    values = root["values"]
    if not torch.is_tensor(values) or values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError("root_condition.values requires [B,T,R] normalized native features")
    _tensor(values, (size, frames, values.shape[-1]), "root_condition.values", dtype=world.dtype)
    _tensor(root["height"], (size, frames, 1), "root_condition.height", dtype=world.dtype)
    for name in ("target_id", "history_id"):
        if _ids(root[name], size, "root_condition." + name) != ids[name]:
            raise ValueError(f"root_condition.{name} does not match the selected target/history")
    if _ids(root["source"], size, "root_condition.source") != ids["root_source"]:
        raise ValueError("Root branch identity does not match selected conditions")
    condition_ids = _ids(root["condition_id"], size, "root_condition.condition_id")
    for name in ("observation", "volume"):
        observation = batch[name]
        if observation["policy_id"] != policy_id or observation["horizon_tokens"] != horizon:
            raise ValueError(f"{name} must match the inference sampling version and training horizon")
        if _ids(observation["root_condition_id"], size, name + ".root_condition_id") != condition_ids:
            raise ValueError(f"{name} was not sampled along the selected Root condition")
        if _ids(observation["history_id"], size, name + ".history_id") != ids["history_id"]:
            raise ValueError(f"{name} uses stale history")
    recovery = batch["recovery"]
    if recovery is None:
        if "generated" in ids["history_source"]:
            raise ValueError("Generated history needs an accepted continuous teacher target bound to this history")
    else:
        _tensor(recovery["accepted"], (size,), "recovery.accepted", dtype=torch.bool)
        for name in ("history_id", "target_id", "teacher_id"):
            _ids(recovery[name], size, "recovery." + name)
        for i, source in enumerate(ids["history_source"]):
            if source == "generated" and (not bool(recovery["accepted"][i]) or
                    recovery["history_id"][i] != ids["history_id"][i] or
                    recovery["target_id"][i] != ids["target_id"][i]):
                raise ValueError("Generated history needs an accepted continuous teacher target bound to this history")
    heads = expected["heads"]
    _tensor(batch["pose_tokens"], (size, horizon, heads), "pose_tokens", dtype=torch.long)
    if ((batch["pose_tokens"] < 0) | (batch["pose_tokens"] >= expected["entries_per_head"])).any():
        raise ValueError("Pose labels must be per-head final-codec indices, not packed IDs")
    _tensor(batch["masks"]["frames"], (size, frames), "masks.frames", dtype=torch.bool)
    _tensor(batch["masks"]["motion_loss_frames"], (size, frames), "masks.motion_loss_frames", dtype=torch.bool)
    _tensor(batch["masks"]["tokens"], (size, horizon, heads), "masks.tokens", dtype=torch.bool)
    if (batch["masks"]["motion_loss_frames"] & ~batch["masks"]["frames"]).any():
        raise ValueError("Motion-loss frames must be valid observed/target frames")
    if not batch["masks"]["tokens"].flatten(1).any(-1).all():
        raise ValueError("Every row needs at least one supervised valid codec token")
    contact = batch["contact_targets"]
    if not torch.is_tensor(contact) or contact.ndim != 3 or contact.shape[-1] < 1:
        raise ValueError("contact_targets requires [B,T,C]")
    _tensor(contact, (size, frames, contact.shape[-1]), "contact_targets", check_finite=False)
    _tensor(batch["contact_valid"], contact.shape, "contact_valid", dtype=torch.bool)
    known_contact = contact[batch["contact_valid"]]
    if not torch.isfinite(known_contact).all() or ((known_contact < 0) | (known_contact > 1)).any():
        raise ValueError("Valid contact_targets must be finite probabilities in [0,1]")
    return {"batch_size": size, "horizon_tokens": horizon,
            "root_sources": dict(Counter(ids["root_source"])),
            "history_sources": dict(Counter(ids["history_source"]))}


def prepare_native_pose_inputs(backbone, batch):
    """Native normalized history features; root values are explicit provider outputs."""
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    history = batch["history"]
    size, frames = len(history), batch["world"].shape[1] - 1
    _tensor(history, (size, 4, 366), "history (native global)")
    dual = backbone.motion_rep.dual_rep
    local = dual.global_to_local(
        history, is_normalized=False, to_normalize=True,
        lengths=torch.full((size,), 4, dtype=torch.long, device=history.device))
    pose = extract_feature_from_motion_rep(local, backbone.local_motion_rep,
                                          backbone._args["local_pose_feature"])
    condition = history.new_zeros(size, frames, pose.shape[-1])
    condition[:, :4] = pose
    known = torch.arange(frames, device=history.device)[None].expand(size, -1) < 4
    return {"local_root_values": batch["root_condition"]["values"].detach(),
            "root_height_trajectory": batch["root_condition"]["height"].detach(),
            "pose_cond": condition.detach(), "has_pose_cond": known,
            "num_tokens": torch.full((size, 1), frames // 4, device=history.device, dtype=torch.long)}


class _HeightAndVolumeEncoder(nn.Module):
    def __init__(self, height_encoder, volume_encoder, volume_width, pose_width):
        super().__init__()
        self.height_encoder = height_encoder
        self.volume_encoder = volume_encoder
        self.volume_projection = nn.Linear(volume_width, pose_width)

    def forward(self, observation, batch_size, device, dtype):
        height, height_valid = self.height_encoder(observation, batch_size, device, dtype)
        memory, valid = self.volume_encoder(observation["scene_control_v2_volume"])
        if memory.ndim != 3 or len(memory) != batch_size or valid.shape != memory.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("volume_encoder must return memory[B,Q,D] and bool valid[B,Q]")
        return torch.cat((height, self.volume_projection(memory)), 1), torch.cat((height_valid, valid), 1)


class NativePoseAdapter(nn.Module):
    """Real native Pose forward, with explicit 3D scene memory and no interception."""

    def __init__(self, backbone, volume_encoder, volume_width):
        super().__init__()
        if not backbone._args.get("terrain_conditioning") or backbone.get_num_frames_per_token() != 4:
            raise ValueError("Native v2 adapter requires terrain-conditioned four-frame Pose tokens")
        if backbone._args["min_tokens"] > 8 or backbone._args["max_tokens"] < 16:
            raise ValueError("Native Pose architecture must support all trained horizons 8/12/16")
        backbone._terrain_encoder = _HeightAndVolumeEncoder(
            backbone._terrain_encoder, volume_encoder, volume_width, backbone._args["n_embd"])
        self.backbone = backbone

    def forward_pose(self, pose_tokens, batch):
        condition = prepare_native_pose_inputs(self.backbone, batch)
        observation = dict(batch["observation"], scene_control_v2_volume=batch["volume"])
        return self.backbone(pose_tokens, terrain_observation=observation, **condition)


def build_native_pose_adapter(*, pose_args, dual_stats_mean, dual_stats_std,
                              volume_encoder, volume_width, backbone_state):
    """Construct the real SOMA native backbone; None explicitly means fresh weights."""
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.motionlib.core.utils.stats import Stats

    stats = Stats()
    stats.register_from_tensors(dual_stats_mean, dual_stats_std)
    rep = GlobalRootGlobalJoints(30, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats)
    backbone = pose_backbone_network(rep, args=pose_args)
    if backbone_state is not None:
        backbone.load_state_dict(backbone_state, strict=True)
    return NativePoseAdapter(backbone, volume_encoder, volume_width)


class SceneControlTrainer(nn.Module):
    def __init__(self, model, *, lineage, policy_id):
        super().__init__()
        self.model = model
        self.lineage = validate_label_lineage(lineage)
        self.policy_id = policy_id

    def _logits(self, tokens, batch):
        logits = self.model.forward_pose(tokens, batch)["pose_logits"]
        _tensor(logits, (*tokens.shape, self.lineage["entries_per_head"]), "pose_logits")
        return logits

    def forward(self, batch, *, request, seed):
        summary = validate_training_batch(batch, lineage=self.lineage, policy_id=self.policy_id)
        if request.batch_size != summary["batch_size"] or request.horizon_tokens != summary["horizon_tokens"]:
            raise ValueError("Provider did not fulfill the requested batch size/horizon")
        if request.policy_id != self.policy_id or request.iteration_policy != ITERATION_POLICY:
            raise ValueError("Curriculum and trainer sampling policy differ")
        for key in ("root_source", "history_source"):
            if tuple(batch[key]) != getattr(request, key):
                raise ValueError(f"Provider did not fulfill requested {key}; no distribution substitution")
        labels, mask_id = batch["pose_tokens"], self.lineage["entries_per_head"]
        rng = torch.Generator(device=labels.device).manual_seed(seed)
        tokens = labels.clone()
        focused = torch.ones_like(labels, dtype=torch.bool)
        for i, mode in enumerate(request.token_mode):
            if mode in ("all_mask", "self_iteration"):
                tokens[i].fill_(mask_id)
            else:
                eligible = batch["masks"]["tokens"][i].flatten().nonzero().flatten()
                fraction = math.cos(float(torch.rand((), generator=rng, device=labels.device)) * math.pi / 2)
                count = max(1, int(len(eligible) * fraction))
                chosen = eligible[torch.randperm(len(eligible), generator=rng, device=labels.device)[:count]]
                focused[i].zero_()
                focused[i].view(-1)[chosen] = True
                masked_count = max(1, int(count * .8))
                tokens[i].view(-1)[chosen[:masked_count]] = mask_id
                remaining = chosen[masked_count:]
                random_values = torch.randint(mask_id, (len(remaining),), generator=rng, device=labels.device)
                tokens[i].view(-1)[remaining] = random_values
        rounds = max(request.self_iteration)
        if rounds:
            # Teacher states come from this model in inference mode, never source token corruption.
            was_training = self.model.training
            self.model.eval()
            own_tokens = torch.full_like(labels, mask_id)
            try:
                with torch.no_grad():
                    for iteration in range(1, rounds + 1):
                        own_tokens = remask_argmax(self._logits(own_tokens, batch),
                                                  next_iteration=iteration, mask_id=mask_id)
                        rows = [i for i, n in enumerate(request.self_iteration) if n == iteration]
                        tokens[rows] = own_tokens[rows]
            finally:
                self.model.train(was_training)
        logits = self._logits(tokens, batch)
        focus = focused & batch["masks"]["tokens"]
        losses = F.cross_entropy(logits.reshape(-1, mask_id), labels.reshape(-1), reduction="none").reshape_as(labels)
        loss = losses[focus].mean()
        return {"loss": loss, "pose_ce": loss.detach(), "pose_logits": logits,
                "input_tokens": tokens.detach(), "focused_tokens": focus.detach(),
                "token_modes": dict(Counter(request.token_mode)), **summary}


@dataclass(frozen=True)
class TrainingBudget:
    kind: str
    max_batches: int
    batch_size: int
    max_exposures: int
    max_adam_updates: int
    approval_id: str = ""
    learning_rate: float = 0.0
    clip_norm: float = 0.0

    def validate(self, mode):
        if mode not in ("preflight", "forward", "train") or self.kind not in ("diagnostic", "formal"):
            raise ValueError("Explicit preflight/forward/train mode and diagnostic/formal budget required")
        for name in ("max_batches", "batch_size", "max_exposures", "max_adam_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"budget.{name} must be a nonnegative integer")
        if not self.max_batches or not self.batch_size or self.max_exposures != self.max_batches * self.batch_size:
            raise ValueError("Explicit bounded budget requires max_exposures == max_batches * batch_size")
        if mode == "train":
            if self.max_adam_updates != self.max_batches or not self.approval_id.strip():
                raise ValueError("Optimizer execution requires approval_id and an exact Adam-update budget")
            if not all(math.isfinite(x) and x > 0 for x in (self.learning_rate, self.clip_norm)):
                raise ValueError("Optimizer execution requires explicit positive learning_rate and clip_norm")
        elif self.max_adam_updates:
            raise ValueError("Preflight/forward budgets must specify zero Adam updates")
        return self


def preflight_training_config(config, *, mode="preflight"):
    from .curriculum import Curriculum

    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError("Expected independent v2 training configuration")
    lineage = validate_label_lineage(config["label_lineage"])
    curriculum = Curriculum.from_dict(config["curriculum"], policy_id=config["policy_id"])
    budget = TrainingBudget(**config["budget"]).validate(mode)
    if mode == "train" and budget.kind == "formal" and lineage["data_kind"] != "formal":
        raise ValueError("Engineering label identities cannot enter formal training")
    return {"schema": CONFIG_SCHEMA, "mode": mode, "label_lineage": lineage,
            "policy_id": curriculum.policy_id, "iteration_policy": ITERATION_POLICY,
            "horizons": list(HORIZONS), "budget": asdict(budget),
            "actual_adam_updates": 0, "formal_exposures": 0, "quality_accepted": False}


def run_bounded_trial(trainer, batch_provider, *, curriculum, budget, output, mode, seed=2333):
    """Provider(request) must return the requested actual source/recovery batch."""
    budget.validate(mode)
    if mode == "preflight":
        raise ValueError("Preflight uses preflight_training_config and does not construct a model")
    if mode == "train" and budget.kind == "formal" and trainer.lineage["data_kind"] != "formal":
        raise ValueError("Engineering label identities cannot enter formal training")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train = mode == "train"
    trainer.train(train)
    trainable = [(name, value) for name, value in trainer.named_parameters() if value.requires_grad]
    receipt = {"schema": "motionbricks.scene_control_v2.training_trial.v1", "mode": mode,
               "budget": asdict(budget), "label_lineage": trainer.lineage,
               "label_data_kind": trainer.lineage["data_kind"],
               "policy_id": trainer.policy_id, "iteration_policy": ITERATION_POLICY,
               "seed": seed, "objective": "focused_valid_codec_token_ce",
               "trainable_parameters": [name for name, _ in trainable],
               "trainable_parameter_elements": sum(value.numel() for _, value in trainable),
               "actual_adam_updates": 0, "diagnostic_adam_updates": 0, "formal_adam_updates": 0,
               "forward_examples": 0, "diagnostic_exposures": 0, "formal_exposures": 0,
               "model_forward_calls": 0,
               "observed_distributions": {key: {} for key in ("horizons", "token_modes", "root_sources", "history_sources")},
               "quality_accepted": False, "status": "running"}
    try:
        optimizer = torch.optim.Adam([value for _, value in trainable], lr=budget.learning_rate) if train else None
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as log:
            for index in range(budget.max_batches):
                seen = index * budget.batch_size
                request = curriculum.draw(budget.batch_size, exposure=seen, seed=seed + index)
                batch = batch_provider(request)
                if train and budget.kind == "formal" and batch["data_kind"] != "formal":
                    raise ValueError("Engineering fixtures cannot be counted as formal training exposures")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(train):
                    result = trainer(batch, request=request, seed=seed + index)
                    if train:
                        result["loss"].backward()
                        nn.utils.clip_grad_norm_(trainer.parameters(), budget.clip_norm, error_if_nonfinite=True)
                        optimizer.step()
                receipt["forward_examples"] += budget.batch_size
                receipt["model_forward_calls"] += 1 + max(request.self_iteration)
                distributions = receipt["observed_distributions"]
                horizon_key = str(request.horizon_tokens)
                distributions["horizons"][horizon_key] = distributions["horizons"].get(horizon_key, 0) + budget.batch_size
                for name in ("token_modes", "root_sources", "history_sources"):
                    distributions[name] = dict(Counter(distributions[name]) + Counter(result[name]))
                if train:
                    receipt["actual_adam_updates"] += 1
                    receipt[budget.kind + "_adam_updates"] += 1
                    receipt[budget.kind + "_exposures"] += budget.batch_size
                row = {"batch": index, "pose_ce": float(result["pose_ce"]),
                       "horizon_tokens": request.horizon_tokens, "token_modes": result["token_modes"],
                       "root_sources": result["root_sources"], "history_sources": result["history_sources"],
                       "actual_adam_updates": receipt["actual_adam_updates"]}
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
        if train:
            with (output / "checkpoint.pt").open("xb") as checkpoint:
                torch.save({"schema": receipt["schema"], "model": trainer.state_dict(),
                            "optimizer": optimizer.state_dict(), "receipt": dict(receipt, status="complete")}, checkpoint)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt
