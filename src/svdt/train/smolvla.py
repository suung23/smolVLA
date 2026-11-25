"""SmolVLA fine-tuning with trust-weighted samples.

How this hooks into LeRobot 0.6.1 (verified against its source, not assumed):

* ``SmolVLAPolicy.forward(batch, reduction="none")`` already returns per-sample
  losses -- the API exists upstream for exactly this purpose.
* ``lerobot.scripts.lerobot_train.update_policy`` accepts a ``sample_weighter``
  and computes ``(per_sample_loss * w).sum() / (w.sum() + eps)``.
* SmolVLA's own preprocessing, normalisation and action-chunk handling stay
  entirely with LeRobot's ``PolicyProcessorPipeline``; nothing here reimplements
  or bypasses them.

Two consequences worth stating plainly:

1. ``weighting.method="uniform"`` installs **no** weighter at all, so training
   takes LeRobot's original ``reduction="mean"`` path and is bit-identical to
   the unweighted baseline. Routing a uniform run through the weighted path
   would change the padding reduction (see ``weighting.padding_normalization``).
2. The dataloader must keep ``episode_index`` and ``frame_index`` in the batch,
   since weights are addressed by stable transition identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..env_info import collect, probe_import, resolve_device
from ..io_utils import ensure_dir, write_json
from ..scoring.sidecar import ScoreSidecar, load_sidecar
from ..weighting.sample_weighter import TrustSampleWeighter, register_with_lerobot


class TrainingUnavailable(RuntimeError):
    """SmolVLA training cannot run here; the message names what is missing."""


@dataclass
class SmolVLATrainingPlan:
    """Everything the run needs, resolved and recorded before anything starts.

    Built and written out even when training cannot run locally, so the exact
    configuration is reviewable (and reproducible elsewhere) from a machine
    without torch.
    """

    config: Config
    policy_path: str
    policy_revision: str | None
    chunk_size: int | None
    device: str
    train_episodes: list[int]
    val_episodes: list[int]
    weighting_method: str
    sidecar_dataset_ids: list[str] = field(default_factory=list)
    uses_weighter: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": {
                "type": self.config.policy.type,
                "pretrained_path": self.policy_path,
                "revision": self.policy_revision,
                "chunk_size": self.chunk_size,
                "device": self.device,
                "overrides": self.config.policy.overrides,
            },
            "dataset": {
                "repo_id": self.config.dataset.repo_id,
                "revision": self.config.dataset.revision,
                "root": self.config.dataset.root,
                "n_train_episodes": len(self.train_episodes),
                "n_val_episodes": len(self.val_episodes),
                "train_episodes": self.train_episodes,
                "val_episodes": self.val_episodes,
            },
            "weighting": {
                "method": self.weighting_method,
                "uses_weighter": self.uses_weighter,
                "alpha": self.config.weighting.effective_alpha(),
                "beta_visibility": self.config.weighting.beta_visibility,
                "beta_perception": self.config.weighting.beta_perception,
                "beta_reliability": self.config.weighting.beta_reliability,
                "w_min": self.config.weighting.w_min,
                "chunk_alignment": self.config.weighting.chunk_alignment,
                "padding_normalization": self.config.weighting.padding_normalization,
                "hard_filter_threshold": self.config.weighting.hard_filter_threshold,
                "sidecar_dataset_ids": self.sidecar_dataset_ids,
            },
            "train": {
                "steps": self.config.train.steps,
                "batch_size": self.config.train.batch_size,
                "seed": self.config.train.seed,
                "lr": self.config.train.lr,
                "output_dir": self.config.train.output_dir,
            },
            "notes": self.notes,
        }


def read_checkpoint_chunk_size(
    pretrained_path: str, revision: str | None = None
) -> int | None:
    """Read ``chunk_size`` from the checkpoint's own ``config.json``.

    Preferred over hard-coding 50: the value belongs to the checkpoint, and a
    future SmolVLA release may change it.
    """
    local = Path(pretrained_path) / "config.json"
    if local.exists():
        data = json.loads(local.read_text(encoding="utf-8"))
        return data.get("chunk_size")
    try:
        from ..data.hub import download_file

        path = download_file(
            pretrained_path, "config.json", repo_type="model", revision=revision
        )
    except Exception:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("chunk_size")
    except (OSError, json.JSONDecodeError):
        return None


def build_training_plan(
    cfg: Config,
    *,
    train_episodes: list[int],
    val_episodes: list[int] | None = None,
    sidecar: ScoreSidecar | None = None,
) -> SmolVLATrainingPlan:
    """Resolve the run configuration without starting it."""
    notes: list[str] = []
    chunk_size = cfg.policy.chunk_size
    if chunk_size is None:
        chunk_size = read_checkpoint_chunk_size(
            cfg.policy.pretrained_path, cfg.policy.revision
        )
        if chunk_size is None:
            notes.append(
                "chunk_size could not be read from the checkpoint; set "
                "policy.chunk_size explicitly if a chunk-aggregating alignment "
                "is used"
            )
        else:
            notes.append(f"chunk_size={chunk_size} read from the checkpoint config")

    uniform = cfg.weighting.method == "uniform"
    if uniform:
        notes.append(
            "uniform weighting installs no sample weighter, so training follows "
            "LeRobot's original reduction='mean' loss and is bit-identical to the "
            "unweighted baseline"
        )

    device = resolve_device(cfg.policy.device)
    if device == "cpu":
        notes.append(
            "no accelerator detected; SmolVLA fine-tuning on CPU is impractically "
            "slow and is intended only as a smoke test"
        )

    return SmolVLATrainingPlan(
        config=cfg,
        policy_path=cfg.policy.pretrained_path,
        policy_revision=cfg.policy.revision,
        chunk_size=chunk_size,
        device=device,
        train_episodes=sorted(train_episodes),
        val_episodes=sorted(val_episodes or []),
        weighting_method=cfg.weighting.method,
        sidecar_dataset_ids=[sidecar.dataset_id] if sidecar is not None else [],
        uses_weighter=not uniform,
        notes=notes,
    )


def make_weighter(
    cfg: Config,
    sidecar: ScoreSidecar,
    *,
    chunk_size: int | None,
    device: Any = None,
) -> TrustSampleWeighter | None:
    """Build the sample weighter, or ``None`` for the uniform baseline."""
    if cfg.weighting.method == "uniform":
        return None
    register_with_lerobot()
    return TrustSampleWeighter(
        sidecar,
        cfg.weighting,
        dataset_id=sidecar.dataset_id,
        chunk_size=chunk_size,
        device=device,
    )


def check_training_requirements() -> dict[str, Any]:
    """Report whether the real training path can run, and why not if it cannot."""
    torch_info = probe_import("torch")
    lerobot_info = probe_import("lerobot")
    blockers: list[str] = []
    if not torch_info.get("importable"):
        blockers.append(
            f"torch is not importable ({torch_info.get('error_type')}: "
            f"{str(torch_info.get('error'))[:160]})"
        )
    if not lerobot_info.get("importable"):
        blockers.append(
            "lerobot is not importable; install with pip install 'svdt[lerobot]'"
        )
    return {
        "can_train": not blockers,
        "blockers": blockers,
        "torch": torch_info,
        "lerobot": lerobot_info,
    }


def run_lerobot_training(
    cfg: Config,
    *,
    train_episodes: list[int],
    val_episodes: list[int] | None = None,
    sidecar_root: str | Path | None = None,
    sidecar_dataset_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fine-tune SmolVLA through LeRobot, with trust weighting if configured.

    Always writes a plan + run manifest first, so a run that cannot start on
    this machine still leaves a complete, reproducible record. ``dry_run`` stops
    after that.
    """
    sidecar: ScoreSidecar | None = None
    if cfg.weighting.method != "uniform":
        root = sidecar_root or cfg.paths.sidecar_root
        dataset_id = sidecar_dataset_id or cfg.dataset.name
        sidecar = load_sidecar(root, dataset_id)

    plan = build_training_plan(
        cfg,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        sidecar=sidecar,
    )
    out_dir = ensure_dir(Path(cfg.train.output_dir))
    requirements = check_training_requirements()
    manifest = {
        "kind": "train_run",
        "plan": plan.to_dict(),
        "requirements": requirements,
        "environment": collect(
            command="train", config=cfg.to_dict(), seed=cfg.train.seed
        ),
    }
    write_json(out_dir / "train_plan.json", manifest)

    if dry_run:
        return {"status": "dry_run", "plan": plan.to_dict(), "output_dir": str(out_dir)}
    if not requirements["can_train"]:
        raise TrainingUnavailable(
            "SmolVLA training cannot run here:\n  - "
            + "\n  - ".join(requirements["blockers"])
            + "\nThe resolved plan was still written to "
            f"{out_dir / 'train_plan.json'}; run `svdt train` on a machine with an "
            "importable torch and lerobot, or use --backend reference for the "
            "torch-free smoke test."
        )

    return _run_with_lerobot(cfg, plan, sidecar, out_dir)


def _run_with_lerobot(
    cfg: Config,
    plan: SmolVLATrainingPlan,
    sidecar: ScoreSidecar | None,
    out_dir: Path,
) -> dict[str, Any]:
    """The real fine-tuning loop. Only reached when torch and lerobot import."""
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    device = torch.device(plan.device)
    torch.manual_seed(cfg.train.seed)

    policy_cls_cfg = _make_policy_config(cfg)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=plan.train_episodes,
        revision=cfg.dataset.revision,
        delta_timestamps=None,
    )
    delta = resolve_delta_timestamps(policy_cls_cfg, dataset.meta)
    if delta is not None:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=plan.train_episodes,
            revision=cfg.dataset.revision,
            delta_timestamps=delta,
        )

    policy = make_policy(policy_cls_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cls_cfg,
        pretrained_path=cfg.policy.pretrained_path,
        pretrained_revision=cfg.policy.revision,
    )
    policy.to(device)
    policy.train()

    weighter = make_weighter(
        cfg, sidecar, chunk_size=plan.chunk_size, device=device
    ) if sidecar is not None else None

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.train.lr if cfg.train.lr is not None else policy_cls_cfg.optimizer_lr,
        betas=tuple(policy_cls_cfg.optimizer_betas),
        eps=policy_cls_cfg.optimizer_eps,
        weight_decay=policy_cls_cfg.optimizer_weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    from ..weighting.loss import torch_weighted_per_sample_loss

    history: list[dict[str, Any]] = []
    step = 0
    done = False
    while not done:
        for batch in loader:
            if step >= cfg.train.steps:
                done = True
                break
            processed = preprocessor(batch)
            if weighter is None:
                loss, loss_dict = policy.forward(processed)
                weight_stats: dict[str, Any] = {"type": "uniform"}
            else:
                # The weighter needs the *raw* identity columns, which the
                # preprocessor may drop or move to device.
                weights, weight_stats = weighter.compute_batch_weights(batch)
                per_sample, loss_dict = policy.forward(processed, reduction="none")
                valid_steps = None
                if processed.get("action_is_pad") is not None:
                    valid_steps = (~processed["action_is_pad"]).sum(dim=1)
                loss, info = torch_weighted_per_sample_loss(
                    per_sample,
                    weights,
                    valid_steps=valid_steps,
                    padding_normalization=cfg.weighting.padding_normalization,
                )
                weight_stats.update(info)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                policy_cls_cfg.optimizer_grad_clip_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % max(1, cfg.train.log_freq) == 0 or step == 1:
                record = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "grad_norm": float(grad_norm),
                    **{
                        k: v
                        for k, v in weight_stats.items()
                        if isinstance(v, (int, float, str))
                    },
                }
                history.append(record)
                print(json.dumps(record))
            if cfg.train.save_freq and step % cfg.train.save_freq == 0:
                _save(policy, preprocessor, postprocessor, out_dir / f"step_{step:08d}")

    final_dir = out_dir / "final"
    _save(policy, preprocessor, postprocessor, final_dir)
    result = {
        "status": "trained",
        "steps": step,
        "output_dir": str(out_dir),
        "checkpoint": str(final_dir),
        "history": history,
        "weighting": weighter.get_stats() if weighter is not None else {"type": "uniform"},
        "plan": plan.to_dict(),
    }
    write_json(out_dir / "train_result.json", result)
    return result


def _make_policy_config(cfg: Config):
    """Build the LeRobot policy config for the configured checkpoint."""
    from lerobot.policies.factory import make_policy_config

    overrides = dict(cfg.policy.overrides)
    if cfg.policy.chunk_size is not None:
        overrides.setdefault("chunk_size", cfg.policy.chunk_size)
    if cfg.policy.n_action_steps is not None:
        overrides.setdefault("n_action_steps", cfg.policy.n_action_steps)
    policy_cfg = make_policy_config(cfg.policy.type, **overrides)
    policy_cfg.pretrained_path = cfg.policy.pretrained_path
    policy_cfg.device = resolve_device(cfg.policy.device)
    policy_cfg.use_amp = cfg.policy.use_amp
    return policy_cfg


def _save(policy, preprocessor, postprocessor, path: Path) -> None:
    ensure_dir(path)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
