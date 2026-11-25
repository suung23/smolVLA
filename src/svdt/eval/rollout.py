"""Optional rollout evaluation, with an explicit refusal when it is not valid.

The project description is specific about this, and so is this module: rollout
success rate is reported **only** when a verified simulator matching the dataset
exists. Otherwise the metric is recorded as ``not_available`` with the reason.

State of the registry, checked against the upstream packages:

``lerobot/aloha_mobile_cabinet``
    ``gym-aloha`` registers only ``AlohaInsertion-v0`` and
    ``AlohaTransferCube-v0``. Neither is the mobile-cabinet task (open the top
    cabinet, store the pot, close it), and the dataset is a real-robot Mobile
    ALOHA recording with no released simulator. So rollout success rate for this
    dataset is ``not_available`` -- offline action metrics come first, and no
    real-robot evaluation is part of any default path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

#: Datasets with a verified simulator, and the env id to use.
VERIFIED_ENVIRONMENTS: dict[str, dict[str, Any]] = {}

#: Datasets explicitly known to have no validated simulator, with the reason.
KNOWN_UNAVAILABLE: dict[str, str] = {
    "lerobot/aloha_mobile_cabinet": (
        "gym-aloha registers only AlohaInsertion-v0 and AlohaTransferCube-v0; "
        "neither matches the mobile-cabinet task, and this dataset is a "
        "real-robot Mobile ALOHA recording with no released simulator"
    ),
}


@dataclass
class RolloutSupport:
    """Whether rollout evaluation is valid and runnable for a dataset."""

    status: str  # "available" | "not_available" | "missing_dependency"
    dataset: str
    env_id: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def runnable(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_success_rate_status": self.status,
            "dataset": self.dataset,
            "env_id": self.env_id,
            "reason": self.reason,
            **self.details,
        }


def check_rollout_support(
    repo_id: str, *, env_id: str | None = None
) -> RolloutSupport:
    """Decide whether a rollout number may be produced at all."""
    if repo_id in KNOWN_UNAVAILABLE and env_id is None:
        return RolloutSupport(
            status="not_available",
            dataset=repo_id,
            reason=KNOWN_UNAVAILABLE[repo_id],
            details={
                "guidance": (
                    "report offline action metrics (action MAE, action chunk "
                    "error) for this dataset; do not claim a rollout success rate"
                )
            },
        )

    entry = VERIFIED_ENVIRONMENTS.get(repo_id)
    resolved_env = env_id or (entry or {}).get("env_id")
    if resolved_env is None:
        return RolloutSupport(
            status="not_available",
            dataset=repo_id,
            reason=(
                f"no verified simulator is registered for {repo_id!r}. Pass "
                "eval.rollout_env_id explicitly only if you have confirmed the "
                "environment matches this dataset's task and action space."
            ),
        )

    package = (entry or {}).get("package", resolved_env.split("/")[0])
    try:
        import gymnasium  # noqa: F401
    except Exception as exc:
        return RolloutSupport(
            status="missing_dependency",
            dataset=repo_id,
            env_id=resolved_env,
            reason=f"gymnasium is not installed ({exc})",
        )
    try:
        __import__(package)
    except Exception as exc:
        return RolloutSupport(
            status="missing_dependency",
            dataset=repo_id,
            env_id=resolved_env,
            reason=(
                f"{package} is not installed ({exc}); "
                f"install the simulator package for {repo_id!r}"
            ),
        )
    return RolloutSupport(
        status="available",
        dataset=repo_id,
        env_id=resolved_env,
        details={k: v for k, v in (entry or {}).items() if k not in {"env_id", "package"}},
    )


def run_rollout(
    repo_id: str,
    policy_fn: Any,
    *,
    env_id: str | None = None,
    episodes: int = 20,
    seed: int = 0,
    max_steps: int | None = None,
    obs_type: str | None = None,
) -> dict[str, Any]:
    """Run closed-loop episodes and report the success rate.

    ``policy_fn(observation) -> action`` keeps this decoupled from any specific
    policy class, so the same adapter serves a SmolVLA policy and a scripted
    control used to sanity-check the harness.

    Returns a ``not_available`` record rather than raising when the dataset has
    no validated environment: a missing metric must be visible in the report,
    not an exception that hides the whole evaluation.
    """
    support = check_rollout_support(repo_id, env_id=env_id)
    if not support.runnable:
        return {
            "rollout_success_rate": None,
            **support.to_dict(),
        }

    import gymnasium as gym

    kwargs: dict[str, Any] = {}
    resolved_obs = obs_type or support.details.get("obs_type")
    if resolved_obs:
        kwargs["obs_type"] = resolved_obs
    env = gym.make(support.env_id, **kwargs)

    successes: list[bool] = []
    rewards: list[float] = []
    lengths: list[int] = []
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        steps = 0
        success = bool(info.get("is_success", False))
        limit = max_steps or getattr(env.spec, "max_episode_steps", None) or 300
        while not done and steps < limit:
            action = policy_fn(observation)
            observation, reward, terminated, truncated, info = env.step(
                np.asarray(action).reshape(-1)
            )
            total += float(reward)
            steps += 1
            success = success or bool(info.get("is_success", False))
            done = bool(terminated or truncated)
        successes.append(success)
        rewards.append(total)
        lengths.append(steps)
    env.close()

    return {
        "rollout_success_rate": round(float(np.mean(successes)), 6),
        "rollout_success_rate_status": "available",
        "dataset": repo_id,
        "env_id": support.env_id,
        "episodes": episodes,
        "seed": seed,
        "mean_return": round(float(np.mean(rewards)), 6),
        "mean_episode_length": round(float(np.mean(lengths)), 3),
        "successes": int(np.sum(successes)),
    }


def rollout_placeholder(repo_id: str, *, env_id: str | None = None) -> dict[str, Any]:
    """The record an evaluation writes when rollout is skipped or impossible."""
    support = check_rollout_support(repo_id, env_id=env_id)
    return {"rollout_success_rate": None, **support.to_dict()}


def summarize_support(datasets: Mapping[str, str | None]) -> dict[str, Any]:
    """Support table for the README / manifest, one row per dataset."""
    return {
        repo_id: check_rollout_support(repo_id, env_id=env_id).to_dict()
        for repo_id, env_id in datasets.items()
    }
