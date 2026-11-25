"""Torch-free reference trainer.

Not a replacement for SmolVLA. Its job is to exercise, end to end and on any
machine, the parts of the pipeline that are easy to get wrong:

* the dataloader's transition identity (``episode_index`` / ``frame_index``),
* action-chunk construction with a real padding mask,
* the sample weighter's batch lookup,
* the weighted loss and its normalisation.

The policy is a ridge-regressed linear map from ``(state, 1)`` to the flattened
action chunk, fit by weighted least squares in closed form on each batch's
running statistics. That makes the smoke test deterministic and fast while still
passing gradients of the *same* weighted objective, so a weighting bug shows up
as a wrong loss rather than as noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from ..config import Config
from ..data.backends import FrameSource, LowDimEpisode
from ..env_info import collect
from ..io_utils import ensure_dir, write_json
from ..ids import make_keys
from ..scoring.sidecar import ScoreSidecar
from ..weighting.loss import uniform_chunk_loss, weighted_chunk_loss
from ..weighting.sample_weighter import TrustSampleWeighter


@dataclass
class ChunkBatch:
    """One training batch of action chunks with identity and padding."""

    episode_index: np.ndarray  # (B,)
    frame_index: np.ndarray  # (B,)
    state: np.ndarray  # (B, Ds)
    action: np.ndarray  # (B, H, Da)
    action_is_pad: np.ndarray  # (B, H) bool
    keys: np.ndarray  # (B,) transition keys

    def as_mapping(self) -> dict[str, Any]:
        """The subset of a LeRobot batch the weighter needs."""
        return {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "observation.state": self.state,
            "action": self.action,
            "action_is_pad": self.action_is_pad,
        }


def build_chunks(
    episode: LowDimEpisode, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Action chunks for every anchor in one episode.

    The last ``chunk_size - 1`` anchors run past the episode end; those steps are
    marked padding and the final real action is repeated, which is the
    convention LeRobot's ``delta_timestamps`` produces (``action_is_pad=True``
    with the boundary value held).
    """
    action = np.asarray(episode.action, dtype=np.float64)
    state = np.asarray(episode.state, dtype=np.float64)
    t, dim = action.shape
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    chunks = np.empty((t, chunk_size, dim), dtype=np.float64)
    pad = np.zeros((t, chunk_size), dtype=bool)
    for h in range(chunk_size):
        idx = np.minimum(np.arange(t) + h, t - 1)
        chunks[:, h] = action[idx]
        pad[:, h] = (np.arange(t) + h) > (t - 1)
    keys = make_keys(
        episode.dataset_id,
        np.full(t, episode.episode_index, dtype=np.int64),
        episode.frame_index,
    )
    return state, chunks, pad, episode.frame_index, keys


def iter_batches(
    episodes: Sequence[LowDimEpisode],
    *,
    chunk_size: int,
    batch_size: int,
    seed: int,
    steps: int,
) -> Iterator[ChunkBatch]:
    """Shuffled batches of chunks drawn from the given episodes."""
    parts = [build_chunks(ep, chunk_size) for ep in episodes]
    state = np.concatenate([p[0] for p in parts])
    chunks = np.concatenate([p[1] for p in parts])
    pad = np.concatenate([p[2] for p in parts])
    frame_index = np.concatenate([p[3] for p in parts])
    keys = np.concatenate([p[4] for p in parts])
    episode_index = np.concatenate(
        [np.full(len(p[3]), ep.episode_index, dtype=np.int64) for p, ep in zip(parts, episodes)]
    )
    n = len(state)
    if n == 0:
        raise ValueError("no transitions to train on")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > n:
            order = rng.permutation(n)
            cursor = 0
        sel = order[cursor : cursor + batch_size]
        cursor += batch_size
        yield ChunkBatch(
            episode_index=episode_index[sel],
            frame_index=frame_index[sel],
            state=state[sel],
            action=chunks[sel],
            action_is_pad=pad[sel],
            keys=keys[sel],
        )


class ReferenceTrainer:
    """Weighted ridge regression on action chunks. Deterministic, numpy only."""

    def __init__(
        self,
        state_dim: int,
        chunk_size: int,
        action_dim: int,
        *,
        ridge: float = 1e-3,
        lr: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.ridge = ridge
        self.lr = lr
        rng = np.random.default_rng(seed)
        # Weights map [state, 1] -> flattened chunk.
        self.W = rng.normal(0.0, 0.01, size=(state_dim + 1, chunk_size * action_dim))

    def features(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        return np.concatenate([s, np.ones((s.shape[0], 1))], axis=1)

    def predict(self, state: np.ndarray) -> np.ndarray:
        flat = self.features(state) @ self.W
        return flat.reshape(-1, self.chunk_size, self.action_dim)

    def step(
        self,
        batch: ChunkBatch,
        weights: np.ndarray | None,
        *,
        padding_normalization: str = "per_sample",
    ) -> dict[str, Any]:
        """One weighted least-squares update towards the batch's targets."""
        x = self.features(batch.state)
        y = batch.action.reshape(len(batch.action), -1)
        valid = (~batch.action_is_pad).astype(np.float64)
        # Row weight: the sample weight, zeroed where the chunk is all padding.
        w = (
            np.ones(len(x))
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        w = np.where(valid.sum(axis=1) > 0, w, 0.0)

        # Closed-form weighted ridge solution, blended in at `lr` so the loss
        # trajectory is informative rather than a single jump.
        xw = x * w[:, None]
        gram = x.T @ xw + self.ridge * np.eye(x.shape[1])
        target = xw.T @ y
        try:
            solution = np.linalg.solve(gram, target)
        except np.linalg.LinAlgError:  # pragma: no cover - singular batch
            solution = np.linalg.lstsq(gram, target, rcond=None)[0]
        self.W = (1 - self.lr) * self.W + self.lr * solution

        pred = self.predict(batch.state)
        if weights is None:
            breakdown = uniform_chunk_loss(pred, batch.action, batch.action_is_pad)
        else:
            breakdown = weighted_chunk_loss(
                pred,
                batch.action,
                w,
                batch.action_is_pad,
                padding_normalization=padding_normalization,
            )
        return breakdown.as_dict()


def run_reference_training(
    cfg: Config,
    source: FrameSource,
    episodes: Sequence[int],
    *,
    sidecar: ScoreSidecar | None = None,
    chunk_size: int = 8,
    steps: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the torch-free trainer and record the result.

    The uniform condition deliberately bypasses the weighter entirely, mirroring
    what :mod:`svdt.train.smolvla` does, so the smoke test checks the same
    branch the real run takes.
    """
    low = [source.read_lowdim(int(e)) for e in episodes]
    if not low:
        raise ValueError("no episodes selected for training")
    steps = int(steps if steps is not None else cfg.train.steps)

    weighter: TrustSampleWeighter | None = None
    if cfg.weighting.method != "uniform":
        if sidecar is None:
            raise ValueError(
                f"weighting method {cfg.weighting.method!r} needs a score sidecar; "
                "run `svdt score` first"
            )
        weighter = TrustSampleWeighter(
            sidecar,
            cfg.weighting,
            dataset_id=low[0].dataset_id,
            chunk_size=chunk_size,
            max_missing_fraction=1.0,
        )

    trainer = ReferenceTrainer(
        state_dim=low[0].state.shape[1],
        chunk_size=chunk_size,
        action_dim=low[0].action.shape[1],
        seed=cfg.train.seed,
    )
    history: list[dict[str, Any]] = []
    weight_trace: list[float] = []
    for i, batch in enumerate(
        iter_batches(
            low,
            chunk_size=chunk_size,
            batch_size=cfg.train.batch_size,
            seed=cfg.train.seed,
            steps=steps,
        ),
        start=1,
    ):
        weights = None
        if weighter is not None:
            weights, _ = weighter.compute_batch_weights(batch.as_mapping())
            weights = np.asarray(weights, dtype=np.float64)
            weight_trace.append(float(weights.mean()))
        record = trainer.step(
            batch,
            weights,
            padding_normalization=cfg.weighting.padding_normalization,
        )
        record["step"] = i
        if i == 1 or i % max(1, cfg.train.log_freq) == 0 or i == steps:
            history.append(record)

    result: dict[str, Any] = {
        "status": "trained",
        "backend": "reference",
        "steps": steps,
        "chunk_size": chunk_size,
        "weighting_method": cfg.weighting.method,
        "n_episodes": len(low),
        "n_transitions": int(sum(ep.num_frames for ep in low)),
        "final_loss": history[-1]["loss"] if history else None,
        "first_loss": history[0]["loss"] if history else None,
        "history": history,
        "mean_batch_weight": (
            round(float(np.mean(weight_trace)), 6) if weight_trace else 1.0
        ),
        "weighting_stats": weighter.get_stats() if weighter is not None else {"type": "uniform"},
        "environment": collect(command="train(reference)", seed=cfg.train.seed),
    }
    out = Path(output_dir or cfg.train.output_dir)
    ensure_dir(out)
    write_json(out / f"reference_train__{cfg.weighting.method}.json", result)
    result["output_dir"] = str(out)
    return result
