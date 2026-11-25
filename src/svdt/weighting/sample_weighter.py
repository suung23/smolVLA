"""Bridge from a score sidecar to LeRobot's per-sample weighting hook.

LeRobot 0.6.1 already defines the integration point we need
(``lerobot.utils.sample_weighting.SampleWeighter``) and its training loop calls
``weighter.compute_batch_weights(batch)`` then computes
``(per_sample_loss * w).sum() / (w.sum() + eps)`` -- exactly the formula this
project specifies. :class:`TrustSampleWeighter` plugs into that hook.

It is duck-typed rather than a hard subclass so it imports with no torch
present; when lerobot *is* importable it registers as a virtual subclass of the
official ABC, so the official factory and training loop accept it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import WeightingConfig
from ..ids import TransitionID, make_keys
from ..scoring.sidecar import ScoreSidecar
from .weights import WeightResult, chunk_weights, compute_weights, weight_report


@dataclass
class BatchWeightStats:
    n: int
    weight_sum: float
    mean_weight: float
    n_missing: int
    n_dropped: int


class TrustSampleWeighter:
    """Per-sample weights from a precomputed score sidecar.

    Resolution order for a batch:

    1. Build each sample's transition key from the batch's own
       ``episode_index``/``frame_index`` columns -- never from the batch's row
       order, and never from the dataset-wide ``index`` column alone.
    2. Look the key up in the sidecar.
    3. Apply the configured chunk-alignment convention and zero out samples
       whose action chunk is entirely padding.

    A key the sidecar does not contain is handled by
    ``WeightingConfig.missing_handling`` and counted; if more than
    ``max_missing_fraction`` of a batch is unresolved the weighter raises, since
    that means the sidecar and the dataset do not belong together.
    """

    def __init__(
        self,
        sidecar: ScoreSidecar,
        cfg: WeightingConfig,
        *,
        dataset_id: str | None = None,
        chunk_size: int | None = None,
        device: Any = None,
        max_missing_fraction: float = 0.05,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.sidecar = sidecar
        self.dataset_id = dataset_id or sidecar.dataset_id
        self.chunk_size = chunk_size
        self.device = device
        self.max_missing_fraction = float(max_missing_fraction)

        # Pre-compute the transition weights once; per-batch work is a lookup.
        self.weights: WeightResult = compute_weights(
            cfg,
            keys=sidecar.keys,
            v=sidecar.v,
            c=sidecar.c,
            q_perception=sidecar.q_perception,
        )
        self._lookup: dict[str, int] = {str(k): i for i, k in enumerate(sidecar.keys)}
        self._w = np.asarray(self.weights.w, dtype=np.float64)

        self._seen = 0
        self._missing = 0
        self._weight_sum = 0.0
        self._batches = 0
        if cfg.chunk_alignment != "anchor" and chunk_size is None:
            raise ValueError(
                f"chunk_alignment={cfg.chunk_alignment!r} needs the policy chunk_size"
            )

    # -- lookup ------------------------------------------------------------ #

    def _weight_for(self, episode_index: int, frame_index: int) -> tuple[float, bool]:
        key = TransitionID(self.dataset_id, int(episode_index), int(frame_index)).key
        pos = self._lookup.get(key)
        if pos is None:
            return self._missing_weight(), True
        return float(self._w[pos]), False

    def _missing_weight(self) -> float:
        if self.cfg.method == "uniform":
            return 1.0
        if self.cfg.missing_handling == "one":
            return 1.0
        if self.cfg.missing_handling == "drop":
            return 0.0
        return 0.0 if self.cfg.method == "hard_filter" else self.cfg.w_min

    def weights_for_keys(self, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        out = np.empty(len(keys), dtype=np.float64)
        missing = np.zeros(len(keys), dtype=bool)
        for i, key in enumerate(keys):
            pos = self._lookup.get(str(key))
            if pos is None:
                out[i] = self._missing_weight()
                missing[i] = True
            else:
                out[i] = self._w[pos]
        return out, missing

    # -- the LeRobot hook -------------------------------------------------- #

    def compute_batch_weights(self, batch: Mapping[str, Any]):
        """``(weights, stats)`` for one training batch.

        Returns a torch tensor when torch is importable (what LeRobot expects)
        and a numpy array otherwise, so the same class is usable in the
        torch-free reference trainer and its tests.
        """
        episode_index, frame_index = _batch_indices(batch)
        n = len(episode_index)
        anchor = np.empty(n, dtype=np.float64)
        missing = np.zeros(n, dtype=bool)
        for i in range(n):
            anchor[i], missing[i] = self._weight_for(episode_index[i], frame_index[i])

        pad_mask = _batch_pad_mask(batch)
        if self.cfg.chunk_alignment == "anchor":
            w = chunk_weights(anchor, alignment="anchor", pad_mask=pad_mask)
        else:
            horizon = (
                pad_mask.shape[1]
                if pad_mask is not None
                else int(self.chunk_size or 1)
            )
            members = np.empty((n, horizon), dtype=np.float64)
            for i in range(n):
                base = int(frame_index[i])
                keys = make_keys(
                    self.dataset_id,
                    np.full(horizon, int(episode_index[i]), dtype=np.int64),
                    np.arange(base, base + horizon, dtype=np.int64),
                )
                members[i], _ = self.weights_for_keys(keys)
            w = chunk_weights(
                anchor,
                alignment=self.cfg.chunk_alignment,
                chunk_member_weights=members,
                pad_mask=pad_mask,
            )

        missing_fraction = float(missing.mean()) if n else 0.0
        if missing_fraction > self.max_missing_fraction:
            sample = [
                TransitionID(self.dataset_id, int(e), int(f)).key
                for e, f, m in zip(episode_index, frame_index, missing)
                if m
            ][:5]
            raise KeyError(
                f"{missing_fraction:.1%} of this batch has no score in sidecar "
                f"{self.sidecar.dataset_id!r} (limit {self.max_missing_fraction:.1%}). "
                f"First unresolved keys: {sample}. The sidecar and the training "
                "dataset are probably not the same variant."
            )

        if self.cfg.normalize_batch_weights and w.sum() > 0:
            # Pure book-keeping: the normalised weighted mean is invariant to a
            # positive rescale. Kept because LeRobot logs raw weight means.
            w = w * (len(w) / w.sum())

        self._seen += n
        self._missing += int(missing.sum())
        self._weight_sum += float(np.asarray(w, dtype=np.float64).sum())
        self._batches += 1

        stats = {
            "mean_weight": round(float(np.mean(w)), 6) if n else 0.0,
            "min_weight": round(float(np.min(w)), 6) if n else 0.0,
            "max_weight": round(float(np.max(w)), 6) if n else 0.0,
            "n_zero_weight": int((np.asarray(w) <= 0).sum()),
            "n_missing_score": int(missing.sum()),
            "type": self.cfg.method,
        }
        return _to_tensor(w, self.device), stats

    def get_stats(self) -> dict[str, Any]:
        """Cumulative statistics, surfaced by LeRobot's logger."""
        report = weight_report(self.weights)
        report.update(
            {
                "batches_seen": self._batches,
                "samples_seen": self._seen,
                "samples_missing_score": self._missing,
                "running_mean_batch_weight": (
                    round(self._weight_sum / self._seen, 6) if self._seen else None
                ),
                "sidecar_dataset_id": self.sidecar.dataset_id,
                "sidecar_fingerprint": self.sidecar.meta.get("fingerprint"),
            }
        )
        return report


def _batch_indices(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-sample ``(episode_index, frame_index)`` from a batch."""
    missing = [k for k in ("episode_index", "frame_index") if k not in batch]
    if missing:
        raise KeyError(
            f"batch is missing {missing}; trust weighting addresses transitions by "
            "(dataset_id, episode_index, frame_index) and cannot fall back to row "
            "order. Keep these columns in the dataloader output."
        )
    ep = _to_numpy(batch["episode_index"]).reshape(-1).astype(np.int64)
    fr = _to_numpy(batch["frame_index"]).reshape(-1).astype(np.int64)
    if ep.shape != fr.shape:
        raise ValueError("episode_index and frame_index have different lengths")
    return ep, fr


def _batch_pad_mask(batch: Mapping[str, Any]) -> np.ndarray | None:
    """LeRobot names the action padding mask ``action_is_pad``; shape (B, H)."""
    for key in ("action_is_pad", "action_pad_mask"):
        if key in batch and batch[key] is not None:
            mask = _to_numpy(batch[key]).astype(bool)
            return mask if mask.ndim == 2 else mask.reshape(mask.shape[0], -1)
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor(array: np.ndarray, device: Any):
    """Return a torch tensor when torch is usable, else a numpy array.

    The availability check goes through the cached probe rather than a bare
    ``import torch`` in the function body: this runs once per training *batch*,
    and on a host where torch's DLLs are blocked a failing import costs tens of
    seconds -- which turned a 12-step smoke test into a seven-minute one.
    """
    from ..env_info import probe_import

    values = np.asarray(array, dtype=np.float32)
    if not probe_import("torch").get("importable"):
        return values
    import torch

    return torch.as_tensor(values, device=device)


def register_with_lerobot() -> bool:
    """Register :class:`TrustSampleWeighter` as a ``SampleWeighter`` subclass.

    Returns False when lerobot is not importable, which is not an error: the
    reference trainer and the unit tests do not need it.
    """
    try:
        from lerobot.utils.sample_weighting import SampleWeighter
    except Exception:
        return False
    SampleWeighter.register(TrustSampleWeighter)
    return True
