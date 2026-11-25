r"""Weighted action loss with explicit chunk/padding alignment.

.. math::

   L_{\text{weighted}} = \frac{\sum_i w_i L_{\text{action},i}}{\sum_i w_i}

Three properties this module exists to guarantee, each covered by a test:

1. **Padding is excluded twice over.** A padded action step contributes to
   neither the per-sample loss nor its denominator, and a sample whose chunk is
   entirely padding contributes to neither the numerator nor the weight sum of
   the outer average.
2. **Uniform weighting reproduces the unweighted loss exactly**, not
   approximately -- with all ``w_i = 1`` the expression above is the plain mean
   over valid elements.
3. **The numpy reference and the torch implementation agree.** The numpy version
   runs with no torch installed, which is what makes the loss testable on a
   machine where torch cannot be imported.

LeRobot 0.6.1's own training loop computes
``(per_sample_loss * w).sum() / (w.sum() + eps)`` from
``policy.forward(batch, reduction="none")``; :func:`weighted_per_sample_loss`
matches that so the two paths cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Guard against a zero weight sum (every sample dropped or fully padded).
EPS = 1e-6

#: How the weight-sum denominator is protected.
#:
#: ``guard`` (default) divides by ``max(sum(w), eps)``. The denominator is then
#: *exactly* ``sum(w)`` whenever anything contributes, which is what lets the
#: uniform control reproduce the unweighted baseline bit for bit.
#:
#: ``lerobot`` divides by ``sum(w) + eps``, reproducing lerobot 0.6.1's own
#: weighted step exactly. The two differ by a relative ``eps/sum(w)`` (~1e-7 at
#: batch size 8), which is irrelevant for gradients but visible in an equality
#: test -- hence both modes exist rather than one silent compromise.
EPS_MODES = ("guard", "lerobot")


def _denominator(weight_sum: float, eps: float, eps_mode: str) -> float:
    if eps_mode == "guard":
        return max(weight_sum, eps)
    if eps_mode == "lerobot":
        return weight_sum + eps
    raise ValueError(f"unknown eps_mode {eps_mode!r}; expected one of {EPS_MODES}")


@dataclass
class LossBreakdown:
    """Scalar loss plus the accounting needed to verify what was averaged."""

    loss: float
    per_sample_loss: np.ndarray
    sample_weights: np.ndarray
    #: Valid (non-padding) action steps per sample.
    valid_steps: np.ndarray
    weight_sum: float
    n_samples: int
    n_contributing: int
    n_valid_elements: int
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "weight_sum": self.weight_sum,
            "n_samples": self.n_samples,
            "n_contributing_samples": self.n_contributing,
            "n_valid_action_elements": self.n_valid_elements,
            "mean_weight": (
                round(float(np.mean(self.sample_weights)), 6) if self.n_samples else None
            ),
            **self.meta,
        }


def _check_shapes(
    pred: np.ndarray, target: np.ndarray, pad_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and target {t.shape} must match")
    if p.ndim != 3:
        raise ValueError(f"expected (B, H, A) action chunks, got {p.shape}")
    if pad_mask is None:
        valid = np.ones(p.shape[:2], dtype=bool)
    else:
        m = np.asarray(pad_mask, dtype=bool)
        if m.shape != p.shape[:2]:
            raise ValueError(
                f"pad_mask must be (B, H) = {p.shape[:2]}, got {m.shape}"
            )
        valid = ~m
    return p, t, valid


def per_element_loss(
    pred: np.ndarray, target: np.ndarray, *, kind: str = "l2"
) -> np.ndarray:
    """Element-wise action loss, shape (B, H, A).

    ``l2`` is the squared error SmolVLA's flow-matching objective reduces to;
    ``l1`` is offered for ablations.
    """
    if kind == "l2":
        return (pred - target) ** 2
    if kind == "l1":
        return np.abs(pred - target)
    raise ValueError(f"unknown loss kind {kind!r}; expected 'l2' or 'l1'")


def per_sample_losses(
    pred: np.ndarray,
    target: np.ndarray,
    pad_mask: np.ndarray | None = None,
    *,
    kind: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    """``(per_sample_loss, valid_steps)``: mean over valid (step, dim) entries.

    Mirrors LeRobot's ``reduction="none"`` branch, which divides each sample by
    ``(~action_is_pad).sum(dim=1) * action_dim`` -- so padding never dilutes a
    sample's loss towards zero.
    """
    p, t, valid = _check_shapes(pred, target, pad_mask)
    elementwise = per_element_loss(p, t, kind=kind)
    masked = elementwise * valid[..., None]
    action_dim = p.shape[2]
    valid_steps = valid.sum(axis=1)
    denom = np.maximum(valid_steps * action_dim, 1)
    return masked.sum(axis=(1, 2)) / denom, valid_steps


def uniform_chunk_loss(
    pred: np.ndarray,
    target: np.ndarray,
    pad_mask: np.ndarray | None = None,
    *,
    kind: str = "l2",
) -> LossBreakdown:
    """Unweighted baseline: plain mean over every valid action element.

    This is the reference that ``weighted_chunk_loss(..., weights=1)`` must
    reproduce exactly.
    """
    p, t, valid = _check_shapes(pred, target, pad_mask)
    elementwise = per_element_loss(p, t, kind=kind)
    masked = elementwise * valid[..., None]
    n_valid = int(valid.sum() * p.shape[2])
    loss = float(masked.sum() / max(n_valid, 1))
    per_sample, valid_steps = per_sample_losses(p, t, pad_mask, kind=kind)
    return LossBreakdown(
        loss=loss,
        per_sample_loss=per_sample,
        sample_weights=np.ones(p.shape[0], dtype=np.float64),
        valid_steps=valid_steps,
        weight_sum=float(p.shape[0]),
        n_samples=int(p.shape[0]),
        n_contributing=int((valid_steps > 0).sum()),
        n_valid_elements=n_valid,
        meta={"mode": "uniform", "loss_kind": kind},
    )


def effective_sample_weights(
    weights: np.ndarray,
    valid_steps: np.ndarray | None,
    *,
    padding_normalization: str = "per_sample",
) -> np.ndarray:
    r"""Weights as they enter the outer average.

    ``per_sample`` (default) zeroes fully padded samples and otherwise leaves
    the weights alone. This is what LeRobot's weighted path does.

    ``per_element`` additionally multiplies by the sample's valid-step count.
    That matters because the two reductions are *not* the same function:

    * LeRobot's unweighted ``reduction="mean"`` is element-weighted,
      :math:`\sum_i S_i / (A \sum_i n_i)`;
    * its weighted path is sample-weighted,
      :math:`\sum_i w_i L_i / \sum_i w_i` with :math:`L_i = S_i / (A n_i)`.

    With unequal ``n_i`` these differ, so "weighted with all weights 1" would
    *not* reproduce the baseline. Scaling by :math:`n_i` makes it reproduce it
    exactly, which is the honest way to run a uniform control through the
    weighted code path.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.any(w < 0):
        raise ValueError("sample weights must be non-negative")
    if valid_steps is None:
        if padding_normalization == "per_element":
            raise ValueError("padding_normalization='per_element' needs valid_steps")
        return w
    vs = np.asarray(valid_steps, dtype=np.float64).reshape(-1)
    if vs.shape != w.shape:
        raise ValueError(
            f"valid_steps {vs.shape} must match the sample weights {w.shape}"
        )
    w = np.where(vs > 0, w, 0.0)
    if padding_normalization == "per_element":
        return w * vs
    if padding_normalization != "per_sample":
        raise ValueError(f"unknown padding_normalization {padding_normalization!r}")
    return w


def weighted_per_sample_loss(
    per_sample_loss: np.ndarray,
    weights: np.ndarray,
    *,
    valid_steps: np.ndarray | None = None,
    eps: float = EPS,
    padding_normalization: str = "per_sample",
    eps_mode: str = "guard",
) -> tuple[float, dict[str, Any]]:
    """The normalised weighted mean, given already-reduced per-sample losses.

    Samples with no valid action step are removed from numerator *and*
    denominator: a fully padded sample has no supervision to contribute, so
    letting its weight sit in the denominator would silently shrink the loss.
    """
    losses = np.asarray(per_sample_loss, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != losses.shape:
        raise ValueError(f"weights {w.shape} must match per-sample losses {losses.shape}")
    eff = effective_sample_weights(
        w, valid_steps, padding_normalization=padding_normalization
    )
    total = float(eff.sum())
    denom = _denominator(total, eps, eps_mode)
    loss = float((losses * eff).sum() / denom)
    return loss, {
        "weight_sum": total,
        "raw_weight_sum": float(np.where(eff > 0, w, 0.0).sum()),
        "n_contributing_samples": int((eff > 0).sum()),
        "padding_normalization": padding_normalization,
        "eps": eps,
        "eps_mode": eps_mode,
    }


def weighted_chunk_loss(
    pred: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    pad_mask: np.ndarray | None = None,
    *,
    kind: str = "l2",
    eps: float = EPS,
    padding_normalization: str = "per_sample",
    eps_mode: str = "guard",
) -> LossBreakdown:
    """Weighted action-chunk loss, the numpy reference implementation."""
    p, t, valid = _check_shapes(pred, target, pad_mask)
    per_sample, valid_steps = per_sample_losses(p, t, pad_mask, kind=kind)
    loss, info = weighted_per_sample_loss(
        per_sample,
        weights,
        valid_steps=valid_steps,
        eps=eps,
        padding_normalization=padding_normalization,
        eps_mode=eps_mode,
    )
    eff = effective_sample_weights(
        weights, valid_steps, padding_normalization=padding_normalization
    )
    return LossBreakdown(
        loss=loss,
        per_sample_loss=per_sample,
        sample_weights=eff,
        valid_steps=valid_steps,
        weight_sum=info["weight_sum"],
        n_samples=int(p.shape[0]),
        n_contributing=info["n_contributing_samples"],
        n_valid_elements=int(valid.sum() * p.shape[2]),
        meta={
            "mode": "weighted",
            "loss_kind": kind,
            "eps": eps,
            "eps_mode": eps_mode,
            "padding_normalization": padding_normalization,
        },
    )


# --------------------------------------------------------------------------- #
# torch mirror
# --------------------------------------------------------------------------- #


def torch_weighted_per_sample_loss(
    per_sample_loss,
    weights,
    *,
    valid_steps=None,
    eps: float = EPS,
    padding_normalization: str = "per_sample",
    eps_mode: str = "guard",
):
    """Differentiable twin of :func:`weighted_per_sample_loss`.

    Written to be numerically identical to the numpy reference *and*, with
    ``padding_normalization="per_sample"``, to LeRobot's own
    ``(per_sample_loss * w).sum() / (w.sum() + eps)`` -- so the baseline path and
    the weighted path cannot silently diverge.
    """
    import torch

    losses = per_sample_loss.reshape(-1)
    w = weights.reshape(-1).to(dtype=losses.dtype, device=losses.device)
    if w.shape != losses.shape:
        raise ValueError(f"weights {tuple(w.shape)} must match losses {tuple(losses.shape)}")
    if bool((w < 0).any()):
        raise ValueError("sample weights must be non-negative")
    if valid_steps is not None:
        vs = valid_steps.reshape(-1).to(device=losses.device)
        w = torch.where(vs > 0, w, torch.zeros_like(w))
        if padding_normalization == "per_element":
            w = w * vs.to(dtype=w.dtype)
        elif padding_normalization != "per_sample":
            raise ValueError(f"unknown padding_normalization {padding_normalization!r}")
    elif padding_normalization == "per_element":
        raise ValueError("padding_normalization='per_element' needs valid_steps")
    total = w.sum()
    if eps_mode == "guard":
        denom = torch.clamp(total, min=eps)
    elif eps_mode == "lerobot":
        denom = total + eps
    else:
        raise ValueError(f"unknown eps_mode {eps_mode!r}; expected one of {EPS_MODES}")
    return (losses * w).sum() / denom, {
        "weight_sum": float(total.detach().item()),
        "n_contributing_samples": int((w > 0).sum().item()),
        "padding_normalization": padding_normalization,
        "eps": eps,
        "eps_mode": eps_mode,
    }


def torch_weighted_chunk_loss(
    pred,
    target,
    weights,
    pad_mask=None,
    *,
    kind: str = "l2",
    eps: float = EPS,
    padding_normalization: str = "per_sample",
    eps_mode: str = "guard",
):
    """Torch counterpart of :func:`weighted_chunk_loss`."""
    import torch

    if pred.shape != target.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match")
    if pred.dim() != 3:
        raise ValueError(f"expected (B, H, A) action chunks, got {tuple(pred.shape)}")
    if kind == "l2":
        elementwise = (pred - target) ** 2
    elif kind == "l1":
        elementwise = (pred - target).abs()
    else:
        raise ValueError(f"unknown loss kind {kind!r}")

    if pad_mask is None:
        valid = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    else:
        valid = ~pad_mask.to(dtype=torch.bool)
    masked = elementwise * valid.unsqueeze(-1)
    action_dim = pred.shape[2]
    valid_steps = valid.sum(dim=1)
    denom = (valid_steps * action_dim).clamp_min(1)
    per_sample = masked.sum(dim=(1, 2)) / denom
    loss, info = torch_weighted_per_sample_loss(
        per_sample,
        weights,
        valid_steps=valid_steps,
        eps=eps,
        padding_normalization=padding_normalization,
        eps_mode=eps_mode,
    )
    info["per_sample_loss"] = per_sample
    info["valid_steps"] = valid_steps
    return loss, info


def torch_uniform_chunk_loss(pred, target, pad_mask=None, *, kind: str = "l2"):
    """Torch counterpart of :func:`uniform_chunk_loss`."""
    import torch

    if kind == "l2":
        elementwise = (pred - target) ** 2
    elif kind == "l1":
        elementwise = (pred - target).abs()
    else:
        raise ValueError(f"unknown loss kind {kind!r}")
    if pad_mask is None:
        valid = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    else:
        valid = ~pad_mask.to(dtype=torch.bool)
    masked = elementwise * valid.unsqueeze(-1)
    n_valid = (valid.sum() * pred.shape[2]).clamp_min(1)
    return masked.sum() / n_valid
