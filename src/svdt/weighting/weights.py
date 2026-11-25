r"""Transition weights.

.. math::

   w_t = \operatorname{clip}\bigl(\alpha v_t + (1-\alpha) c_t,\; w_{\min},\; 1\bigr)

Defaults ``alpha=0.5``, ``w_min=0.15``. Both are *initial* values and may only
be tuned on validation episodes -- :func:`svdt.data.splits.tuning_episodes` is
what a tuning script is expected to ask for.

The five training conditions all come out of this one function:

=====================  ==========================================
``method``             effect
=====================  ==========================================
``uniform``            every weight is 1 (identical to the
                       unweighted baseline, bit for bit)
``visibility_only``    alpha = 1
``reliability_only``   alpha = 0
``combined``           alpha as configured
``hard_filter``        weight 1 above the threshold, 0 below
=====================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import WeightingConfig


@dataclass
class WeightResult:
    """Weights plus everything needed to explain them."""

    keys: np.ndarray
    w: np.ndarray
    #: Pre-clip trust score ``alpha*v + (1-alpha)*c``.
    trust: np.ndarray
    #: True where the transition had no usable score and the fallback applied.
    missing: np.ndarray
    #: True where the transition is excluded from training entirely.
    dropped: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)

    def as_mapping(self) -> dict[str, float]:
        return {str(k): float(w) for k, w in zip(self.keys, self.w)}


def compute_weights(
    cfg: WeightingConfig,
    *,
    keys: np.ndarray,
    v: np.ndarray | None = None,
    c: np.ndarray | None = None,
    q_perception: np.ndarray | None = None,
) -> WeightResult:
    """Apply the weighting formula to aligned score arrays.

    ``v``/``c`` may contain NaN for transitions a scorer could not evaluate.
    Those are resolved by ``cfg.missing_handling`` -- ``w_min`` (default, keep
    the sample at the floor), ``one`` (treat as fully trusted) or ``drop``
    (exclude it) -- and are always reported, never hidden.
    """
    cfg.validate()
    keys = np.asarray(keys, dtype=object)
    n = len(keys)
    alpha = cfg.effective_alpha()
    coefficients: dict[str, float] = {
        "visibility_only": {"v": 1.0},
        "reliability_only": {"c": 1.0},
        "perception_only": {"q_perception": 1.0},
        "combined": {"v": alpha, "c": 1.0 - alpha},
        "hard_filter": {"v": alpha, "c": 1.0 - alpha},
        "visibility_perception": {"v": cfg.beta_visibility, "q_perception": cfg.beta_perception},
        "perception_reliability": {"q_perception": cfg.beta_perception, "c": cfg.beta_reliability},
        "combined_three_way": {
            "v": cfg.beta_visibility,
            "q_perception": cfg.beta_perception,
            "c": cfg.beta_reliability,
        },
    }.get(cfg.method, {})
    coefficients = {name: float(w) for name, w in coefficients.items() if w > 0}
    if cfg.method != "uniform" and not coefficients:
        raise ValueError(f"weighting method {cfg.method!r} has no positive input coefficients")
    values_in = {"v": v, "c": c, "q_perception": q_perception}

    def prepare(arr: np.ndarray | None, name: str) -> tuple[np.ndarray, np.ndarray]:
        if arr is None:
            return np.zeros(n), np.zeros(n, dtype=bool)
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.size != n:
            raise ValueError(f"{name} has {a.size} values but there are {n} keys")
        ok = np.isfinite(a)
        if np.any(ok & ((a < -1e-6) | (a > 1 + 1e-6))):
            raise ValueError(f"{name} must lie in [0,1]")
        return np.where(ok, np.clip(a, 0.0, 1.0), 0.0), ok

    v_arr, v_ok = prepare(v, "v")
    c_arr, c_ok = prepare(c, "c")
    q_arr, q_ok = prepare(q_perception, "q_perception")

    if cfg.method == "uniform":
        # Bit-for-bit identical to unweighted training: every sample weighs 1,
        # so the normalised weighted mean reduces to the plain mean.
        return WeightResult(
            keys=keys,
            w=np.ones(n, dtype=np.float32),
            trust=np.ones(n, dtype=np.float32),
            missing=np.zeros(n, dtype=bool),
            dropped=np.zeros(n, dtype=bool),
            meta={
                "method": "uniform",
                "alpha": None,
                "w_min": None,
                "note": "uniform weighting is exactly the unweighted baseline",
            },
        )

    arrays = {"v": v_arr, "c": c_arr, "q_perception": q_arr}
    availability = {"v": v_ok, "c": c_ok, "q_perception": q_ok}
    missing_inputs = []
    for name in coefficients:
        if values_in[name] is None:
            missing_inputs.append(name)
    if missing_inputs:
        raise ValueError(
            f"weighting method {cfg.method!r} needs scores: {', '.join(missing_inputs)}"
        )
    required = list(coefficients)
    all_present = np.logical_and.reduce([availability[name] for name in required])
    any_present = np.logical_or.reduce([availability[name] for name in required])
    missing = ~all_present
    partial = missing & any_present
    numerator = np.zeros(n, dtype=np.float64)
    denominator = np.zeros(n, dtype=np.float64)
    for name, coefficient in coefficients.items():
        present = availability[name]
        numerator += coefficient * arrays[name] * present
        denominator += coefficient * present
    with np.errstate(invalid="ignore", divide="ignore"):
        trust = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), 0.0)
    if cfg.missing_handling != "renormalize":
        trust = numerator / max(sum(coefficients.values()), 1e-12)
    else:
        missing = missing & ~partial
    dropped = np.zeros(n, dtype=bool)

    if cfg.method == "hard_filter":
        keep = trust >= cfg.hard_filter_threshold
        w = np.where(keep, 1.0, 0.0)
        dropped = ~keep
    else:
        w = np.clip(trust, cfg.w_min, cfg.w_max)

    # Resolve transitions that still have no usable score at all.
    if missing.any():
        if cfg.missing_handling in {"w_min", "renormalize"}:
            w = np.where(missing, cfg.w_min if cfg.method != "hard_filter" else 0.0, w)
            if cfg.method == "hard_filter":
                dropped = dropped | missing
        elif cfg.missing_handling == "one":
            w = np.where(missing, 1.0, w)
            dropped = dropped & ~missing
        else:  # drop
            w = np.where(missing, 0.0, w)
            dropped = dropped | missing
        trust = np.where(missing, np.nan, trust)

    return WeightResult(
        keys=keys,
        w=w.astype(np.float32),
        trust=trust.astype(np.float32),
        missing=missing,
        dropped=dropped,
        meta={
            "method": cfg.method,
            "alpha": alpha if cfg.method in {"combined", "hard_filter", "visibility_only", "reliability_only"} else None,
            "alpha_configured": cfg.alpha,
            "w_min": cfg.w_min,
            "w_max": cfg.w_max,
            "hard_filter_threshold": (
                cfg.hard_filter_threshold if cfg.method == "hard_filter" else None
            ),
            "missing_handling": cfg.missing_handling,
            "n_partial_scores_renormalized": int(partial.sum()),
            "chunk_alignment": cfg.chunk_alignment,
            "uses_visibility": "v" in coefficients,
            "uses_reliability": "c" in coefficients,
            "uses_perception": "q_perception" in coefficients,
            "coefficients": coefficients,
        },
    )


def hard_filter_mask(result: WeightResult) -> np.ndarray:
    """Transitions a hard-filter run keeps."""
    return ~np.asarray(result.dropped, dtype=bool)


def chunk_weights(
    anchor_weights: np.ndarray,
    *,
    alignment: str,
    chunk_member_weights: np.ndarray | None = None,
    pad_mask: np.ndarray | None = None,
) -> np.ndarray:
    r"""Map per-transition weights onto SmolVLA action-chunk samples.

    A SmolVLA training sample anchored at ``t`` supervises the chunk
    ``a_t ... a_{t+H-1}``, so one sample has one weight but ``H`` scored
    transitions behind it. Three documented conventions:

    ``anchor`` (default)
        The sample inherits ``w_t`` of its anchor transition. The anchor is the
        transition whose observation the policy actually conditions on, so this
        is the convention that matches "weight the transition".

    ``chunk_mean`` / ``chunk_min``
        Aggregate ``w`` over the chunk's **non-padding** steps. ``chunk_min`` is
        the conservative choice: one untrustworthy future action drags the whole
        chunk down.

    Padding steps are excluded from the aggregation, and a sample whose chunk is
    entirely padding gets weight 0 so it leaves both numerator and denominator.

    Args:
        anchor_weights: ``(B,)`` weight of each sample's anchor transition.
        alignment: ``anchor`` | ``chunk_mean`` | ``chunk_min``.
        chunk_member_weights: ``(B, H)`` weight of every transition in the
            chunk; required for the aggregate conventions.
        pad_mask: ``(B, H)`` ``True`` where the step is padding.
    """
    anchor = np.asarray(anchor_weights, dtype=np.float64).reshape(-1)
    if alignment == "anchor":
        if pad_mask is None:
            return anchor.astype(np.float32)
        valid = ~np.asarray(pad_mask, dtype=bool)
        all_pad = ~valid.any(axis=1)
        return np.where(all_pad, 0.0, anchor).astype(np.float32)

    if chunk_member_weights is None:
        raise ValueError(
            f"chunk_alignment={alignment!r} needs chunk_member_weights of shape (B, H)"
        )
    members = np.asarray(chunk_member_weights, dtype=np.float64)
    if members.ndim != 2 or members.shape[0] != anchor.size:
        raise ValueError(
            f"chunk_member_weights must be (B, H) with B={anchor.size}, got {members.shape}"
        )
    if pad_mask is None:
        valid = np.ones_like(members, dtype=bool)
    else:
        valid = ~np.asarray(pad_mask, dtype=bool)
        if valid.shape != members.shape:
            raise ValueError(
                f"pad_mask shape {valid.shape} does not match chunk weights {members.shape}"
            )
    counts = valid.sum(axis=1)
    all_pad = counts == 0

    if alignment == "chunk_mean":
        total = np.where(valid, members, 0.0).sum(axis=1)
        # `where=` without `out=` leaves the skipped entries uninitialised, so
        # the all-padding rows are given an explicit 0 before the divide.
        out = np.zeros_like(total)
        np.divide(total, np.maximum(counts, 1), out=out, where=counts > 0)
    elif alignment == "chunk_min":
        filled = np.where(valid, members, np.inf)
        out = filled.min(axis=1)
        out = np.where(np.isfinite(out), out, 0.0)
    else:
        raise ValueError(f"unknown chunk_alignment {alignment!r}")
    return np.where(all_pad, 0.0, out).astype(np.float32)


def weight_report(result: WeightResult, *, n_total: int | None = None) -> dict[str, Any]:
    """Applied-transition accounting, as the evaluation section requires."""
    w = np.asarray(result.w, dtype=np.float64)
    n = w.size
    kept = w > 0
    out: dict[str, Any] = {
        **result.meta,
        "n_transitions_scored": int(n),
        "n_transitions_total": int(n_total if n_total is not None else n),
        "n_applied": int(kept.sum()),
        "n_dropped": int(np.asarray(result.dropped, dtype=bool).sum()),
        "n_missing_score": int(np.asarray(result.missing, dtype=bool).sum()),
        "retained_fraction": round(float(kept.mean()), 6) if n else None,
        "weight_sum": round(float(w.sum()), 6),
        "weight_mean": round(float(w.mean()), 6) if n else None,
        "weight_std": round(float(w.std()), 6) if n else None,
        "effective_sample_size": (
            round(float(w.sum() ** 2 / np.maximum((w**2).sum(), 1e-12)), 3) if n else None
        ),
    }
    if n:
        qs = np.percentile(w, [0, 5, 25, 50, 75, 95, 100])
        out["weight_quantiles"] = {
            "min": round(float(qs[0]), 6),
            "p05": round(float(qs[1]), 6),
            "p25": round(float(qs[2]), 6),
            "median": round(float(qs[3]), 6),
            "p75": round(float(qs[4]), 6),
            "p95": round(float(qs[5]), 6),
            "max": round(float(qs[6]), 6),
        }
        out["weight_histogram"] = np.histogram(w, bins=10, range=(0.0, 1.0))[0].tolist()
        trust = np.asarray(result.trust, dtype=np.float64)
        finite = trust[np.isfinite(trust)]
        if finite.size:
            out["trust_mean"] = round(float(finite.mean()), 6)
            # The uniform method has no floor (w_min is None there), so there is
            # nothing to be "at the floor" of.
            floor = result.meta.get("w_min")
            out["trust_at_floor_fraction"] = (
                round(float(np.mean(finite <= float(floor) + 1e-9)), 6)
                if floor is not None
                else None
            )
    return out
