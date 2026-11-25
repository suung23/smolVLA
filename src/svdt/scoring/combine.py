"""Weighted combination of score components with an explicit missing policy.

Shared by visibility and reliability because both face the same problem: a
component can be *unavailable* for a given transition (no track, no next state,
no segmentation evidence), and the project rules forbid quietly substituting a
high score for it.

The three policies are:

``renormalize`` (default)
    Drop unavailable components and rescale the remaining weights to sum to 1,
    per sample. A transition scored from fewer components records how many were
    used, so a reader can discount it.
``floor``
    Substitute ``missing_value`` and keep the original weights. Use when a
    missing component should actively penalise the transition.
``error``
    Refuse to score. Use in experiments that must not contain partial evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: Guard for the geometric mean so a zero component does not produce -inf.
GEOMETRIC_EPS = 1e-6


@dataclass
class CombineResult:
    """Outcome of combining components for ``N`` transitions."""

    value: np.ndarray  # (N,) float32 in [0,1]
    #: Per-sample count of components that contributed.
    n_available: np.ndarray  # (N,) int16
    #: True where *no* component had evidence; ``value`` is then the configured
    #: ``all_missing_value`` and must not be read as a real score.
    all_missing: np.ndarray  # (N,) bool
    #: Effective weight actually applied per component, after the policy.
    effective_weights: dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-component availability, carried through for the sidecar.
    availability: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n": int(self.value.size),
            "all_missing": int(self.all_missing.sum()),
            "mean_components_used": (
                round(float(self.n_available.mean()), 3) if self.value.size else None
            ),
            "component_availability": {
                k: round(float(np.mean(v)), 4) for k, v in self.availability.items()
            },
        }


def combine_components(
    components: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    policy: str = "renormalize",
    missing_value: float = 0.0,
    all_missing_value: float = 0.0,
    mode: str = "arithmetic",
    label: str = "score",
) -> CombineResult:
    """Combine ``components`` into one score per transition.

    ``components`` values must already lie in ``[0, 1]``; this is checked, not
    assumed, because a component that silently exceeds the range would break
    the ``w_t`` clip downstream.
    """
    if policy not in {"renormalize", "floor", "error"}:
        raise ValueError(f"unknown missing policy {policy!r}")
    if mode not in {"arithmetic", "geometric"}:
        raise ValueError(f"unknown combine mode {mode!r}")

    names = [k for k in weights if weights[k] > 0.0]
    dropped = [k for k in weights if weights[k] <= 0.0]
    missing_keys = sorted(set(names) - set(components))
    if missing_keys:
        raise KeyError(f"{label}: no values supplied for component(s) {missing_keys}")
    if not names:
        raise ValueError(f"{label}: every component weight is zero")

    n = len(np.asarray(components[names[0]]).reshape(-1))
    values: dict[str, np.ndarray] = {}
    avail: dict[str, np.ndarray] = {}
    for k in names:
        v = np.asarray(components[k], dtype=np.float64).reshape(-1)
        if v.size != n:
            raise ValueError(
                f"{label}: component {k!r} has {v.size} values, expected {n}"
            )
        a = np.asarray(
            availability.get(k, np.ones(n, dtype=bool)), dtype=bool
        ).reshape(-1)
        if a.size != n:
            raise ValueError(
                f"{label}: availability for {k!r} has {a.size} values, expected {n}"
            )
        finite = np.isfinite(v)
        if np.any(a & ~finite):
            raise ValueError(f"{label}: component {k!r} has non-finite available values")
        if np.any(a & ((v < -1e-6) | (v > 1 + 1e-6))):
            bad = v[a & ((v < -1e-6) | (v > 1 + 1e-6))]
            raise ValueError(
                f"{label}: component {k!r} must lie in [0,1]; "
                f"saw min={bad.min():.4f} max={bad.max():.4f}"
            )
        values[k] = np.clip(np.where(finite, v, 0.0), 0.0, 1.0)
        avail[k] = a

    stacked_avail = np.stack([avail[k] for k in names], axis=1)  # (N, K)
    n_available = stacked_avail.sum(axis=1).astype(np.int16)
    all_missing = n_available == 0

    if policy == "error" and not stacked_avail.all():
        culprits = [k for k in names if not avail[k].all()]
        raise ValueError(
            f"{label}: missing_policy='error' but component(s) {culprits} "
            "have transitions without evidence"
        )

    base_w = np.array([float(weights[k]) for k in names], dtype=np.float64)  # (K,)
    if policy == "renormalize":
        eff = stacked_avail * base_w[None, :]
        totals = eff.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            eff = np.where(totals > 0, eff / np.where(totals > 0, totals, 1.0), 0.0)
        filled = np.stack([values[k] for k in names], axis=1)
    else:  # floor
        eff = np.broadcast_to(base_w[None, :], stacked_avail.shape).copy()
        filled = np.stack(
            [np.where(avail[k], values[k], missing_value) for k in names], axis=1
        )

    if mode == "arithmetic":
        out = (eff * filled).sum(axis=1)
    else:
        logs = np.log(np.clip(filled, GEOMETRIC_EPS, 1.0))
        out = np.exp((eff * logs).sum(axis=1))
        # With renormalised weights summing to 1 this is a true weighted
        # geometric mean; with the 'floor' policy the weights already sum to 1.
    out = np.where(all_missing, float(all_missing_value), out)
    out = np.clip(out, 0.0, 1.0)

    result = CombineResult(
        value=out.astype(np.float32),
        n_available=n_available,
        all_missing=all_missing,
        effective_weights={k: eff[:, i].astype(np.float32) for i, k in enumerate(names)},
        availability={k: avail[k] for k in names},
    )
    for k in dropped:
        # Zero-weighted components are still reported, so an ablation that
        # switches one off leaves a trace in the sidecar.
        result.availability[k] = np.asarray(
            availability.get(k, np.zeros(n, dtype=bool)), dtype=bool
        ).reshape(-1)
    return result


def renormalized_weight_table(
    weights: Mapping[str, float], present: Sequence[str]
) -> dict[str, float]:
    """Scalar version of the renormalise rule, for reporting and tests."""
    total = sum(float(weights[k]) for k in present if weights.get(k, 0.0) > 0)
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {
        k: (float(weights[k]) / total if k in present and weights.get(k, 0.0) > 0 else 0.0)
        for k in weights
    }
