"""Offline metrics: action error, score ranking quality, degradation, seeds.

All numpy, so evaluation of scores and of offline action error runs anywhere.
Rollout success rate is deliberately *not* here -- it needs a simulator and
lives in :mod:`svdt.eval.rollout`, which refuses to invent a number when no
validated environment exists for a dataset.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# action error
# --------------------------------------------------------------------------- #


def action_mae(
    pred: np.ndarray, target: np.ndarray, pad_mask: np.ndarray | None = None
) -> dict[str, float]:
    """Mean absolute error over valid action elements.

    Reported both per element and per step so a chunk-horizon change does not
    silently rescale the number.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and target {t.shape} must match")
    if p.ndim != 3:
        raise ValueError(f"expected (B, H, A) chunks, got {p.shape}")
    valid = (
        np.ones(p.shape[:2], dtype=bool)
        if pad_mask is None
        else ~np.asarray(pad_mask, dtype=bool)
    )
    err = np.abs(p - t) * valid[..., None]
    n_elements = max(int(valid.sum()) * p.shape[2], 1)
    per_step_norm = np.linalg.norm(p - t, axis=-1) * valid
    return {
        "action_mae": float(err.sum() / n_elements),
        "action_rmse": float(
            np.sqrt((((p - t) ** 2) * valid[..., None]).sum() / n_elements)
        ),
        "action_l2_per_step": float(per_step_norm.sum() / max(int(valid.sum()), 1)),
        "n_valid_elements": n_elements,
        "n_valid_steps": int(valid.sum()),
    }


def action_chunk_error(
    pred: np.ndarray, target: np.ndarray, pad_mask: np.ndarray | None = None
) -> dict[str, Any]:
    """Per-horizon-step error, which is what a chunked policy is judged on.

    Error at step 0 is the immediately executed action; error at later steps
    shows how far ahead the policy stays usable.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    valid = (
        np.ones(p.shape[:2], dtype=bool)
        if pad_mask is None
        else ~np.asarray(pad_mask, dtype=bool)
    )
    per_step = []
    for h in range(p.shape[1]):
        m = valid[:, h]
        if not m.any():
            per_step.append(None)
            continue
        per_step.append(
            float(np.abs(p[m, h] - t[m, h]).mean())
        )
    counts = valid.sum(axis=0)
    finite = [v for v in per_step if v is not None]
    return {
        "per_step_mae": per_step,
        "per_step_counts": counts.tolist(),
        "first_step_mae": per_step[0] if per_step else None,
        "last_valid_step_mae": finite[-1] if finite else None,
        "mean_over_steps": float(np.mean(finite)) if finite else None,
        "chunk_size": int(p.shape[1]),
    }


# --------------------------------------------------------------------------- #
# ranking / classification quality of the scores
# --------------------------------------------------------------------------- #


def auroc(scores: np.ndarray, positive: np.ndarray, *, lower_is_positive: bool = True) -> float | None:
    """Rank-based AUROC with proper tie handling.

    ``lower_is_positive`` reflects how the scores are meant to behave: a
    corrupted transition should receive a *low* v/c/w.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    if lower_is_positive:
        s = -s
    ranks = _average_ranks(s)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return round(float(u / (n_pos * n_neg)), 6)


def auprc(scores: np.ndarray, positive: np.ndarray, *, lower_is_positive: bool = True) -> float | None:
    """Average precision for detecting corrupted transitions."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    # A corrupted transition is expected to score low, so rank ascending.
    order = np.argsort(s) if lower_is_positive else np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    recall_gain = y_sorted.astype(np.float64) / max(int(y.sum()), 1)
    return round(float((precision * recall_gain).sum()), 6)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so tied scores cannot inflate AUROC."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def ranking_metrics(
    scores: np.ndarray, positive: np.ndarray, *, lower_is_positive: bool = True
) -> dict[str, Any]:
    """AUROC/AUPRC plus the separation numbers that make them interpretable."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)
    ok = np.isfinite(s)
    n_pos = int(y[ok].sum())
    n_neg = int((~y[ok]).sum())
    out: dict[str, Any] = {
        "n": int(ok.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_non_finite": int((~ok).sum()),
        "auroc": auroc(s, y, lower_is_positive=lower_is_positive),
        "auprc": auprc(s, y, lower_is_positive=lower_is_positive),
        "positive_prevalence": round(n_pos / max(n_pos + n_neg, 1), 6),
    }
    if n_pos and n_neg:
        out["mean_score_positive"] = round(float(s[ok][y[ok]].mean()), 6)
        out["mean_score_negative"] = round(float(s[ok][~y[ok]].mean()), 6)
        out["median_score_positive"] = round(float(np.median(s[ok][y[ok]])), 6)
        out["median_score_negative"] = round(float(np.median(s[ok][~y[ok]])), 6)
        # Precision at the prevalence-matched cut: how good the top-k actually is.
        k = n_pos
        order = np.argsort(s[ok]) if lower_is_positive else np.argsort(-s[ok])
        out[f"precision_at_{k}"] = round(float(y[ok][order[:k]].mean()), 6)
    return out


# --------------------------------------------------------------------------- #
# degradation and seed aggregation
# --------------------------------------------------------------------------- #


def degradation_table(
    clean_metric: float,
    by_condition: Mapping[str, float],
    *,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    """Relative degradation per corruption condition.

    ``higher_is_better=False`` suits error metrics (MAE), ``True`` suits success
    rates. The sign convention is recorded so a table cannot be misread.
    """
    rows: dict[str, Any] = {}
    for name, value in by_condition.items():
        if clean_metric == 0:
            relative = None
        elif higher_is_better:
            relative = (clean_metric - value) / abs(clean_metric)
        else:
            relative = (value - clean_metric) / abs(clean_metric)
        rows[name] = {
            "value": round(float(value), 6),
            "absolute_change": round(float(value - clean_metric), 6),
            "relative_degradation": (
                None if relative is None else round(float(relative), 6)
            ),
        }
    return {
        "clean": round(float(clean_metric), 6),
        "higher_is_better": higher_is_better,
        "conditions": rows,
    }


def seed_statistics(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    """Mean/std across training seeds, with n so a single seed is obvious."""
    vals = np.asarray([float(v) for v in values_by_seed.values()], dtype=np.float64)
    if vals.size == 0:
        return {"n_seeds": 0}
    return {
        "n_seeds": int(vals.size),
        "seeds": sorted(int(s) for s in values_by_seed),
        "mean": round(float(vals.mean()), 6),
        # Sample std (ddof=1) once there is more than one seed; a single seed has
        # no spread and reporting 0.0 would overstate confidence.
        "std": (round(float(vals.std(ddof=1)), 6) if vals.size > 1 else None),
        "min": round(float(vals.min()), 6),
        "max": round(float(vals.max()), 6),
        "values": {str(k): round(float(v), 6) for k, v in sorted(values_by_seed.items())},
    }


def aggregate_over_seeds(
    runs: Sequence[Mapping[str, Any]], *, metric_keys: Sequence[str], seed_key: str = "seed"
) -> dict[str, Any]:
    """Collect ``metric_keys`` across runs and summarise per metric."""
    out: dict[str, Any] = {}
    for key in metric_keys:
        collected = {
            int(run[seed_key]): float(run[key])
            for run in runs
            if seed_key in run and key in run and run[key] is not None
        }
        out[key] = seed_statistics(collected)
    return out


def distribution_summary(values: np.ndarray, *, name: str = "value") -> dict[str, Any]:
    """Quantiles and histogram, used for the v/c/w distribution reporting."""
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"name": name, "n": 0}
    qs = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
    return {
        "name": name,
        "n": int(finite.size),
        "n_non_finite": int(v.size - finite.size),
        "mean": round(float(finite.mean()), 6),
        "std": round(float(finite.std()), 6),
        "min": round(float(qs[0]), 6),
        "p05": round(float(qs[1]), 6),
        "p25": round(float(qs[2]), 6),
        "median": round(float(qs[3]), 6),
        "p75": round(float(qs[4]), 6),
        "p95": round(float(qs[5]), 6),
        "max": round(float(qs[6]), 6),
        "histogram_0_1": np.histogram(finite, bins=10, range=(0.0, 1.0))[0].tolist(),
    }
