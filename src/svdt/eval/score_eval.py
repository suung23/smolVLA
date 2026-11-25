"""Do the proposed scores actually identify the injected corruption?

This is the experiment that validates the method rather than the model. The
injected corruption labels are used **here and only here**: they come from the
corruption manifest, never from anything a scorer can see.

Reported per score (v, c, w) and per component, so a weak overall number can be
attributed: for instance, a temporal shift is caught almost entirely by
the synchronisation component while an action outlier is caught by continuity and
consistency.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..config import WeightingConfig
from ..scoring.sidecar import ScoreSidecar
from ..weighting.weights import compute_weights, weight_report
from .metrics import distribution_summary, ranking_metrics


def labels_for_sidecar(
    sidecar: ScoreSidecar, corruption_labels: Mapping[str, bool]
) -> np.ndarray:
    """Align injected-corruption labels to the sidecar by transition key.

    Keys absent from the label map are negatives: a clean variant contributes no
    entries, and the clean parent episodes are exactly the negatives we want.
    """
    return np.array(
        [bool(corruption_labels.get(str(k), False)) for k in sidecar.keys], dtype=bool
    )


def evaluate_scores(
    sidecar: ScoreSidecar,
    corruption_labels: Mapping[str, bool],
    *,
    weighting: WeightingConfig | None = None,
    per_condition: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Ranking quality of v/c/w and of every component, plus distributions.

    Args:
        sidecar: merged sidecar covering clean and corrupted transitions.
        corruption_labels: ``transition_key -> was_corrupted``.
        weighting: if given, ``w`` is derived and evaluated too.
        per_condition: ``condition_name -> transition keys``, so each corruption
            type and strength gets its own AUROC instead of one pooled number.
    """
    y = labels_for_sidecar(sidecar, corruption_labels)
    out: dict[str, Any] = {
        "n_transitions": len(sidecar),
        "n_corrupted": int(y.sum()),
        "n_clean": int((~y).sum()),
        "label_source": "injected corruption manifest (evaluation ground truth only)",
        "scores": {},
        "components": {},
        "distributions": {},
    }

    series: dict[str, np.ndarray] = {}
    if sidecar.v is not None:
        series["v"] = np.asarray(sidecar.v, dtype=np.float64)
    if sidecar.c is not None:
        series["c"] = np.asarray(sidecar.c, dtype=np.float64)
    if sidecar.q_perception is not None:
        series["q_perception"] = np.asarray(sidecar.q_perception, dtype=np.float64)

    weights = None
    if weighting is not None:
        weights = compute_weights(
            weighting,
            keys=sidecar.keys,
            v=sidecar.v,
            c=sidecar.c,
            q_perception=sidecar.q_perception,
        )
        series["w"] = np.asarray(weights.w, dtype=np.float64)
        out["weighting"] = weight_report(weights)

    for name, values in series.items():
        out["scores"][name] = ranking_metrics(values, y, lower_is_positive=True)
        out["distributions"][name] = distribution_summary(values, name=name)

    for name, values in sorted(sidecar.components.items()):
        out["components"][name] = ranking_metrics(
            np.asarray(values, dtype=np.float64), y, lower_is_positive=True
        )

    if per_condition:
        out["per_condition"] = {}
        lookup = {str(k): i for i, k in enumerate(sidecar.keys)}
        clean_mask = ~y
        for condition, keys in per_condition.items():
            pos = np.array(
                [lookup[str(k)] for k in keys if str(k) in lookup], dtype=np.int64
            )
            if pos.size == 0:
                out["per_condition"][condition] = {"status": "no_matching_transitions"}
                continue
            # Compare this condition's corrupted transitions against all clean
            # ones, so conditions are not contaminated by each other.
            sel = np.zeros(len(sidecar), dtype=bool)
            sel[pos] = True
            keep = sel | clean_mask
            entry: dict[str, Any] = {"n_corrupted": int(sel.sum())}
            for name, values in series.items():
                entry[name] = ranking_metrics(
                    values[keep], sel[keep], lower_is_positive=True
                )
            for name, values in sorted(sidecar.components.items()):
                vals = np.asarray(values, dtype=np.float64)
                entry[f"component__{name}"] = ranking_metrics(
                    vals[keep], sel[keep], lower_is_positive=True
                )
            out["per_condition"][condition] = entry

    flagged = {
        name: int(np.asarray(flag, dtype=bool).sum())
        for name, flag in sorted(sidecar.flags.items())
    }
    out["flag_counts"] = flagged
    return out


def hard_filter_tradeoff(
    sidecar: ScoreSidecar,
    corruption_labels: Mapping[str, bool],
    *,
    thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    alpha: float = 0.5,
    weighting: WeightingConfig | None = None,
) -> dict[str, Any]:
    """What a hard-filtering baseline actually discards, per threshold.

    The point of the table is the cost side: a threshold that removes most
    corrupted transitions may also remove a large share of clean data, which is
    the trade-off soft weighting is meant to avoid.
    """
    y = labels_for_sidecar(sidecar, corruption_labels)
    if weighting is not None:
        out = compute_weights(
            weighting,
            keys=sidecar.keys,
            v=sidecar.v,
            c=sidecar.c,
            q_perception=sidecar.q_perception,
        )
        trust = np.asarray(out.trust, dtype=np.float64)
    else:
        v = np.asarray(sidecar.v, dtype=np.float64) if sidecar.v is not None else None
        c = np.asarray(sidecar.c, dtype=np.float64) if sidecar.c is not None else None
        if v is None and c is None:
            raise ValueError("sidecar carries neither v nor c")
        if v is None:
            trust = c
        elif c is None:
            trust = v
        else:
            trust = alpha * np.nan_to_num(v, nan=0.0) + (1 - alpha) * np.nan_to_num(c, nan=0.0)

    rows = []
    for threshold in thresholds:
        keep = trust >= threshold
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "retained_fraction": round(float(keep.mean()), 6),
                "retained_clean_fraction": (
                    round(float(keep[~y].mean()), 6) if (~y).any() else None
                ),
                "retained_corrupted_fraction": (
                    round(float(keep[y].mean()), 6) if y.any() else None
                ),
                "corrupted_removed_fraction": (
                    round(float((~keep)[y].mean()), 6) if y.any() else None
                ),
                "clean_removed_fraction": (
                    round(float((~keep)[~y].mean()), 6) if (~y).any() else None
                ),
                "n_retained": int(keep.sum()),
            }
        )
    return {
        "alpha": alpha if weighting is None else weighting.alpha,
        "method": weighting.method if weighting is not None else "legacy_v_c",
        "n": len(sidecar),
        "rows": rows,
    }


def component_contribution(
    sidecar: ScoreSidecar,
    corruption_labels: Mapping[str, bool],
    *,
    lambdas: Mapping[str, float],
    prefix: str = "q_",
) -> dict[str, Any]:
    """Leave-one-out sensitivity of the combined score to each component.

    Answers "does this lambda earn its weight?" by recombining the stored
    components with one of them removed and re-measuring AUROC -- no video is
    re-decoded, which is why the components are kept in the sidecar.
    """
    y = labels_for_sidecar(sidecar, corruption_labels)
    present = {
        name: np.asarray(sidecar.components[f"{prefix}{name}"], dtype=np.float64)
        for name in lambdas
        if f"{prefix}{name}" in sidecar.components
    }
    if not present:
        return {"status": "no_matching_components", "looked_for": sorted(lambdas)}

    def combine(weights: Mapping[str, float]) -> np.ndarray:
        total = sum(weights.values())
        if total <= 0:
            return np.zeros(len(sidecar))
        acc = np.zeros(len(sidecar))
        for name, weight in weights.items():
            if weight <= 0 or name not in present:
                continue
            acc += (weight / total) * np.nan_to_num(present[name], nan=0.0)
        return acc

    full = combine(lambdas)
    out: dict[str, Any] = {
        "full": ranking_metrics(full, y, lower_is_positive=True),
        "leave_one_out": {},
        "single_component": {},
    }
    for name in present:
        reduced = {k: (0.0 if k == name else v) for k, v in lambdas.items()}
        out["leave_one_out"][name] = ranking_metrics(
            combine(reduced), y, lower_is_positive=True
        )
        out["single_component"][name] = ranking_metrics(
            present[name], y, lower_is_positive=True
        )
    return out
