"""Calibration metrics for task-relevant perception confidence."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..scoring.confidence import PerceptionObservation


def calibration_metrics(
    observations: Sequence[PerceptionObservation], *, bins: int = 10
) -> dict[str, Any]:
    """Brier score, reliability bins, ECE, and empirical prediction quality."""
    labeled = [r for r in observations if r.quality_label is not None]
    scored = [r for r in labeled if r.available and r.calibrated_confidence is not None]
    missing_raw = sum(r.raw_confidence is None for r in labeled)
    unscored_failures = sum(
        r.raw_confidence is None and r.quality_label is False for r in labeled
    )
    if not scored:
        return {
            "status": "no_scored_labels",
            "n_labeled": len(labeled),
            "n_missing_raw_score": missing_raw,
            "n_unscored_failures": unscored_failures,
            "labeled_success_rate": (
                round(float(np.mean([bool(r.quality_label) for r in labeled])), 6)
                if labeled else None
            ),
            "n_scored": 0,
        }
    p = np.clip(np.asarray([r.calibrated_confidence for r in scored], dtype=np.float64), 0, 1)
    y = np.asarray([bool(r.quality_label) for r in scored], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        member = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if not member.any():
            continue
        confidence = float(p[member].mean())
        accuracy = float(y[member].mean())
        count = int(member.sum())
        ece += count / len(p) * abs(confidence - accuracy)
        rows.append({
            "lower": round(float(edges[i]), 6),
            "upper": round(float(edges[i + 1]), 6),
            "count": count,
            "mean_confidence": round(confidence, 6),
            "empirical_success_rate": round(accuracy, 6),
        })
    return {
        "status": "ok",
        "n_labeled": len(labeled),
        "n_missing_raw_score": int(missing_raw),
        "n_unscored_failures": int(unscored_failures),
        "raw_score_coverage": round(len(scored) / max(len(labeled), 1), 6),
        "n_scored": len(scored),
        "positive_rate": round(float(y.mean()), 6),
        "labeled_success_rate": round(
            float(np.mean([bool(r.quality_label) for r in labeled])), 6
        ),
        "brier_score": round(float(np.mean((p - y) ** 2)), 6),
        "ece": round(float(ece), 6),
        "bins": rows,
    }


def evaluate_test_confidence(
    observations: Sequence[PerceptionObservation], *, bins: int = 10
) -> dict[str, Any]:
    """Evaluate held-out test labels only, with per-entity/condition reports."""
    test = [r for r in observations if r.split == "test"]
    groups: dict[str, list[PerceptionObservation]] = defaultdict(list)
    for row in test:
        groups[f"entity:{row.entity_id}"].append(row)
        groups[f"output_type:{row.output_type}"].append(row)
        groups[f"camera:{row.camera_id or 'unspecified'}"].append(row)
        groups[f"condition:{row.condition_id}|severity:{row.severity or 'unspecified'}"].append(row)
    return {
        "split": "test",
        "overall": calibration_metrics(test, bins=bins),
        "by_entity": {
            name.removeprefix("entity:"): calibration_metrics(rows, bins=bins)
            for name, rows in groups.items() if name.startswith("entity:")
        },
        "by_output_type": {
            name.removeprefix("output_type:"): calibration_metrics(rows, bins=bins)
            for name, rows in groups.items() if name.startswith("output_type:")
        },
        "by_camera": {
            name.removeprefix("camera:"): calibration_metrics(rows, bins=bins)
            for name, rows in groups.items() if name.startswith("camera:")
        },
        "by_condition": {
            name.removeprefix("condition:"): calibration_metrics(rows, bins=bins)
            for name, rows in groups.items() if name.startswith("condition:")
        },
    }


def write_reliability_svg(metrics: dict[str, Any], path: str | Path) -> None:
    """Write a dependency-free reliability diagram from test calibration bins."""
    from ..io_utils import ensure_dir

    target = Path(path)
    ensure_dir(target.parent)
    width, height = 680, 520
    left, top, size = 78, 48, 460
    x = lambda value: left + float(value) * size
    y = lambda value: top + (1.0 - float(value)) * size
    bins = metrics.get("overall", {}).get("bins", [])
    points = " ".join(
        f"{x(b['mean_confidence']):.1f},{y(b['empirical_success_rate']):.1f}"
        for b in bins
    )
    circles = "\n".join(
        f'<circle cx="{x(b["mean_confidence"]):.1f}" cy="{y(b["empirical_success_rate"]):.1f}" r="{min(10, 4 + b["count"] ** 0.5):.1f}" fill="#147d92" fill-opacity="0.8"><title>n={b["count"]}, confidence={b["mean_confidence"]}, success={b["empirical_success_rate"]}</title></circle>'
        for b in bins
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Held-out test reliability diagram</text>
<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top}" stroke="#9aa4b2" stroke-dasharray="6 5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+size}" stroke="#263445"/>
<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top+size}" stroke="#263445"/>
<polyline points="{points}" fill="none" stroke="#147d92" stroke-width="3"/>
{circles}
<text x="{left+size/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean calibrated confidence</text>
<text transform="translate(20 {top+size/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Empirical success rate</text>
<text x="{left}" y="{top+size+20}" font-family="sans-serif" font-size="11">0</text><text x="{left+size-5}" y="{top+size+20}" font-family="sans-serif" font-size="11">1</text>
<text x="{left-22}" y="{top+size+4}" font-family="sans-serif" font-size="11">0</text><text x="{left-22}" y="{top+5}" font-family="sans-serif" font-size="11">1</text>
<text x="{width-115}" y="{height-48}" font-family="sans-serif" font-size="11">Point size: sample count</text>
</svg>'''
    target.write_text(svg, encoding="utf-8")
