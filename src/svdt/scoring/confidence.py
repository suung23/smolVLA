"""Task-relevant perception confidence, calibration, and aggregation.

Raw model scores are not probabilities. This module fits calibration mappings
on labelled validation episodes, applies them to observations with finite raw
scores, and leaves missing raw scores unavailable instead of inventing a value.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..ids import TransitionID


@dataclass
class PerceptionObservation:
    key: str
    entity_id: str
    entity_role: str
    output_type: str
    camera_id: str
    raw_confidence: float | None
    raw_score_type: str
    model_id: str
    model_version: str
    condition_id: str = "clean"
    severity: str = ""
    quality_label: bool | None = None
    label_source: str = ""
    available: bool = True
    missing_reason: str = ""
    calibrated_confidence: float | None = None
    calibrator_id: str = ""
    split: str = ""
    timestamp: float | None = None
    fps: float | None = None
    stride: int | None = None
    timestamp_source: str = ""

    def __post_init__(self) -> None:
        if not self.key or not self.entity_id or not self.output_type:
            raise ValueError("key, entity_id, and output_type are required")
        TransitionID.parse(self.key)
        if self.raw_confidence is not None and not math.isfinite(float(self.raw_confidence)):
            self.raw_confidence = None
        if self.raw_confidence is None:
            self.available = False
            self.missing_reason = self.missing_reason or "raw_confidence_missing"
            self.calibrated_confidence = None

    @property
    def calibration_group(self) -> tuple[str, str, str, str]:
        return (self.model_id, self.model_version, self.output_type, self.raw_score_type)


@dataclass
class Calibrator:
    method: str
    group: tuple[str, str, str, str]
    n_samples: int
    n_positive: int
    n_negative: int
    parameters: dict[str, Any]
    calibrator_id: str

    def predict(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if self.method == "platt":
            mean = float(self.parameters["mean"])
            scale = float(self.parameters["scale"])
            z = float(self.parameters["intercept"]) + float(self.parameters["slope"]) * ((x - mean) / scale)
            return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        if self.method == "isotonic":
            return np.interp(
                x,
                np.asarray(self.parameters["x"], dtype=np.float64),
                np.asarray(self.parameters["y"], dtype=np.float64),
                left=float(self.parameters["y"][0]),
                right=float(self.parameters["y"][-1]),
            )
        raise ValueError(f"unknown calibrator method {self.method!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _fit_platt(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mean = float(x.mean())
    scale = float(x.std()) or 1.0
    z = (x - mean) / scale
    # Newton updates for binary logistic regression with a small L2 penalty.
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(100):
        p = _sigmoid(design @ beta)
        grad = design.T @ (p - y) + np.array([0.0, 1e-6 * beta[1]])
        weights = np.maximum(p * (1.0 - p), 1e-8)
        hessian = design.T @ (design * weights[:, None])
        hessian[1, 1] += 1e-6
        step = np.linalg.solve(hessian, grad)
        beta -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "intercept": float(beta[0]), "slope": float(beta[1])}


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    unique, inverse = np.unique(xs, return_inverse=True)
    sums = np.bincount(inverse, weights=ys)
    counts = np.bincount(inverse).astype(np.float64)
    blocks: list[list[float]] = []  # start, end, positives, count
    for i, (s, c) in enumerate(zip(sums, counts)):
        blocks.append([float(i), float(i), float(s), float(c)])
        while len(blocks) >= 2:
            a, b = blocks[-2], blocks[-1]
            if a[2] / a[3] <= b[2] / b[3]:
                break
            blocks[-2:] = [[a[0], b[1], a[2] + b[2], a[3] + b[3]]]
    fitted = np.empty(len(unique), dtype=np.float64)
    for start, end, positive, count in blocks:
        fitted[int(start) : int(end) + 1] = positive / count
    return {"x": unique.tolist(), "y": np.clip(fitted, 0.0, 1.0).tolist()}


def fit_calibrators(
    observations: Sequence[PerceptionObservation],
    *,
    validation_episodes: set[int],
    method: str = "platt",
    min_samples: int = 20,
    min_per_class: int = 2,
) -> dict[tuple[str, str, str, str], Calibrator]:
    """Fit one calibrator per model/version/output/score semantics group.

    Only labelled, available observations from validation episodes are used.
    """
    if method not in {"platt", "isotonic"}:
        raise ValueError("calibration method must be 'platt' or 'isotonic'")
    groups: dict[tuple[str, str, str, str], list[PerceptionObservation]] = {}
    for row in observations:
        episode = TransitionID.parse(row.key).episode_index
        if episode in validation_episodes and row.available and row.raw_confidence is not None and row.quality_label is not None:
            groups.setdefault(row.calibration_group, []).append(row)

    fitted: dict[tuple[str, str, str, str], Calibrator] = {}
    for group, rows in groups.items():
        x = np.asarray([r.raw_confidence for r in rows], dtype=np.float64)
        y = np.asarray([bool(r.quality_label) for r in rows], dtype=np.float64)
        positives, negatives = int(y.sum()), int((1 - y).sum())
        if len(rows) < min_samples or min(positives, negatives) < min_per_class:
            continue
        params = _fit_platt(x, y) if method == "platt" else _fit_isotonic(x, y)
        group_text = "|".join(group)
        identity = json.dumps(
            {"group": group, "method": method, "n": len(rows), "params": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        cal_id = f"{method}:{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        fitted[group] = Calibrator(method, group, len(rows), positives, negatives, params, cal_id)
    return fitted


def apply_calibrators(
    observations: Sequence[PerceptionObservation],
    calibrators: Mapping[tuple[str, str, str, str], Calibrator],
    *,
    episode_splits: Mapping[int, str],
) -> list[PerceptionObservation]:
    """Apply fitted mappings; missing raw scores remain excluded and null."""
    output: list[PerceptionObservation] = []
    for row in observations:
        episode = TransitionID.parse(row.key).episode_index
        row.split = episode_splits.get(episode, "unassigned")
        row.calibrated_confidence = None
        cal = calibrators.get(row.calibration_group)
        if row.raw_confidence is None or not row.available:
            row.available = False
            row.missing_reason = row.missing_reason or (
                "raw_confidence_missing" if row.raw_confidence is None else "prediction_unavailable"
            )
        elif cal is None:
            row.available = False
            row.missing_reason = "no_validation_calibrator"
        else:
            row.calibrated_confidence = float(cal.predict(np.array([row.raw_confidence]))[0])
            row.calibrator_id = cal.calibrator_id
            row.available = True
            row.missing_reason = ""
        output.append(row)
    return output


def load_observations_csv(
    predictions_path: str | Path, labels_path: str | Path | None = None
) -> list[PerceptionObservation]:
    """Load model predictions and optional success labels from CSV files.

    Prediction columns: key, entity_id, entity_role, output_type, camera_id,
    raw_confidence, raw_score_type, model_id, model_version, condition_id,
    severity, available, missing_reason. Labels are joined on key, entity_id,
    output_type, and camera_id, and contain quality_label and label_source.
    """
    def rows(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    label_map: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if labels_path:
        for label in rows(labels_path):
            key = (label["key"], label["entity_id"], label["output_type"], label.get("camera_id", ""))
            label_map[key] = label

    out: list[PerceptionObservation] = []
    for pred in rows(predictions_path):
        match_key = (pred["key"], pred["entity_id"], pred["output_type"], pred.get("camera_id", ""))
        label = label_map.get(match_key, {})
        raw_text = pred.get("raw_confidence", "").strip()
        raw = float(raw_text) if raw_text else None
        success = label.get("quality_label", "").strip().lower()
        if success and success not in {"1", "true", "yes", "0", "false", "no"}:
            raise ValueError(f"invalid quality_label {success!r} for key {pred['key']}")
        out.append(
            PerceptionObservation(
                key=pred["key"],
                entity_id=pred["entity_id"],
                entity_role=pred.get("entity_role", ""),
                output_type=pred["output_type"],
                camera_id=pred.get("camera_id", ""),
                raw_confidence=raw,
                raw_score_type=pred.get("raw_score_type", "backend_confidence"),
                model_id=pred.get("model_id", "unknown"),
                model_version=pred.get("model_version", "unknown"),
                condition_id=pred.get("condition_id", "clean") or "clean",
                severity=pred.get("severity", ""),
                quality_label=(success in {"1", "true", "yes"}) if success else None,
                label_source=label.get("label_source", ""),
                available=pred.get("available", "true").strip().lower() not in {"0", "false", "no"},
                missing_reason=pred.get("missing_reason", ""),
                timestamp=float(pred["timestamp"]) if pred.get("timestamp", "").strip() else None,
                fps=float(pred["fps"]) if pred.get("fps", "").strip() else None,
                stride=int(pred["stride"]) if pred.get("stride", "").strip() else None,
                timestamp_source=pred.get("timestamp_source", ""),
            )
        )
    return out


def observations_from_score_sidecar(
    sidecar: Any,
    *,
    cameras: Sequence[str],
    entities: Sequence[Any] = (),
    model_id: str = "unknown",
    model_version: str = "unknown",
    fps: float | None = None,
    stride: int | None = None,
) -> list[PerceptionObservation]:
    """Build role/camera prediction rows from raw backend scores in a sidecar.

    This adapter preserves model scores only where the backend supplied them.
    For instance-level outputs that the current role-based backend cannot
    distinguish, provide a prediction CSV with explicit entity IDs instead.
    """
    specs = list(entities) or [
        type("Entity", (), {"entity_id": r, "role": r, "output_type": "segmentation"})()
        for r in ("object", "gripper")
    ]
    roles = [str(spec.role) for spec in specs]
    if len(roles) != len(set(roles)):
        raise ValueError(
            "sidecar role scores cannot identify multiple entities with the same role; "
            "supply --predictions CSV with one row per entity"
        )
    rows: list[PerceptionObservation] = []
    for spec in specs:
        for camera in cameras:
            raw = sidecar.raw.get(f"seg_confidence__{spec.role}__{camera}")
            for index, key in enumerate(sidecar.keys):
                identity = TransitionID.parse(str(key))
                variant = identity.dataset_id.split("__", 1)
                condition = variant[1] if len(variant) == 2 else "clean"
                value = raw[index] if raw is not None and index < len(raw) else np.nan
                val = float(value) if np.isfinite(value) else None
                rows.append(
                    PerceptionObservation(
                        key=str(key),
                        entity_id=str(spec.entity_id),
                        entity_role=str(spec.role),
                        output_type=str(spec.output_type),
                        camera_id=str(camera),
                        raw_confidence=val,
                        raw_score_type="backend_confidence",
                        model_id=model_id,
                        model_version=model_version,
                        condition_id=condition,
                        timestamp=(identity.frame_index / fps if fps and fps > 0 else None),
                        fps=fps,
                        stride=stride,
                        timestamp_source=("frame_index/fps" if fps and fps > 0 else ""),
                        missing_reason=(
                            "score_column_unavailable" if raw is None else
                            "raw_confidence_missing" if val is None else ""
                        ),
                    )
                )
    return rows


def write_observations(path: str | Path, observations: Sequence[PerceptionObservation]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in observations:
            f.write(json.dumps(asdict(row), ensure_ascii=False, allow_nan=False) + "\n")


def read_observations(path: str | Path) -> list[PerceptionObservation]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                out.append(PerceptionObservation(**data))
    return out


def aggregate_by_transition(
    observations: Sequence[PerceptionObservation],
    *,
    entity_weights: Mapping[str, float] | None = None,
    critical_entities: Iterable[str] = (),
    camera_aggregation: str = "mean",
    expected_entities: Iterable[str] = (),
) -> dict[str, float]:
    """Aggregate calibrated values with explicit missing and camera handling.

    A missing raw score never contributes zero. Non-critical missing entities
    are omitted and weights renormalized; a missing critical entity makes the
    frame aggregate unavailable. Critical scores cap the aggregate so they
    cannot be masked by another entity's high score.
    """
    if camera_aggregation not in {"mean", "minimum", "maximum"}:
        raise ValueError("camera_aggregation must be mean, minimum, or maximum")
    weights = dict(entity_weights or {})
    critical = set(critical_entities)
    expected_configured = set(expected_entities)
    grouped: dict[str, dict[str, list[float]]] = {}
    expected: dict[str, set[str]] = {}
    for row in observations:
        expected.setdefault(row.key, set()).add(row.entity_id)
        if row.available and row.calibrated_confidence is not None:
            grouped.setdefault(row.key, {}).setdefault(row.entity_id, []).append(float(row.calibrated_confidence))

    output: dict[str, float] = {}
    for key, entities in expected.items():
        entities |= expected_configured
        values: dict[str, float] = {}
        for entity, cameras in grouped.get(key, {}).items():
            arr = np.asarray(cameras, dtype=np.float64)
            values[entity] = float(arr.mean() if camera_aggregation == "mean" else arr.min() if camera_aggregation == "minimum" else arr.max())
        if critical & (entities - set(values)):
            output[key] = float("nan")
            continue
        usable = {k: v for k, v in values.items() if weights.get(k, 1.0) > 0}
        if not usable:
            output[key] = float("nan")
            continue
        w = np.asarray([weights.get(k, 1.0) for k in usable], dtype=np.float64)
        x = np.asarray([usable[k] for k in usable], dtype=np.float64)
        geo = float(np.exp(np.sum(w * np.log(np.clip(x, 1e-12, 1.0))) / w.sum()))
        if critical:
            geo = min(geo, min(values[k] for k in critical if k in values))
        output[key] = geo
    return output

\n