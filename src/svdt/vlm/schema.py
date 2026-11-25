"""JSON schema and validator for VLM annotations.

Validation is hand-rolled rather than depending on ``jsonschema``: the schema is
small, and a VLM annotation must be checkable in an environment that installs
nothing beyond the core dependencies. ``ANNOTATION_SCHEMA`` is still a real
JSON-Schema document so it can be handed to a model as a response format or
checked by ``jsonschema`` if available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

EPISODE_ANNOTATION_VERSION = 1

#: Event kinds the pipeline understands. Anything else is kept but ignored.
EVENT_KINDS = (
    "contact",
    "push",
    "pull",
    "open",
    "close",
    "grasp",
    "release",
    "completion",
    "retry",
    "other",
)

ANNOTATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "svdt VLM episode annotation",
    "type": "object",
    "required": ["version", "dataset_id", "episode_index", "model", "prompt_version"],
    "additionalProperties": False,
    "properties": {
        "version": {"type": "integer", "minimum": 1},
        "dataset_id": {"type": "string", "minLength": 1},
        "episode_index": {"type": "integer", "minimum": 0},
        "model": {"type": "string", "minLength": 1},
        "prompt_version": {"type": "string", "minLength": 1},
        "task_summary": {"type": ["string", "null"]},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "role": {
                        "type": "string",
                        "enum": ["manipulated", "target", "distractor", "tool", "unknown"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "start_frame", "end_frame"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "start_frame": {"type": "integer", "minimum": 0},
                    "end_frame": {"type": "integer", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "frame"],
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": list(EVENT_KINDS)},
                    "frame": {"type": "integer", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "note": {"type": ["string", "null"]},
                },
            },
        },
        "keyframes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["frame"],
                "additionalProperties": False,
                "properties": {
                    "frame": {"type": "integer", "minimum": 0},
                    "phase": {"type": ["string", "null"]},
                    "visible_objects": {"type": "array", "items": {"type": "string"}},
                    "manipulated_object": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "uncertainty_note": {"type": ["string", "null"]},
                },
            },
        },
        "uncertainty": {"type": ["string", "null"]},
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class SchemaError(ValueError):
    """An annotation that does not satisfy the schema."""


@dataclass
class EventCandidate:
    kind: str
    frame: int
    confidence: float | None = None
    note: str | None = None


@dataclass
class KeyframeAnnotation:
    frame: int
    phase: str | None = None
    visible_objects: list[str] = field(default_factory=list)
    manipulated_object: str | None = None
    confidence: float | None = None
    uncertainty_note: str | None = None


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SchemaError(message)


def _check_number(
    value: Any, path: str, *, lo: float | None = None, hi: float | None = None
) -> None:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} must be a number")
    if lo is not None:
        _require(float(value) >= lo, f"{path} must be >= {lo}")
    if hi is not None:
        _require(float(value) <= hi, f"{path} must be <= {hi}")


def _check_object(
    node: Any, spec: Mapping[str, Any], path: str
) -> None:
    _require(isinstance(node, Mapping), f"{path} must be an object")
    props: Mapping[str, Any] = spec.get("properties", {})
    for key in spec.get("required", []):
        _require(key in node, f"{path}.{key} is required")
    if spec.get("additionalProperties") is False:
        extra = sorted(set(node) - set(props))
        _require(not extra, f"{path} has unexpected key(s) {extra}")
    for key, value in node.items():
        sub = props.get(key)
        if sub is None:
            continue
        _check_value(value, sub, f"{path}.{key}")


def _check_value(value: Any, spec: Mapping[str, Any], path: str) -> None:
    types = spec.get("type")
    allowed = [types] if isinstance(types, str) else list(types or [])
    if allowed:
        ok = False
        for t in allowed:
            if t == "null" and value is None:
                ok = True
            elif t == "string" and isinstance(value, str):
                ok = True
            elif t == "integer" and isinstance(value, int) and not isinstance(value, bool):
                ok = True
            elif t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                ok = True
            elif t == "boolean" and isinstance(value, bool):
                ok = True
            elif t == "array" and isinstance(value, (list, tuple)):
                ok = True
            elif t == "object" and isinstance(value, Mapping):
                ok = True
        _require(ok, f"{path} must be of type {allowed}, got {type(value).__name__}")
    if value is None:
        return
    if "enum" in spec:
        _require(value in spec["enum"], f"{path} must be one of {spec['enum']}")
    if "minLength" in spec and isinstance(value, str):
        _require(len(value) >= spec["minLength"], f"{path} must be non-empty")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_number(value, path, lo=spec.get("minimum"), hi=spec.get("maximum"))
    if isinstance(value, (list, tuple)) and "items" in spec:
        for i, item in enumerate(value):
            _check_value(item, spec["items"], f"{path}[{i}]")
    if isinstance(value, Mapping) and "properties" in spec:
        _check_object(value, spec, path)


def validate_annotation(
    annotation: Mapping[str, Any],
    *,
    dataset_id: str | None = None,
    episode_index: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Validate one episode annotation and return it normalised.

    Beyond the schema, the checks a VLM realistically gets wrong are enforced:
    frame indices inside the episode, phase intervals ordered, and phases that
    do not overlap (an overlapping phase would make the grouping key ambiguous).
    """
    _check_object(annotation, ANNOTATION_SCHEMA, "annotation")
    out = dict(annotation)

    if out["version"] != EPISODE_ANNOTATION_VERSION:
        raise SchemaError(
            f"annotation version {out['version']} != expected {EPISODE_ANNOTATION_VERSION}"
        )
    if dataset_id is not None and out["dataset_id"] != dataset_id:
        raise SchemaError(
            f"annotation is for dataset {out['dataset_id']!r}, expected {dataset_id!r}"
        )
    if episode_index is not None and int(out["episode_index"]) != int(episode_index):
        raise SchemaError(
            f"annotation is for episode {out['episode_index']}, expected {episode_index}"
        )

    phases = list(out.get("phases") or [])
    for i, phase in enumerate(phases):
        if phase["end_frame"] < phase["start_frame"]:
            raise SchemaError(
                f"annotation.phases[{i}] ends ({phase['end_frame']}) before it starts "
                f"({phase['start_frame']})"
            )
        if num_frames is not None and phase["end_frame"] >= num_frames:
            raise SchemaError(
                f"annotation.phases[{i}].end_frame={phase['end_frame']} exceeds the "
                f"episode length {num_frames}"
            )
    ordered = sorted(phases, key=lambda p: p["start_frame"])
    for a, b in zip(ordered, ordered[1:]):
        if b["start_frame"] <= a["end_frame"]:
            raise SchemaError(
                f"annotation phases overlap: {a['name']!r} ends at {a['end_frame']} but "
                f"{b['name']!r} starts at {b['start_frame']}"
            )

    for group in ("events", "keyframes"):
        for i, item in enumerate(out.get(group) or []):
            if num_frames is not None and item["frame"] >= num_frames:
                raise SchemaError(
                    f"annotation.{group}[{i}].frame={item['frame']} exceeds the episode "
                    f"length {num_frames}"
                )
    return out


def events_of(annotation: Mapping[str, Any]) -> list[EventCandidate]:
    return [
        EventCandidate(
            kind=e["kind"],
            frame=int(e["frame"]),
            confidence=e.get("confidence"),
            note=e.get("note"),
        )
        for e in (annotation.get("events") or [])
    ]


def keyframes_of(annotation: Mapping[str, Any]) -> list[KeyframeAnnotation]:
    return [
        KeyframeAnnotation(
            frame=int(k["frame"]),
            phase=k.get("phase"),
            visible_objects=list(k.get("visible_objects") or []),
            manipulated_object=k.get("manipulated_object"),
            confidence=k.get("confidence"),
            uncertainty_note=k.get("uncertainty_note"),
        )
        for k in (annotation.get("keyframes") or [])
    ]


def keyframe_indices(num_frames: int, stride: int) -> Sequence[int]:
    """Sparse keyframe positions a VLM should be asked about."""
    if stride < 1:
        raise ValueError("keyframe_stride must be >= 1")
    return list(range(0, max(num_frames, 1), stride))
