"""VLM adapters and their on-disk cache.

Design constraints taken straight from the project rules:

* The core pipeline must work with no VLM. :class:`NullVLMAdapter` is the
  default and returns nothing.
* An adapter that needs an API key, a GPU or the network is kept out of the
  core: :class:`ReplayVLMAdapter` reads annotations produced by a separate,
  opt-in run, so scoring stays offline and reproducible.
* Output is cached by ``(dataset_id, episode, model, prompt_version)``, so a
  prompt change invalidates the cache instead of silently reusing stale
  annotations.
* VLM output may only shape *task-phase grouping* and event priors. It is never
  a reliability label.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import VLMConfig
from ..io_utils import fingerprint, read_json, write_json
from .schema import (
    EPISODE_ANNOTATION_VERSION,
    SchemaError,
    validate_annotation,
)


class VLMCache:
    """Content-addressed annotation cache.

    The key includes the model id and the prompt version, so re-running with a
    new prompt produces a new entry rather than reusing the old answer.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def key(self, dataset_id: str, episode_index: int, model: str, prompt_version: str) -> str:
        return fingerprint(
            {
                "dataset_id": dataset_id,
                "episode_index": int(episode_index),
                "model": model,
                "prompt_version": prompt_version,
                "schema_version": EPISODE_ANNOTATION_VERSION,
            }
        )

    def path(self, dataset_id: str, episode_index: int, model: str, prompt_version: str) -> Path:
        digest = self.key(dataset_id, episode_index, model, prompt_version)
        return self.root / dataset_id / f"ep{episode_index:06d}__{digest}.json"

    def get(
        self, dataset_id: str, episode_index: int, model: str, prompt_version: str
    ) -> dict[str, Any] | None:
        path = self.path(dataset_id, episode_index, model, prompt_version)
        data = read_json(path)
        if not isinstance(data, Mapping):
            return None
        try:
            return validate_annotation(
                data, dataset_id=dataset_id, episode_index=episode_index
            )
        except SchemaError:
            # A cached entry that no longer validates is treated as absent
            # rather than being half-trusted.
            return None

    def put(self, annotation: Mapping[str, Any]) -> Path:
        validated = validate_annotation(annotation)
        path = self.path(
            validated["dataset_id"],
            int(validated["episode_index"]),
            validated["model"],
            validated["prompt_version"],
        )
        return write_json(path, validated)


class VLMAdapter(ABC):
    """Produces structured annotations for whole episodes."""

    name: str = "abstract"

    @abstractmethod
    def annotate_episode(
        self,
        dataset_id: str,
        episode_index: int,
        *,
        num_frames: int,
        task: str | None = None,
        frames: np.ndarray | None = None,
        frame_index: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        """Annotation for one episode, or ``None`` when unavailable."""

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name}


class NullVLMAdapter(VLMAdapter):
    """The default: no VLM, no annotations, no hidden dependency."""

    name = "null"

    def annotate_episode(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return None


class ReplayVLMAdapter(VLMAdapter):
    """Serve annotations produced by a separate, opt-in VLM run.

    Expected layout: ``<root>/<dataset_id>/ep<NNNNNN>*.json``, each file a
    schema-valid annotation. Validation happens on read, so a malformed or
    stale file fails loudly instead of quietly degrading the task-phase
    grouping.
    """

    name = "replay"

    def __init__(self, root: str | Path, *, strict: bool = True) -> None:
        self.root = Path(root)
        self.strict = strict

    def annotate_episode(
        self,
        dataset_id: str,
        episode_index: int,
        *,
        num_frames: int,
        task: str | None = None,
        frames: np.ndarray | None = None,
        frame_index: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        folder = self.root / dataset_id
        if not folder.exists():
            return None
        matches = sorted(folder.glob(f"ep{episode_index:06d}*.json"))
        if not matches:
            return None
        data = read_json(matches[0])
        if not isinstance(data, Mapping):
            return None
        try:
            return validate_annotation(
                data,
                dataset_id=dataset_id,
                episode_index=episode_index,
                num_frames=num_frames,
            )
        except SchemaError:
            if self.strict:
                raise
            return None

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, "root": str(self.root), "strict": self.strict}


def make_vlm_adapter(cfg: VLMConfig) -> VLMAdapter:
    """Build the configured adapter. Disabled config yields the null adapter."""
    if not cfg.enabled or cfg.adapter == "null":
        return NullVLMAdapter()
    if cfg.adapter == "replay":
        root = cfg.params.get("root")
        if not root:
            raise ValueError(
                "vlm.adapter='replay' needs vlm.params.root pointing at the "
                "directory of precomputed annotations"
            )
        return ReplayVLMAdapter(root, strict=bool(cfg.params.get("strict", True)))
    raise ValueError(
        f"unknown vlm adapter {cfg.adapter!r}; expected 'null' or 'replay'. "
        "An online adapter is deliberately not part of the core pipeline: run "
        "your VLM separately and replay its cached output."
    )


def assert_not_label_source(cfg: VLMConfig) -> None:
    """Refuse to use VLM output as a reliability label unless explicitly allowed.

    Called wherever a code path would turn an annotation into a correctness
    judgement. The default configuration always raises, which is the point.
    """
    if not cfg.allow_as_reliability_label:
        raise PermissionError(
            "VLM output must not be used as a reliability label for c_t. The "
            "annotations are suggestions: use them for task-phase grouping and "
            "event priors only. Set vlm.allow_as_reliability_label=true to "
            "override, and say so in the experiment write-up."
        )


def phases_from_annotation(
    annotation: Mapping[str, Any] | None,
    num_frames: int,
    *,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """``(task_phase_per_frame, source)`` for the reliability grouping.

    The only sanctioned use of a VLM annotation. Frames outside every declared
    phase keep phase ``-1``, a distinct group rather than being folded into a
    neighbour.
    """
    if annotation is None or not annotation.get("phases"):
        if fallback is not None:
            return np.asarray(fallback, dtype=np.int16), "fallback"
        return np.zeros(num_frames, dtype=np.int16), "none"

    phases = sorted(annotation["phases"], key=lambda p: int(p["start_frame"]))
    names = [p["name"] for p in phases]
    lookup = {name: i for i, name in enumerate(dict.fromkeys(names))}
    out = np.full(num_frames, -1, dtype=np.int16)
    for phase in phases:
        start = max(0, int(phase["start_frame"]))
        stop = min(num_frames - 1, int(phase["end_frame"]))
        if stop >= start:
            out[start : stop + 1] = lookup[phase["name"]]
    return out, "vlm"


def event_frames(
    annotation: Mapping[str, Any] | None,
    num_frames: int,
    *,
    kinds: Sequence[str] | None = None,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """Boolean mask of frames a VLM flagged as an event of interest.

    Used only as a *prior*: the reliability scorer may relax its continuity
    penalty near a declared contact/release, since a sharp action there is
    expected. It never marks a transition as wrong.
    """
    mask = np.zeros(num_frames, dtype=bool)
    if annotation is None:
        return mask
    wanted = set(kinds) if kinds else None
    for event in annotation.get("events") or []:
        if wanted is not None and event["kind"] not in wanted:
            continue
        if float(event.get("confidence", 1.0)) < min_confidence:
            continue
        frame = int(event["frame"])
        if 0 <= frame < num_frames:
            mask[frame] = True
    return mask


def annotation_summary(annotations: Mapping[int, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Coverage report for the run manifest."""
    present = {k: v for k, v in annotations.items() if v}
    models = sorted({str(v.get("model")) for v in present.values()})
    prompts = sorted({str(v.get("prompt_version")) for v in present.values()})
    confidences = [
        float(v["overall_confidence"])
        for v in present.values()
        if v.get("overall_confidence") is not None
    ]
    return {
        "episodes_requested": len(annotations),
        "episodes_annotated": len(present),
        "models": models,
        "prompt_versions": prompts,
        "mean_overall_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "n_phases": sum(len(v.get("phases") or []) for v in present.values()),
        "n_events": sum(len(v.get("events") or []) for v in present.values()),
        "usage": "task-phase grouping and event priors only; never a c_t label",
    }
