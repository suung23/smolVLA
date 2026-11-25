"""Optional VLM metadata: structured episode/keyframe annotations.

Strictly a *suggestion* layer. The guard rails are enforced in code, not just
documented:

* :func:`svdt.vlm.adapter.phases_from_annotation` is the only consumer, and it
  only produces task-phase grouping keys and event priors.
* A VLM annotation can never become a reliability label unless
  ``vlm.allow_as_reliability_label`` is explicitly set, and even then
  :func:`svdt.vlm.adapter.assert_not_label_source` refuses unless the config
  says so.
* Every experiment in this repo runs with ``vlm.enabled=false`` by default.
"""

from .adapter import (
    NullVLMAdapter,
    annotation_summary,
    event_frames,
    ReplayVLMAdapter,
    VLMAdapter,
    VLMCache,
    assert_not_label_source,
    make_vlm_adapter,
    phases_from_annotation,
)
from .schema import (
    ANNOTATION_SCHEMA,
    EPISODE_ANNOTATION_VERSION,
    EventCandidate,
    KeyframeAnnotation,
    SchemaError,
    validate_annotation,
)

__all__ = [
    "ANNOTATION_SCHEMA",
    "annotation_summary",
    "event_frames",
    "EPISODE_ANNOTATION_VERSION",
    "EventCandidate",
    "KeyframeAnnotation",
    "NullVLMAdapter",
    "ReplayVLMAdapter",
    "SchemaError",
    "VLMAdapter",
    "VLMCache",
    "assert_not_label_source",
    "make_vlm_adapter",
    "phases_from_annotation",
    "validate_annotation",
]
