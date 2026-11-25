"""Segmentation/tracking interface, kept separate from the visibility scorer.

The visibility scorer never imports a segmentation model. It asks a backend for
per-role evidence and reacts to what it gets, so a colour heuristic, a prompted
SAM-style model and a pre-computed mask file are interchangeable.

Two rules are enforced by the types here:

* A backend that cannot see a target returns ``evidence=False``. It must not
  invent an area or a confidence -- the scorer turns that into
  ``missing_evidence`` rather than a high score.
* ``area_px`` is raw pixels. Converting that into a *visible fraction* needs an
  expected unoccluded area, which is the visibility scorer's business (and is
  itself allowed to be unavailable).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: Canonical target roles. ``object`` is the manipulated thing, ``gripper`` the
#: end-effector/pusher, ``goal`` the task target region (auxiliary input).
ROLE_OBJECT = "object"
ROLE_GRIPPER = "gripper"
ROLE_GOAL = "goal"
ALL_ROLES = (ROLE_OBJECT, ROLE_GRIPPER, ROLE_GOAL)


@dataclass
class TargetObservation:
    """What a backend saw of one target in one frame."""

    role: str
    area_px: float = 0.0
    #: Backend's own confidence in [0,1]; ``None`` when it does not produce one.
    confidence: float | None = None
    #: Mask centroid (x, y) in pixels, when available. Feeds motion tracking.
    centroid: tuple[float, float] | None = None
    #: Full boolean mask; optional, retained only for visualisation.
    mask: np.ndarray | None = None
    #: False means "the backend had nothing to say", not "area is zero".
    #: A genuinely fully-occluded target has ``evidence=True, area_px=0``.
    evidence: bool = True
    #: Free-form backend diagnostics preserved in the sidecar.
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return self.evidence and self.area_px > 0


@dataclass
class SegmentationResult:
    """Per-frame segmentation output for one camera."""

    backend: str
    camera_key: str
    image_area_px: int
    targets: dict[str, TargetObservation] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, role: str) -> TargetObservation:
        return self.targets.get(role, TargetObservation(role=role, evidence=False))


class SegmentationBackend(ABC):
    """Produces per-role evidence for a batch of frames."""

    #: Short identifier recorded in the sidecar.
    name: str = "abstract"
    #: Roles this backend claims to be able to find.
    roles: tuple[str, ...] = ()

    @abstractmethod
    def segment_batch(
        self, images: np.ndarray, camera_key: str, *, keep_masks: bool = False
    ) -> list[SegmentationResult]:
        """Segment ``images`` of shape (B, H, W, 3) uint8 RGB."""

    def describe(self) -> dict[str, Any]:
        """Backend provenance for the manifest."""
        return {"backend": self.name, "roles": list(self.roles)}

    def close(self) -> None:  # pragma: no cover - stateless by default
        """Release model resources."""


def centroid_of(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def largest_component(mask: np.ndarray, *, min_area: int = 1) -> np.ndarray:
    """Keep only the biggest connected blob.

    Compression artefacts scatter a handful of stray pixels with the target's
    colour; counting them as visible area would inflate the score.
    """
    if mask.sum() == 0:
        return mask
    import cv2

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] < min_area:
        return np.zeros_like(mask)
    return labels == best


def validate_roles(requested: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(requested) - set(ALL_ROLES))
    if unknown:
        raise ValueError(f"unknown segmentation role(s) {unknown}; expected {list(ALL_ROLES)}")
    return tuple(requested)
