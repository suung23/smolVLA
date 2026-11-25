"""No-evidence backend.

The honest default for a dataset we have no segmentation for. Every role comes
back with ``evidence=False``, which the visibility scorer turns into
``missing_evidence`` and a renormalised lambda set -- never into a high score.

Use it to run the pipeline on a new dataset and see what the image-quality term
alone says, before wiring up a real segmentation backend.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import (
    ROLE_GRIPPER,
    ROLE_OBJECT,
    SegmentationBackend,
    SegmentationResult,
    TargetObservation,
    validate_roles,
)


class NullSegmentationBackend(SegmentationBackend):
    """Reports no evidence for every requested role."""

    name = "null"

    def __init__(
        self, roles: Sequence[str] = (ROLE_OBJECT, ROLE_GRIPPER), reason: str | None = None
    ) -> None:
        self.roles = validate_roles(roles)
        self.reason = reason or "no segmentation backend configured"

    def segment_batch(
        self, images: np.ndarray, camera_key: str, *, keep_masks: bool = False
    ) -> list[SegmentationResult]:
        arr = np.asarray(images)
        if arr.ndim != 4:
            raise ValueError(f"expected (B, H, W, 3) images, got {arr.shape}")
        image_area = int(arr.shape[1] * arr.shape[2])
        return [
            SegmentationResult(
                backend=self.name,
                camera_key=camera_key,
                image_area_px=image_area,
                targets={
                    role: TargetObservation(role=role, evidence=False) for role in self.roles
                },
                extras={"reason": self.reason},
            )
            for _ in range(arr.shape[0])
        ]

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "roles": list(self.roles), "reason": self.reason}
