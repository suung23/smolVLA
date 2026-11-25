"""Interchangeable segmentation backends behind one interface.

``make_segmentation_backend`` is the only thing the visibility scorer calls, so
swapping a colour heuristic for a prompted model is a config change.
"""

from __future__ import annotations

from typing import Sequence

from ...config import SegmentationConfig
from .base import (
    ALL_ROLES,
    ROLE_GOAL,
    ROLE_GRIPPER,
    ROLE_OBJECT,
    SegmentationBackend,
    SegmentationResult,
    TargetObservation,
)
from .null import NullSegmentationBackend
from .prompted import (
    MaskTrackSegmentationBackend,
    PromptedSegmentationBackend,
    available_segmenters,
    register_segmenter,
)

__all__ = [
    "ALL_ROLES",
    "ROLE_GOAL",
    "ROLE_GRIPPER",
    "ROLE_OBJECT",
    "MaskTrackSegmentationBackend",
    "NullSegmentationBackend",
    "PromptedSegmentationBackend",
    "SegmentationBackend",
    "SegmentationResult",
    "TargetObservation",
    "available_segmenters",
    "make_segmentation_backend",
    "register_segmenter",
]

#: Roles the visibility scorer asks for by default. ``goal`` is auxiliary.
DEFAULT_ROLES = (ROLE_OBJECT, ROLE_GRIPPER, ROLE_GOAL)


def make_segmentation_backend(
    cfg: SegmentationConfig,
    *,
    dataset_id: str,
    roles: Sequence[str] | None = None,
) -> SegmentationBackend:
    """Instantiate the configured backend.

    An unknown backend name, or params a backend does not accept, raise -- the
    pipeline must not fall back to a weaker scorer behind the user's back.
    """
    wanted = tuple(roles) if roles is not None else DEFAULT_ROLES
    params = dict(cfg.params)

    if cfg.backend == "null":
        return NullSegmentationBackend(
            roles=wanted, reason=params.pop("reason", None)
        )

    if cfg.backend == "mask_track":
        root = params.pop("root", None)
        if not root:
            raise ValueError(
                "segmentation.params.root must point at the offline mask/track directory"
            )
        backend = MaskTrackSegmentationBackend(
            root=root,
            dataset_id=params.pop("dataset_id", dataset_id),
            roles=wanted,
            require_all_roles=bool(params.pop("require_all_roles", True)),
        )
        if params:
            raise ValueError(f"mask_track got unsupported params: {sorted(params)}")
        return backend

    if cfg.backend == "prompted":
        segmenter = params.pop("segmenter", "transformers_zero_shot")
        return PromptedSegmentationBackend(
            segmenter=segmenter,
            prompts=cfg.prompts,
            roles=wanted,
            min_area_px=int(params.pop("min_area_px", 4)),
            single_blob=bool(params.pop("single_blob", True)),
            params=params,
        )

    raise ValueError(
        f"unknown segmentation backend {cfg.backend!r}; "
        "expected one of: null, mask_track, prompted"
    )
