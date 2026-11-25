"""Image-quality measurements: sharpness, exposure, camera motion.

Split into two stages on purpose:

1. :func:`measure_batch` extracts **raw** physical quantities from pixels
   (Laplacian variance, clipped-pixel fractions, inter-frame displacement).
   This is the only stage that touches images, so each frame is decoded once.
2. :func:`normalize` maps those raw numbers into ``[0, 1]`` using references
   that are only knowable after seeing the dataset (e.g. the 90th-percentile
   sharpness). Keeping it separate means a reference can be recomputed, or
   pinned from a clean reference run, without re-decoding video.

Every raw value is kept in the sidecar so the sensitivity analysis can revisit
the normalisation without re-running the expensive stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ImageQualityConfig

#: Keys produced by :func:`measure_batch`.
RAW_KEYS = (
    "laplacian_var",
    "dark_fraction",
    "bright_fraction",
    "mean_brightness",
    "displacement_px",
    "displacement_norm",
)


@dataclass
class RawImageQuality:
    """Raw per-frame measurements for one camera, aligned to ``frame_index``."""

    frame_index: np.ndarray
    laplacian_var: np.ndarray
    dark_fraction: np.ndarray
    bright_fraction: np.ndarray
    mean_brightness: np.ndarray
    displacement_px: np.ndarray
    displacement_norm: np.ndarray
    #: False where a measurement could not be taken (e.g. no previous frame).
    motion_valid: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in RAW_KEYS}


def _gray(images: np.ndarray) -> np.ndarray:
    """(B, H, W) float32 luminance in [0, 1]."""
    arr = np.asarray(images)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (B, H, W, 3) images, got {arr.shape}")
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (arr.astype(np.float32) @ w) / 255.0


def measure_batch(
    images: np.ndarray,
    frame_index: np.ndarray,
    cfg: ImageQualityConfig,
    *,
    previous_gray: np.ndarray | None = None,
) -> tuple[RawImageQuality, np.ndarray]:
    """Measure one batch; returns the measurements and the last grey frame.

    ``previous_gray`` carries the final frame of the preceding batch so
    inter-frame motion is continuous across batch boundaries. The first frame
    of an episode has no predecessor: its motion is marked invalid and later
    back-filled by :func:`normalize`.
    """
    import cv2

    gray = _gray(images)
    n = gray.shape[0]
    lap = np.empty(n, dtype=np.float32)
    dark = np.empty(n, dtype=np.float32)
    bright = np.empty(n, dtype=np.float32)
    mean_b = np.empty(n, dtype=np.float32)
    disp = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    diag = float(np.hypot(gray.shape[1], gray.shape[2])) or 1.0
    prev = previous_gray
    for i in range(n):
        g = gray[i]
        lap[i] = float(cv2.Laplacian(g, cv2.CV_32F).var())
        dark[i] = float(np.mean(g <= cfg.exposure_dark_level))
        bright[i] = float(np.mean(g >= cfg.exposure_bright_level))
        mean_b[i] = float(g.mean())
        if prev is not None and prev.shape == g.shape:
            # Phase correlation gives global translation robustly and cheaply;
            # it is the dominant term for a shaking or moving-base camera.
            try:
                (dx, dy), _ = cv2.phaseCorrelate(
                    prev.astype(np.float64), g.astype(np.float64)
                )
                disp[i] = float(np.hypot(dx, dy))
                valid[i] = True
            except cv2.error:  # pragma: no cover - degenerate uniform frames
                disp[i] = 0.0
                valid[i] = False
        prev = g

    return (
        RawImageQuality(
            frame_index=np.asarray(frame_index, dtype=np.int64),
            laplacian_var=lap,
            dark_fraction=dark,
            bright_fraction=bright,
            mean_brightness=mean_b,
            displacement_px=disp,
            displacement_norm=(disp / diag).astype(np.float32),
            motion_valid=valid,
            extras={"image_diagonal_px": diag},
        ),
        gray[-1] if n else previous_gray,
    )


def concat_raw(parts: list[RawImageQuality]) -> RawImageQuality:
    """Join per-batch measurements of one episode."""
    if not parts:
        raise ValueError("nothing to concatenate")
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
    return RawImageQuality(
        frame_index=cat("frame_index"),
        laplacian_var=cat("laplacian_var"),
        dark_fraction=cat("dark_fraction"),
        bright_fraction=cat("bright_fraction"),
        mean_brightness=cat("mean_brightness"),
        displacement_px=cat("displacement_px"),
        displacement_norm=cat("displacement_norm"),
        motion_valid=cat("motion_valid"),
        extras=dict(parts[0].extras),
    )


def sharpness_reference(values: np.ndarray, cfg: ImageQualityConfig) -> float:
    """Dataset-level Laplacian-variance reference.

    A percentile, not the maximum: one specular highlight should not define
    "sharp" for the whole dataset.
    """
    if cfg.sharpness_reference is not None:
        ref = float(cfg.sharpness_reference)
        if ref <= 0:
            raise ValueError("image_quality.sharpness_reference must be positive")
        return ref
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    ref = float(np.percentile(finite, cfg.sharpness_reference_percentile))
    return ref if ref > 1e-12 else 1.0


@dataclass
class ImageQualityScores:
    """Normalised sub-scores and their weighted combination, all in [0, 1]."""

    sharpness: np.ndarray
    exposure: np.ndarray
    camera_motion: np.ndarray
    combined: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "q_sharpness": self.sharpness,
            "q_exposure": self.exposure,
            "q_camera_motion": self.camera_motion,
            "q_image_quality": self.combined,
        }


def normalize(
    raw: RawImageQuality,
    cfg: ImageQualityConfig,
    *,
    reference: float,
) -> ImageQualityScores:
    """Map raw measurements onto ``[0, 1]`` sub-scores."""
    lap = np.asarray(raw.laplacian_var, dtype=np.float64)
    # sqrt compresses the long right tail of Laplacian variance, so a frame
    # twice as sharp as the reference does not read as "4x better".
    sharp = np.clip(np.sqrt(np.maximum(lap, 0.0) / reference), 0.0, 1.0)

    clipped = np.asarray(raw.dark_fraction, dtype=np.float64) + np.asarray(
        raw.bright_fraction, dtype=np.float64
    )
    tol = max(float(cfg.exposure_clip_tolerance), 1e-9)
    clip_term = np.clip(1.0 - clipped / tol, 0.0, 1.0)

    mean_b = np.asarray(raw.mean_brightness, dtype=np.float64)
    lo, hi = float(cfg.exposure_target_low), float(cfg.exposure_target_high)
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"invalid exposure target band: ({lo}, {hi})")
    # Distance outside the acceptable band, scaled by the room available on
    # that side, so an all-white frame scores 0 rather than merely "low".
    below = np.clip((lo - mean_b) / max(lo, 1e-9), 0.0, 1.0)
    above = np.clip((mean_b - hi) / max(1.0 - hi, 1e-9), 0.0, 1.0)
    band_term = np.clip(1.0 - (below + above), 0.0, 1.0)
    exposure = clip_term * band_term

    motion_tol = max(float(cfg.camera_motion_tolerance), 1e-9)
    disp = np.asarray(raw.displacement_norm, dtype=np.float64).copy()
    valid = np.asarray(raw.motion_valid, dtype=bool)
    if valid.any() and not valid.all():
        # The first frame of an episode has no predecessor. Back-fill from the
        # nearest measured frame instead of claiming perfect stillness.
        idx = np.arange(len(disp))
        disp[~valid] = np.interp(idx[~valid], idx[valid], disp[valid])
    elif not valid.any():
        disp[:] = 0.0
    motion = np.clip(1.0 - disp / motion_tol, 0.0, 1.0)

    w = cfg.weights()
    combined = (
        w["sharpness"] * sharp + w["exposure"] * exposure + w["camera_motion"] * motion
    )
    return ImageQualityScores(
        sharpness=sharp.astype(np.float32),
        exposure=exposure.astype(np.float32),
        camera_motion=motion.astype(np.float32),
        combined=np.clip(combined, 0.0, 1.0).astype(np.float32),
    )
