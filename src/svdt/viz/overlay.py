"""Visual verification of what the scorers actually see.

Three outputs, all written as PNG with OpenCV so matplotlib stays optional:

* ``render_mask_overlay`` -- one frame with per-role masks tinted, so a wrong
  colour rule or a lost track is visible at a glance instead of being inferred
  from a number.
* ``save_episode_contact_sheet`` -- a grid of frames across an episode with the
  overlay and the per-frame v/c/w printed on each tile.
* ``save_score_timeline`` -- v_t, c_t, w_t and the components over time, with
  corrupted frames shaded.

Looking at these is the cheapest way to catch a scorer that is confidently
wrong, which is why they are part of the deliverable rather than a debug aside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.backends import FrameSource
from ..io_utils import ensure_dir
from ..scoring.segmentation import (
    ROLE_GOAL,
    ROLE_GRIPPER,
    ROLE_OBJECT,
    SegmentationBackend,
)

#: Overlay tints per role, BGR for OpenCV.
ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    ROLE_OBJECT: (0, 140, 255),  # orange
    ROLE_GRIPPER: (255, 80, 0),  # blue
    ROLE_GOAL: (0, 200, 0),  # green
}


def render_mask_overlay(
    image: np.ndarray,
    masks: Mapping[str, np.ndarray | None],
    *,
    alpha: float = 0.45,
    scale: int = 4,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Tint each role's mask over the frame; returns a BGR uint8 image.

    Small frames are upscaled with nearest-neighbour
    interpolation: the point is to see which *pixels* were counted, and a smooth
    resize would hide a one-pixel error.
    """
    import cv2

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

    for role, mask in masks.items():
        if mask is None:
            continue
        m = np.asarray(mask, dtype=bool)
        if m.shape != canvas.shape[:2] or not m.any():
            continue
        tint = np.array(ROLE_COLORS.get(role, (200, 200, 200)), dtype=np.float32)
        canvas[m] = (1 - alpha) * canvas[m] + alpha * tint

    out = np.clip(canvas, 0, 255).astype(np.uint8)
    if scale > 1:
        out = cv2.resize(
            out, (out.shape[1] * scale, out.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )
    for i, text in enumerate(labels or []):
        cv2.putText(
            out,
            text,
            (4, 14 + 14 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (4, 14 + 14 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def save_episode_contact_sheet(
    source: FrameSource,
    episode_index: int,
    backend: SegmentationBackend,
    path: str | Path,
    *,
    camera_key: str | None = None,
    scores: Mapping[str, Mapping[int, float]] | None = None,
    n_tiles: int = 12,
    scale: int = 3,
    columns: int = 4,
) -> Path:
    """Grid of overlaid frames spanning one episode, annotated with scores."""
    import cv2

    from ..scoring.segmentation.prompted import MaskTrackSegmentationBackend

    camera = camera_key or source.schema.camera_keys[0]
    total = source.episode_length(episode_index)
    stride = max(1, total // max(1, n_tiles))
    if isinstance(backend, MaskTrackSegmentationBackend):
        backend.set_episode(episode_index)

    tiles: list[np.ndarray] = []
    for batch in source.iter_image_batches(
        episode_index, [camera], batch_size=1, stride=stride, max_frames=n_tiles
    ):
        images = batch.images[camera]
        if isinstance(backend, MaskTrackSegmentationBackend):
            results = backend.segment_frames(
                batch.frame_index, images.shape[1:3], camera, keep_masks=True
            )
        else:
            results = backend.segment_batch(images, camera, keep_masks=True)
        for image, result, frame in zip(images, results, batch.frame_index):
            masks = {
                role: obs.mask for role, obs in result.targets.items() if obs.mask is not None
            }
            labels = [f"f{int(frame)}"]
            for name, table in (scores or {}).items():
                value = table.get(int(frame))
                if value is not None:
                    labels.append(f"{name}={value:.2f}")
            tiles.append(
                render_mask_overlay(image, masks, scale=scale, labels=labels)
            )

    if not tiles:
        raise ValueError(f"episode {episode_index} produced no frames to visualise")

    height, width = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * height, columns * width, 3), 30, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile

    out = Path(path)
    ensure_dir(out.parent)
    if not cv2.imwrite(str(out), sheet):
        raise OSError(f"could not write {out}")
    return out


def save_score_timeline(
    path: str | Path,
    frame_index: Sequence[int],
    series: Mapping[str, Sequence[float]],
    *,
    corrupted: Sequence[bool] | None = None,
    title: str = "",
    width: int = 1100,
    height: int = 420,
) -> Path:
    """Score timeline as a PNG.

    Uses matplotlib when available (nicer axes) and falls back to drawing with
    OpenCV, so the visualisation never becomes the reason a run fails on a
    minimal install.
    """
    out = Path(path)
    ensure_dir(out.parent)
    frames = np.asarray(frame_index, dtype=np.float64)
    corrupt = (
        np.asarray(corrupted, dtype=bool)
        if corrupted is not None
        else np.zeros(len(frames), dtype=bool)
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _timeline_opencv(out, frames, series, corrupt, title, width, height)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    if corrupt.any():
        # Shade the injected-corruption spans so score drops can be read
        # against them directly.
        edges = np.flatnonzero(np.diff(corrupt.astype(np.int8)) != 0) + 1
        bounds = np.concatenate([[0], edges, [len(corrupt)]])
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if corrupt[start]:
                ax.axvspan(
                    frames[start],
                    frames[min(stop, len(frames) - 1)],
                    color="crimson",
                    alpha=0.12,
                    lw=0,
                )
    for name, values in series.items():
        vals = np.asarray(values, dtype=np.float64)
        style = {"lw": 2.0} if name in {"v", "c", "w"} else {"lw": 1.0, "alpha": 0.65}
        ax.plot(frames[: len(vals)], vals, label=name, **style)
    ax.set_xlabel("frame_index")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", ncols=3, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    if corrupt.any():
        ax.plot([], [], color="crimson", alpha=0.3, lw=8, label="injected corruption")
        ax.legend(loc="lower left", ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _timeline_opencv(
    out: Path,
    frames: np.ndarray,
    series: Mapping[str, Sequence[float]],
    corrupt: np.ndarray,
    title: str,
    width: int,
    height: int,
) -> Path:
    import cv2

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    pad = 48
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    n = max(len(frames), 2)

    for i, flag in enumerate(corrupt):
        if flag:
            x0 = pad + int(plot_w * i / (n - 1))
            x1 = pad + int(plot_w * min(i + 1, n - 1) / (n - 1))
            cv2.rectangle(canvas, (x0, pad), (max(x1, x0 + 1), pad + plot_h), (220, 220, 255), -1)

    cv2.rectangle(canvas, (pad, pad), (pad + plot_w, pad + plot_h), (0, 0, 0), 1)
    palette = [(200, 60, 60), (60, 140, 40), (40, 80, 220), (140, 100, 20), (120, 40, 160)]
    for k, (name, values) in enumerate(series.items()):
        vals = np.asarray(values, dtype=np.float64)
        color = palette[k % len(palette)]
        points = [
            (
                pad + int(plot_w * i / (n - 1)),
                pad + plot_h - int(plot_h * float(np.clip(v, 0, 1))),
            )
            for i, v in enumerate(vals)
            if np.isfinite(v)
        ]
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, color, 2 if name in {"v", "c", "w"} else 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (pad + 6, pad + 16 + 16 * k),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if title:
        cv2.putText(
            canvas, title, (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    if not cv2.imwrite(str(out), canvas):
        raise OSError(f"could not write {out}")
    return out


def series_from_sidecar(
    sidecar: Any, episode_index: int, *, include_components: bool = True
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``(frame_index, {name: values})`` for one episode of a sidecar."""
    mask = np.asarray(sidecar.episode_index) == int(episode_index)
    frames = np.asarray(sidecar.frame_index)[mask]
    order = np.argsort(frames)
    series: dict[str, np.ndarray] = {}
    for name, arr in (("v", sidecar.v), ("q_perception", sidecar.q_perception), ("c", sidecar.c)):
        if arr is not None:
            series[name] = np.asarray(arr)[mask][order]
    if include_components:
        for name, arr in sorted(sidecar.components.items()):
            series[name] = np.asarray(arr)[mask][order]
    return frames[order], series
