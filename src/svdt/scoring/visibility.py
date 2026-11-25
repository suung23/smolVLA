r"""Video visibility score :math:`v_t`.

What it measures
----------------
How much of the information needed to *judge* the action is present in the
frame at time ``t``. It says nothing about whether the recorded action label is
correct -- that is :mod:`svdt.scoring.reliability`'s job.

.. math::

   v_t = \lambda_o q_{object,t} + \lambda_g q_{gripper,t}
       + \lambda_s q_{seg,t} + \lambda_i q_{image,t}

with defaults ``0.30 / 0.25 / 0.20 / 0.25`` (validated to sum to 1).

Visible fraction needs a denominator
------------------------------------
``q_object`` is ``visible_area / expected_unoccluded_area``. The expected area
is not knowable from a single frame, so it comes from, in order:

1. ``segmentation.expected_area_px[role]`` if the user supplies it;
2. a robust high percentile of the areas observed in that episode, provided at
   least ``min_frames_for_expected_area`` frames had a detection;
3. nothing -- in which case the component is flagged ``missing_evidence`` and
   its weight is renormalised away. It is never given a default high score.

Compensation between components
-------------------------------
A weighted sum lets a sharp image mask a fully occluded object. Every component
is therefore preserved separately in the sidecar, and
:func:`compensation_report` quantifies how often that happens so the
sensitivity analysis can see it rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import SegmentationConfig, VisibilityConfig
from ..data.backends import FrameSource, LowDimEpisode
from ..data.schema import DatasetSchema
from ..ids import make_keys
from . import image_quality as iq
from .combine import combine_components
from .segmentation import (
    DEFAULT_ROLES,
    ROLE_GRIPPER,
    ROLE_OBJECT,
    MaskTrackSegmentationBackend,
    SegmentationBackend,
    make_segmentation_backend,
)

#: Roles that feed a visible-fraction component (``goal`` is auxiliary only).
FRACTION_ROLES = (ROLE_OBJECT, ROLE_GRIPPER)


@dataclass
class EpisodeVisibilityRaw:
    """Per-frame raw evidence for one episode, before normalisation."""

    episode_index: int
    frame_index: np.ndarray
    #: Dataset this episode came from. A corrupted variant and its clean parent
    #: share episode/frame indices, so identity must travel with the evidence.
    dataset_id: str = ""
    #: role -> (T,) observed mask area in pixels
    area_px: dict[str, np.ndarray] = field(default_factory=dict)
    #: role -> (T,) backend confidence in [0,1]; NaN where the backend gives none
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    #: Role/camera-specific raw model scores, retaining NaN when absent.
    confidence_by_camera: dict[str, np.ndarray] = field(default_factory=dict)
    evidence_by_camera: dict[str, np.ndarray] = field(default_factory=dict)
    #: role -> (T,) bool, False means the backend produced no evidence
    evidence: dict[str, np.ndarray] = field(default_factory=dict)
    #: camera_key -> raw image-quality measurements
    image_quality: dict[str, iq.RawImageQuality] = field(default_factory=dict)
    image_area_px: int = 0

    @property
    def num_frames(self) -> int:
        return len(self.frame_index)


@dataclass
class VisibilityResult:
    """Scores and components for a set of transitions."""

    keys: np.ndarray  # (N,) transition keys
    episode_index: np.ndarray
    frame_index: np.ndarray
    v: np.ndarray  # (N,) float32
    #: Per-transition dataset id; differs within one result when clean episodes
    #: and corrupted variants are scored together (as they must be, so the
    #: sharpness/expected-area references are shared).
    dataset_ids: np.ndarray | None = None
    components: dict[str, np.ndarray] = field(default_factory=dict)
    availability: dict[str, np.ndarray] = field(default_factory=dict)
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    missing_evidence: np.ndarray | None = None
    n_components_used: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)


# --------------------------------------------------------------------------- #
# stage 1: raw evidence extraction (touches pixels once)
# --------------------------------------------------------------------------- #


def extract_episode_raw(
    source: FrameSource,
    episode_index: int,
    *,
    backend: SegmentationBackend,
    iq_cfg,
    camera_keys: Sequence[str],
    roles: Sequence[str] = DEFAULT_ROLES,
    batch_size: int = 32,
    stride: int = 1,
    max_frames: int | None = None,
    keep_masks: bool = False,
) -> EpisodeVisibilityRaw:
    """Stream one episode's frames and collect raw per-frame evidence.

    Multi-camera datasets are reduced by taking, per role, the **best** view:
    a target hidden from the wrist camera but clearly visible from the
    overhead one is visible. Image quality is averaged across cameras, since a
    blurred view degrades the overall observation regardless of which camera it
    came from.
    """
    if isinstance(backend, MaskTrackSegmentationBackend):
        backend.set_episode(episode_index)

    frames: list[np.ndarray] = []
    per_cam_iq: dict[str, list[iq.RawImageQuality]] = {c: [] for c in camera_keys}
    prev_gray: dict[str, np.ndarray | None] = {c: None for c in camera_keys}
    area_batches: dict[str, list[np.ndarray]] = {r: [] for r in roles}
    conf_batches: dict[str, list[np.ndarray]] = {r: [] for r in roles}
    ev_batches: dict[str, list[np.ndarray]] = {r: [] for r in roles}
    cam_conf_batches: dict[str, list[np.ndarray]] = {
        f"{r}__{c}": [] for r in roles for c in camera_keys
    }
    cam_ev_batches: dict[str, list[np.ndarray]] = {
        f"{r}__{c}": [] for r in roles for c in camera_keys
    }
    image_area = 0

    for batch in source.iter_image_batches(
        episode_index,
        camera_keys,
        batch_size=batch_size,
        stride=stride,
        max_frames=max_frames,
    ):
        frames.append(batch.frame_index)
        n = len(batch.frame_index)
        best_area = {r: np.full(n, -1.0) for r in roles}
        best_conf = {r: np.full(n, np.nan) for r in roles}
        any_ev = {r: np.zeros(n, dtype=bool) for r in roles}

        for cam in camera_keys:
            images = batch.images[cam]
            this_cam_conf = {r: np.full(n, np.nan, dtype=np.float32) for r in roles}
            this_cam_ev = {r: np.zeros(n, dtype=bool) for r in roles}
            image_area = int(images.shape[1] * images.shape[2])
            raw_iq, prev_gray[cam] = iq.measure_batch(
                images, batch.frame_index, iq_cfg, previous_gray=prev_gray[cam]
            )
            per_cam_iq[cam].append(raw_iq)

            if isinstance(backend, MaskTrackSegmentationBackend):
                results = backend.segment_frames(
                    batch.frame_index,
                    (images.shape[1], images.shape[2]),
                    cam,
                    keep_masks=keep_masks,
                )
            else:
                results = backend.segment_batch(images, cam, keep_masks=keep_masks)

            for i, res in enumerate(results):
                for role in roles:
                    obs = res.get(role)
                    if not obs.evidence:
                        continue
                    this_cam_ev[role][i] = True
                    if obs.confidence is not None and np.isfinite(obs.confidence):
                        this_cam_conf[role][i] = float(obs.confidence)
                    any_ev[role][i] = True
                    if obs.area_px > best_area[role][i]:
                        best_area[role][i] = float(obs.area_px)
                        best_conf[role][i] = (
                            np.nan if obs.confidence is None else float(obs.confidence)
                        )

            for role in roles:
                key = f"{role}__{cam}"
                cam_conf_batches[key].append(this_cam_conf[role])
                cam_ev_batches[key].append(this_cam_ev[role])

        for role in roles:
            area = np.where(any_ev[role], np.maximum(best_area[role], 0.0), np.nan)
            area_batches[role].append(area.astype(np.float32))
            conf_batches[role].append(best_conf[role].astype(np.float32))
            ev_batches[role].append(any_ev[role])

    if not frames:
        raise ValueError(f"episode {episode_index} produced no frames")

    return EpisodeVisibilityRaw(
        episode_index=episode_index,
        frame_index=np.concatenate(frames),
        dataset_id=source.dataset_id,
        area_px={r: np.concatenate(area_batches[r]) for r in roles},
        confidence={r: np.concatenate(conf_batches[r]) for r in roles},
        confidence_by_camera={k: np.concatenate(v) for k, v in cam_conf_batches.items()},
        evidence_by_camera={k: np.concatenate(v) for k, v in cam_ev_batches.items()},
        evidence={r: np.concatenate(ev_batches[r]) for r in roles},
        image_quality={
            cam: iq.concat_raw(parts) for cam, parts in per_cam_iq.items() if parts
        },
        image_area_px=image_area,
    )


# --------------------------------------------------------------------------- #
# stage 2: expected area + normalisation
# --------------------------------------------------------------------------- #


def estimate_expected_area(
    areas: np.ndarray,
    evidence: np.ndarray,
    cfg: SegmentationConfig,
    role: str,
) -> tuple[float | None, str]:
    """``(expected_area_px, provenance)``; ``None`` means no usable denominator."""
    configured = (cfg.expected_area_px or {}).get(role)
    if configured is not None:
        if float(configured) <= 0:
            raise ValueError(f"expected_area_px[{role!r}] must be positive")
        return float(configured), "config"
    detected = np.asarray(areas, dtype=np.float64)[np.asarray(evidence, dtype=bool)]
    detected = detected[np.isfinite(detected) & (detected > 0)]
    if detected.size < cfg.min_frames_for_expected_area:
        return None, "insufficient_detections"
    ref = float(np.percentile(detected, cfg.expected_area_percentile))
    if ref <= 0:
        return None, "degenerate_reference"
    return ref, f"episode_p{cfg.expected_area_percentile:g}"


def _combine_camera_quality(
    raws: dict[str, iq.RawImageQuality], cfg, reference: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mean image-quality score across cameras, plus per-camera sub-scores."""
    per_cam: dict[str, np.ndarray] = {}
    combined: list[np.ndarray] = []
    detail: dict[str, np.ndarray] = {}
    for cam, raw in raws.items():
        scores = iq.normalize(raw, cfg, reference=reference)
        per_cam[cam] = scores.combined
        combined.append(scores.combined.astype(np.float64))
        for name, arr in scores.as_dict().items():
            detail[f"{name}__{cam}"] = arr
        for name, arr in raw.as_dict().items():
            detail[f"raw_{name}__{cam}"] = np.asarray(arr, dtype=np.float32)
    if not combined:
        raise ValueError("no image-quality measurements to combine")
    return np.mean(np.stack(combined, axis=0), axis=0).astype(np.float32), detail


class VisibilityScorer:
    """Turns raw per-frame evidence into :math:`v_t`."""

    def __init__(
        self,
        cfg: VisibilityConfig,
        schema: DatasetSchema,
        *,
        backend: SegmentationBackend | None = None,
        camera_keys: Sequence[str] | None = None,
        roles: Sequence[str] = DEFAULT_ROLES,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.schema = schema
        self.roles = tuple(roles)
        self.camera_keys = list(camera_keys or schema.camera_keys)
        if not self.camera_keys:
            raise ValueError(
                "visibility scoring needs at least one camera; set dataset.cameras"
            )
        self.backend = backend or make_segmentation_backend(
            cfg.segmentation, dataset_id=schema.dataset_id, roles=self.roles
        )

    # -- raw collection ---------------------------------------------------- #

    def collect(
        self,
        source: FrameSource,
        episodes: Iterable[int],
        *,
        batch_size: int = 32,
        stride: int = 1,
        max_frames_per_episode: int | None = None,
    ) -> list[EpisodeVisibilityRaw]:
        return [
            extract_episode_raw(
                source,
                int(ep),
                backend=self.backend,
                iq_cfg=self.cfg.image_quality,
                camera_keys=self.camera_keys,
                roles=self.roles,
                batch_size=batch_size,
                stride=stride,
                max_frames=max_frames_per_episode,
            )
            for ep in episodes
        ]

    # -- scoring ----------------------------------------------------------- #

    def score(self, raws: Sequence[EpisodeVisibilityRaw]) -> VisibilityResult:
        if not raws:
            raise ValueError("no episodes to score")

        # Sharpness reference is a dataset-level quantity: pool every camera of
        # every episode so one blurred episode cannot redefine "sharp".
        pooled = np.concatenate(
            [
                np.asarray(r.laplacian_var, dtype=np.float64)
                for raw in raws
                for r in raw.image_quality.values()
            ]
        )
        reference = iq.sharpness_reference(pooled, self.cfg.image_quality)

        comp_object: list[np.ndarray] = []
        comp_gripper: list[np.ndarray] = []
        comp_seg: list[np.ndarray] = []
        comp_image: list[np.ndarray] = []
        av_object: list[np.ndarray] = []
        av_gripper: list[np.ndarray] = []
        av_seg: list[np.ndarray] = []
        eps: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        raw_cols: dict[str, list[np.ndarray]] = {}
        expected_areas: dict[str, dict[str, Any]] = {r: {} for r in FRACTION_ROLES}

        for raw in raws:
            n = raw.num_frames
            eps.append(np.full(n, raw.episode_index, dtype=np.int64))
            frames.append(raw.frame_index)

            # visible fraction per role
            fractions: dict[str, np.ndarray] = {}
            availability: dict[str, np.ndarray] = {}
            for role in FRACTION_ROLES:
                areas = raw.area_px.get(role)
                evidence = raw.evidence.get(role)
                if areas is None or evidence is None:
                    fractions[role] = np.zeros(n, dtype=np.float32)
                    availability[role] = np.zeros(n, dtype=bool)
                    expected_areas[role][f"{raw.dataset_id or self.schema.dataset_id}|{raw.episode_index}"] = {
                        "expected_area_px": None,
                        "provenance": "role_not_requested",
                    }
                    continue
                expected, provenance = estimate_expected_area(
                    areas, evidence, self.cfg.segmentation, role
                )
                expected_areas[role][f"{raw.dataset_id or self.schema.dataset_id}|{raw.episode_index}"] = {
                    "expected_area_px": expected,
                    "provenance": provenance,
                    "detections": int(np.asarray(evidence, bool).sum()),
                }
                if expected is None:
                    # No denominator -> no visible-fraction evidence at all.
                    fractions[role] = np.zeros(n, dtype=np.float32)
                    availability[role] = np.zeros(n, dtype=bool)
                    continue
                frac = np.where(
                    np.asarray(evidence, bool),
                    np.clip(np.nan_to_num(areas, nan=0.0) / expected, 0.0, 1.0),
                    0.0,
                )
                fractions[role] = frac.astype(np.float32)
                availability[role] = np.asarray(evidence, bool)

            comp_object.append(fractions[ROLE_OBJECT])
            av_object.append(availability[ROLE_OBJECT])
            comp_gripper.append(fractions[ROLE_GRIPPER])
            av_gripper.append(availability[ROLE_GRIPPER])

            # segmentation confidence: mean over roles that reported one
            conf_stack, conf_avail = [], []
            for role in self.roles:
                c = raw.confidence.get(role)
                e = raw.evidence.get(role)
                if c is None or e is None:
                    continue
                ok = np.asarray(e, bool) & np.isfinite(c)
                conf_stack.append(np.where(ok, np.nan_to_num(c, nan=0.0), 0.0))
                conf_avail.append(ok)
            if conf_stack:
                cs = np.stack(conf_stack, axis=0)
                ca = np.stack(conf_avail, axis=0)
                denom = ca.sum(axis=0)
                seg_score = np.where(denom > 0, cs.sum(axis=0) / np.maximum(denom, 1), 0.0)
                seg_avail = denom > 0
            else:
                seg_score = np.zeros(n, dtype=np.float64)
                seg_avail = np.zeros(n, dtype=bool)
            comp_seg.append(np.clip(seg_score, 0.0, 1.0).astype(np.float32))
            av_seg.append(seg_avail)

            # image quality (always available: it needs no segmentation)
            image_score, detail = _combine_camera_quality(
                raw.image_quality, self.cfg.image_quality, reference
            )
            comp_image.append(image_score)
            for key, arr in detail.items():
                raw_cols.setdefault(key, []).append(arr)
            for role in self.roles:
                if role in raw.area_px:
                    raw_cols.setdefault(f"area_px__{role}", []).append(
                        np.nan_to_num(raw.area_px[role], nan=-1.0).astype(np.float32)
                    )
                    raw_cols.setdefault(f"seg_confidence__{role}", []).append(
                        np.asarray(raw.confidence[role], dtype=np.float32)
                    )
                    raw_cols.setdefault(f"seg_evidence__{role}", []).append(
                        np.asarray(raw.evidence[role], dtype=bool)
                    )
            for role_camera, values in raw.confidence_by_camera.items():
                raw_cols.setdefault(f"seg_confidence__{role_camera}", []).append(
                    np.asarray(values, dtype=np.float32)
                )
                raw_cols.setdefault(f"seg_evidence__{role_camera}", []).append(
                    np.asarray(raw.evidence_by_camera[role_camera], dtype=bool)
                )

        components = {
            "object": np.concatenate(comp_object),
            "gripper": np.concatenate(comp_gripper),
            "segmentation": np.concatenate(comp_seg),
            "image_quality": np.concatenate(comp_image),
        }
        n_total = components["object"].size
        availability = {
            "object": np.concatenate(av_object),
            "gripper": np.concatenate(av_gripper),
            "segmentation": np.concatenate(av_seg),
            "image_quality": np.ones(n_total, dtype=bool),
        }

        combined = combine_components(
            components,
            availability,
            self.cfg.lambdas(),
            policy=self.cfg.missing_policy,
            missing_value=self.cfg.missing_value,
            all_missing_value=self.cfg.all_missing_value,
            mode="arithmetic",
            label="visibility",
        )

        episode_index = np.concatenate(eps)
        frame_index = np.concatenate(frames)
        keys = np.concatenate(
            [
                make_keys(
                    raw.dataset_id or self.schema.dataset_id,
                    np.full(raw.num_frames, raw.episode_index, dtype=np.int64),
                    raw.frame_index,
                )
                for raw in raws
            ]
        )
        dataset_ids = np.concatenate(
            [
                np.full(
                    raw.num_frames, raw.dataset_id or self.schema.dataset_id, dtype=object
                )
                for raw in raws
            ]
        )
        # missing_evidence marks transitions whose *object/gripper* visible
        # fraction had no denominator or no detection -- the honest flag the
        # project rules ask for, distinct from "fully occluded".
        missing = ~(availability["object"] | availability["gripper"])

        return VisibilityResult(
            keys=keys,
            episode_index=episode_index,
            frame_index=frame_index,
            dataset_ids=dataset_ids,
            v=combined.value,
            components={f"q_{k}": v for k, v in components.items()},
            availability={f"avail_{k}": v for k, v in availability.items()},
            raw={k: np.concatenate(v) for k, v in raw_cols.items()},
            missing_evidence=missing,
            n_components_used=combined.n_available,
            meta={
                "lambdas": self.cfg.lambdas(),
                "missing_policy": self.cfg.missing_policy,
                "sharpness_reference": reference,
                "sharpness_reference_source": (
                    "config"
                    if self.cfg.image_quality.sharpness_reference is not None
                    else f"dataset_p{self.cfg.image_quality.sharpness_reference_percentile:g}"
                ),
                "image_quality_weights": self.cfg.image_quality.weights(),
                "segmentation": self.backend.describe(),
                "cameras": list(self.camera_keys),
                "roles": list(self.roles),
                "expected_area_px": expected_areas,
                "combine": combined.summary(),
                "all_missing_transitions": int(combined.all_missing.sum()),
            },
        )


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def compensation_report(
    result: VisibilityResult, *, low: float = 0.3, high: float = 0.7
) -> dict[str, Any]:
    """Quantify component compensation inside the weighted sum.

    The project description asks explicitly whether object/gripper occlusion
    and image quality can mask each other. This counts the transitions where
    they disagree strongly and reports what the score would have been under a
    min-combination, so the effect is measured rather than assumed.
    """
    occ = np.minimum(result.components["q_object"], result.components["q_gripper"])
    img = result.components["q_image_quality"]
    masked = (occ <= low) & (img >= high)
    inverse = (occ >= high) & (img <= low)
    lam = {k.replace("q_", ""): v for k, v in result.meta["lambdas"].items()}
    del lam
    min_combo = np.minimum.reduce(
        [result.components[k] for k in sorted(result.components)]
    )
    return {
        "n": int(len(result)),
        "occluded_but_sharp": {
            "count": int(masked.sum()),
            "fraction": round(float(masked.mean()), 4) if len(result) else None,
            "mean_v": round(float(result.v[masked].mean()), 4) if masked.any() else None,
            "note": (
                "object/gripper visibility <= "
                f"{low} while image quality >= {high}: the weighted sum keeps v_t "
                "mid-range even though the target cannot be seen"
            ),
        },
        "visible_but_degraded": {
            "count": int(inverse.sum()),
            "fraction": round(float(inverse.mean()), 4) if len(result) else None,
            "mean_v": round(float(result.v[inverse].mean()), 4) if inverse.any() else None,
        },
        "mean_v": round(float(result.v.mean()), 4) if len(result) else None,
        "mean_min_combination": round(float(min_combo.mean()), 4) if len(result) else None,
        "spearman_v_vs_min": _spearman(result.v, min_combo),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return None
    return round(float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb)), 4)


def align_to_lowdim(
    result: VisibilityResult, episodes: Sequence[LowDimEpisode]
) -> np.ndarray:
    """Positions in ``result`` for every transition of ``episodes``.

    Uses the stable transition key, never row order -- the visibility pass may
    have been run with a stride, so the two sets need not match one-to-one.
    """
    from ..ids import align_to

    target = np.concatenate([ep.keys() for ep in episodes]) if episodes else np.array([])
    return align_to(target, result.keys, required=False)
