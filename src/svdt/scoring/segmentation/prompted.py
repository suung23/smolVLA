"""Prompted segmentation / tracking backends for datasets where colour fails.

ALOHA cabinet frames have no colour rule that isolates a handle, a door or a
pot, so object evidence has to come from a prompted model. Two routes, both
opt-in:

``mask_track``
    Read masks or tracks produced **offline** by a separate run (SAM 2,
    Grounded-SAM, Cutie, a human annotation pass, ...) from an ``.npz`` per
    episode. This is the recommended route: the heavy model runs once, outside
    the scoring loop, and the scoring loop stays cheap and reproducible.

``transformers_zero_shot``
    Run a zero-shot detector from ``transformers`` inside the scoring loop.
    Requires ``pip install 'svdt[segmentation]'``, an importable torch, model
    weight downloads (~700 MB for Grounding-DINO-tiny) and realistically a GPU.
    Without a segmentation head it yields **box-level** masks, which is an
    approximation recorded in the sidecar as ``mask_source="box"``.

Neither route trains a segmentation model. If the user supplies no masks and no
prompted model, the right answer is :class:`NullSegmentationBackend` and a
``missing_evidence`` flag -- not a fabricated score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .base import (
    ALL_ROLES,
    SegmentationBackend,
    SegmentationResult,
    TargetObservation,
    centroid_of,
    largest_component,
    validate_roles,
)


@runtime_checkable
class PromptedSegmenter(Protocol):
    """Maps a batch of frames plus per-role text prompts to per-role masks."""

    def __call__(
        self, images: np.ndarray, prompts: dict[str, list[str]]
    ) -> list[dict[str, dict[str, Any]]]:
        """Return, per frame, ``{role: {"mask": bool (H,W), "score": float}}``.

        A role that was not found must be *omitted* from the frame dict, so the
        caller can distinguish "model ran and saw nothing" (role present with a
        zero mask) from "model could not evaluate this role" (role absent).
        """


_SEGMENTER_FACTORIES: dict[str, Callable[..., PromptedSegmenter]] = {}


def register_segmenter(name: str, factory: Callable[..., PromptedSegmenter]) -> None:
    """Register an external segmenter factory under ``name``."""
    _SEGMENTER_FACTORIES[name] = factory


def available_segmenters() -> list[str]:
    return sorted(_SEGMENTER_FACTORIES)


def build_segmenter(name: str, **params: Any) -> PromptedSegmenter:
    if name not in _SEGMENTER_FACTORIES:
        raise ValueError(
            f"unknown segmenter {name!r}; registered: {available_segmenters()}. "
            "Register your own with svdt.scoring.segmentation.prompted.register_segmenter."
        )
    return _SEGMENTER_FACTORIES[name](**params)


# --------------------------------------------------------------------------- #
# mask_track: read offline-produced masks
# --------------------------------------------------------------------------- #


class MaskTrackSegmentationBackend(SegmentationBackend):
    """Serve masks/areas precomputed by an offline segmentation or tracking run.

    Expected layout, one file per episode::

        <root>/<dataset_id>/ep_<episode_index:06d>.npz
            frame_index : (T,)   int64
            <role>_area : (T,)   float32          # required per served role
            <role>_score: (T,)   float32          # optional confidence in [0,1]
            <role>_cx   : (T,)   float32          # optional centroid
            <role>_cy   : (T,)   float32
            <role>_mask : (T,H,W) bool/uint8      # optional, for visualisation
            <role>_valid: (T,)   bool             # optional; False -> no evidence

    Storing areas rather than full masks keeps a 1500-frame ALOHA episode in
    kilobytes. ``<role>_valid`` is how an offline tracker reports frames it
    could not evaluate (lost track), which becomes ``missing_evidence``.
    """

    name = "mask_track"

    def __init__(
        self,
        root: str | Path,
        dataset_id: str,
        roles: Sequence[str],
        *,
        require_all_roles: bool = True,
    ) -> None:
        self.roles = validate_roles(roles)
        self.root = Path(root)
        self.dataset_id = dataset_id
        self.require_all_roles = require_all_roles
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        #: Set by the visibility scorer before each episode.
        self._episode_index: int | None = None

    def episode_file(self, episode_index: int) -> Path:
        return self.root / self.dataset_id / f"ep_{episode_index:06d}.npz"

    def set_episode(self, episode_index: int) -> None:
        """Select which episode subsequent ``segment_batch`` calls refer to."""
        self._episode_index = int(episode_index)

    def _load(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._cache:
            return self._cache[episode_index]
        path = self.episode_file(episode_index)
        if not path.exists():
            raise FileNotFoundError(
                f"no mask/track file for episode {episode_index}: {path}. "
                "Produce one with your offline segmentation run, or switch the "
                "segmentation backend to 'null' to score with missing evidence."
            )
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if "frame_index" not in arrays:
            raise ValueError(f"{path} is missing the 'frame_index' array")
        if self.require_all_roles:
            missing = [r for r in self.roles if f"{r}_area" not in arrays]
            if missing:
                raise ValueError(f"{path} lacks area arrays for role(s) {missing}")
        self._cache[episode_index] = arrays
        return arrays

    def segment_batch(
        self, images: np.ndarray, camera_key: str, *, keep_masks: bool = False
    ) -> list[SegmentationResult]:
        raise NotImplementedError(
            "MaskTrackSegmentationBackend is frame-indexed, not image-indexed; "
            "call segment_frames(frame_index, image_shape, camera_key)."
        )

    def segment_frames(
        self,
        frame_index: Sequence[int] | np.ndarray,
        image_shape: tuple[int, int],
        camera_key: str,
        *,
        keep_masks: bool = False,
    ) -> list[SegmentationResult]:
        if self._episode_index is None:
            raise RuntimeError("call set_episode() before segment_frames()")
        arrays = self._load(self._episode_index)
        lookup = {int(f): i for i, f in enumerate(arrays["frame_index"])}
        height, width = image_shape
        image_area = int(height * width)
        out: list[SegmentationResult] = []
        for frame in np.asarray(frame_index, dtype=np.int64):
            row = lookup.get(int(frame))
            targets: dict[str, TargetObservation] = {}
            for role in self.roles:
                if row is None or f"{role}_area" not in arrays:
                    targets[role] = TargetObservation(role=role, evidence=False)
                    continue
                valid_arr = arrays.get(f"{role}_valid")
                valid = bool(valid_arr[row]) if valid_arr is not None else True
                if not valid:
                    targets[role] = TargetObservation(role=role, evidence=False)
                    continue
                score_arr = arrays.get(f"{role}_score")
                cx_arr, cy_arr = arrays.get(f"{role}_cx"), arrays.get(f"{role}_cy")
                mask_arr = arrays.get(f"{role}_mask") if keep_masks else None
                targets[role] = TargetObservation(
                    role=role,
                    area_px=float(arrays[f"{role}_area"][row]),
                    confidence=(
                        float(np.clip(score_arr[row], 0.0, 1.0))
                        if score_arr is not None
                        else None
                    ),
                    centroid=(
                        (float(cx_arr[row]), float(cy_arr[row]))
                        if cx_arr is not None and cy_arr is not None
                        else None
                    ),
                    mask=(
                        np.asarray(mask_arr[row]).astype(bool)
                        if mask_arr is not None
                        else None
                    ),
                    evidence=True,
                )
            out.append(
                SegmentationResult(
                    backend=self.name,
                    camera_key=camera_key,
                    image_area_px=image_area,
                    targets=targets,
                    extras={"episode_index": self._episode_index},
                )
            )
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "roles": list(self.roles),
            "root": str(self.root),
            "dataset_id": self.dataset_id,
            "provenance": "masks/tracks produced by an offline segmentation run",
        }


# --------------------------------------------------------------------------- #
# transformers_zero_shot: run a prompted model in the loop
# --------------------------------------------------------------------------- #


class PromptedSegmentationBackend(SegmentationBackend):
    """Run a registered prompted segmenter over each batch of frames."""

    name = "prompted"

    def __init__(
        self,
        segmenter: str | PromptedSegmenter,
        prompts: dict[str, list[str]],
        *,
        roles: Sequence[str] | None = None,
        min_area_px: int = 4,
        single_blob: bool = True,
        params: dict[str, Any] | None = None,
    ) -> None:
        bad = sorted(set(prompts) - set(ALL_ROLES))
        if bad:
            raise ValueError(f"prompts given for unknown role(s) {bad}")
        if not prompts:
            raise ValueError(
                "the prompted backend needs at least one role prompt, e.g. "
                "{'object': ['the pot'], 'gripper': ['the robot gripper']}"
            )
        self.roles = validate_roles(roles if roles is not None else list(prompts))
        missing = [r for r in self.roles if r not in prompts]
        if missing:
            raise ValueError(f"no prompt supplied for role(s) {missing}")
        self.prompts = {k: list(v) for k, v in prompts.items()}
        self.min_area_px = int(min_area_px)
        self.single_blob = bool(single_blob)
        self._segmenter_name = segmenter if isinstance(segmenter, str) else "callable"
        self._params = dict(params or {})
        self._segmenter: PromptedSegmenter | None = (
            None if isinstance(segmenter, str) else segmenter
        )
        self._segmenter_spec = segmenter

    def _ensure(self) -> PromptedSegmenter:
        if self._segmenter is None:
            assert isinstance(self._segmenter_spec, str)
            self._segmenter = build_segmenter(self._segmenter_spec, **self._params)
        return self._segmenter

    def segment_batch(
        self, images: np.ndarray, camera_key: str, *, keep_masks: bool = False
    ) -> list[SegmentationResult]:
        arr = np.asarray(images)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"expected (B, H, W, 3) uint8 images, got {arr.shape}")
        segmenter = self._ensure()
        per_frame = segmenter(arr, {r: self.prompts[r] for r in self.roles})
        if len(per_frame) != arr.shape[0]:
            raise ValueError(
                f"segmenter returned {len(per_frame)} results for {arr.shape[0]} frames"
            )
        image_area = int(arr.shape[1] * arr.shape[2])
        out: list[SegmentationResult] = []
        for found in per_frame:
            targets: dict[str, TargetObservation] = {}
            for role in self.roles:
                hit = found.get(role)
                if hit is None:
                    targets[role] = TargetObservation(role=role, evidence=False)
                    continue
                mask = np.asarray(hit["mask"]).astype(bool)
                if self.single_blob:
                    mask = largest_component(mask, min_area=self.min_area_px)
                area = float(mask.sum())
                targets[role] = TargetObservation(
                    role=role,
                    area_px=area if area >= self.min_area_px else 0.0,
                    confidence=(
                        float(np.clip(hit["score"], 0.0, 1.0))
                        if hit.get("score") is not None
                        else None
                    ),
                    centroid=centroid_of(mask),
                    mask=mask if keep_masks else None,
                    evidence=True,
                    extras={"mask_source": hit.get("mask_source", "mask")},
                )
            out.append(
                SegmentationResult(
                    backend=self.name,
                    camera_key=camera_key,
                    image_area_px=image_area,
                    targets=targets,
                    extras={"segmenter": self._segmenter_name},
                )
            )
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "segmenter": self._segmenter_name,
            "roles": list(self.roles),
            "prompts": self.prompts,
            "params": {k: v for k, v in self._params.items() if "token" not in k.lower()},
        }

    def close(self) -> None:
        close = getattr(self._segmenter, "close", None)
        if callable(close):
            close()
        self._segmenter = None


class _TransformersZeroShotSegmenter:
    """Zero-shot detection via ``transformers``, optionally refined by SAM.

    Heavy and opt-in. Downloads model weights on first use and wants a GPU;
    without a SAM checkpoint the "mask" is the detection box, flagged as
    ``mask_source="box"`` so a reader knows the visible-fraction estimate is
    coarse.
    """

    def __init__(
        self,
        detector_model: str = "IDEA-Research/grounding-dino-tiny",
        sam_model: str | None = None,
        device: str = "auto",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "transformers_zero_shot needs an importable torch and transformers: "
                "pip install 'svdt[segmentation]'"
            ) from exc
        from ...env_info import resolve_device

        self.device = resolve_device(device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.processor = AutoProcessor.from_pretrained(detector_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_model
        ).to(self.device)
        self.model.eval()
        self.sam = None
        self.sam_processor = None
        if sam_model:
            from transformers import AutoProcessor as SamAutoProcessor
            from transformers import SamModel

            self.sam_processor = SamAutoProcessor.from_pretrained(sam_model)
            self.sam = SamModel.from_pretrained(sam_model).to(self.device).eval()

    def __call__(
        self, images: np.ndarray, prompts: dict[str, list[str]]
    ) -> list[dict[str, dict[str, Any]]]:
        import torch
        from PIL import Image

        roles = list(prompts)
        # Grounding-DINO expects lowercase, period-separated phrases.
        text = ". ".join(p.lower().rstrip(".") for role in roles for p in prompts[role]) + "."
        phrase_role: list[tuple[str, str]] = [
            (p.lower().rstrip("."), role) for role in roles for p in prompts[role]
        ]
        pil = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in images]
        inputs = self.processor(images=pil, text=[text] * len(pil), return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[f.shape[:2] for f in images],
        )
        out: list[dict[str, dict[str, Any]]] = []
        for frame, res in zip(images, results):
            height, width = frame.shape[:2]
            best: dict[str, dict[str, Any]] = {}
            labels = res.get("text_labels", res.get("labels", []))
            for box, score, label in zip(res["boxes"], res["scores"], labels):
                label_text = str(label).lower().strip().rstrip(".")
                role = next(
                    (r for phrase, r in phrase_role if phrase and phrase in label_text),
                    None,
                )
                if role is None:
                    continue
                s = float(score)
                if role in best and best[role]["score"] >= s:
                    continue
                x0, y0, x1, y1 = (int(round(float(v))) for v in box.tolist())
                mask = np.zeros((height, width), dtype=bool)
                mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
                best[role] = {"mask": mask, "score": s, "mask_source": "box", "box": (x0, y0, x1, y1)}
            if self.sam is not None and best:
                best = self._refine_with_sam(frame, best)
            out.append(best)
        return out

    def _refine_with_sam(
        self, frame: np.ndarray, found: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        import torch
        from PIL import Image

        roles = list(found)
        boxes = [[list(found[r]["box"])] for r in roles]
        inputs = self.sam_processor(
            Image.fromarray(np.asarray(frame, dtype=np.uint8)),
            input_boxes=boxes,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        for i, role in enumerate(roles):
            found[role]["mask"] = masks[i, 0].numpy().astype(bool)
            found[role]["mask_source"] = "sam"
        return found

    def close(self) -> None:
        self.model = None
        self.sam = None


register_segmenter("transformers_zero_shot", _TransformersZeroShotSegmenter)
