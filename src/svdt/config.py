"""Typed configuration: nested dataclasses, YAML files, dotted CLI overrides.

Design rules enforced here:

* Every number quoted in the project description is a *default*, not a
  hard-coded constant.
* Unknown keys are an error, never silently ignored (requirement: unsupported
  settings must not be quietly dropped).
* Weight coefficients are validated (``sum == 1``) at construction time.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin

import yaml

# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


@dataclass
class CameraSpec:
    """How one camera is used by the visibility scorer.

    ``key`` is the dataset feature key, verified against ``meta/info.json``;
    nothing is guessed from camera *names*.
    """

    key: str
    role: str = "scene"  # "scene" | "wrist" | "aux"
    weight: float = 1.0


@dataclass
class DatasetConfig:
    """Dataset identity plus the facts the pipeline must not guess."""

    repo_id: str = ""
    #: Pin the dataset content. Recorded in every manifest.
    revision: str | None = None
    #: Local LeRobotDataset root; ``None`` -> HF cache.
    root: str | None = None
    #: Short name used for ``dataset_id`` and output folders.
    name: str = "dataset"

    fps: float | None = None  # None -> read from meta/info.json
    cameras: list[CameraSpec] = field(default_factory=list)
    state_key: str = "observation.state"
    action_key: str = "action"

    #: What a recorded ``action`` row *means* physically. Drives the
    #: action/state consistency component. Verify this against the dataset.
    action_semantics: str = "absolute_position_target"
    #: Indices of the action vector that are gripper/aperture commands
    #: Empty when the robot has no gripper.
    gripper_action_indices: list[int] = field(default_factory=list)
    #: Indices of the state vector positionally comparable to the action.
    #: ``None`` -> identity mapping over ``min(state_dim, action_dim)``.
    state_to_action_indices: list[int] | None = None

    task_prompt: str | None = None  # None -> read from meta/tasks
    license: str | None = None
    homepage: str | None = None

    def validate(self) -> None:
        known = {
            "absolute_position_target",
            "delta_position",
            "velocity",
            "unknown",
        }
        if self.action_semantics not in known:
            raise ValueError(
                f"unknown action_semantics {self.action_semantics!r}; expected {sorted(known)}"
            )


# --------------------------------------------------------------------------- #
# visibility
# --------------------------------------------------------------------------- #


@dataclass
class SegmentationConfig:
    """Which segmentation/tracking backend produces object evidence."""

    backend: str = "null"  # "null" | "mask_track" | "prompted"
    #: Backend-specific params, forwarded verbatim.
    params: dict[str, Any] = field(default_factory=dict)
    #: Prompts for the prompted backend, per target role.
    prompts: dict[str, list[str]] = field(default_factory=dict)
    #: Expected (unoccluded) object pixel area. ``None`` -> estimated from a
    #: robust per-episode percentile of observed area; if that is also
    #: unavailable the component is flagged ``missing_evidence`` rather than
    #: being handed a default high score.
    expected_area_px: dict[str, float] | None = None
    expected_area_percentile: float = 90.0
    #: Minimum number of frames with a detection before an episode-level
    #: expected area is trusted.
    min_frames_for_expected_area: int = 8


@dataclass
class ImageQualityConfig:
    """Sharpness / exposure / camera-motion sub-scores, each in [0, 1]."""

    w_sharpness: float = 0.4
    w_exposure: float = 0.3
    w_camera_motion: float = 0.3

    #: Laplacian variance at/above this robust reference scores 1.0.
    #: ``None`` -> per-dataset percentile of observed values.
    sharpness_reference: float | None = None
    sharpness_reference_percentile: float = 90.0
    #: Pixel fraction below/above these levels counts as clipped.
    exposure_dark_level: float = 0.02
    exposure_bright_level: float = 0.98
    #: Clipped fraction at/above this scores 0.
    exposure_clip_tolerance: float = 0.25
    #: Mean-brightness band that is not penalised.
    exposure_target_low: float = 0.25
    exposure_target_high: float = 0.75
    #: Normalised global frame-to-frame displacement that scores 0.
    camera_motion_tolerance: float = 0.15

    def weights(self) -> dict[str, float]:
        return {
            "sharpness": self.w_sharpness,
            "exposure": self.w_exposure,
            "camera_motion": self.w_camera_motion,
        }


@dataclass
class VisibilityConfig:
    """v_t = l_o*q_object + l_g*q_gripper + l_s*q_seg + l_i*q_image."""

    lambda_object: float = 0.30
    lambda_gripper: float = 0.25
    lambda_segmentation: float = 0.20
    lambda_image_quality: float = 0.25

    #: What to do when a component has no evidence.
    #: "renormalize" (default) drops it and rescales the remaining lambdas;
    #: "floor" substitutes ``missing_value``; "error" refuses to score.
    missing_policy: str = "renormalize"
    missing_value: float = 0.0
    #: If every component is missing, v_t is undefined -> this value is stored
    #: together with ``missing_evidence=True``.
    all_missing_value: float = 0.0

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    image_quality: ImageQualityConfig = field(default_factory=ImageQualityConfig)

    def lambdas(self) -> dict[str, float]:
        return {
            "object": self.lambda_object,
            "gripper": self.lambda_gripper,
            "segmentation": self.lambda_segmentation,
            "image_quality": self.lambda_image_quality,
        }

    def validate(self) -> None:
        total = sum(self.lambdas().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"visibility lambdas must sum to 1, got {total:.6f}: {self.lambdas()}"
            )
        if any(v < 0 for v in self.lambdas().values()):
            raise ValueError(f"visibility lambdas must be non-negative: {self.lambdas()}")
        if self.missing_policy not in {"renormalize", "floor", "error"}:
            raise ValueError(f"unknown missing_policy: {self.missing_policy!r}")
        iq_total = sum(self.image_quality.weights().values())
        if abs(iq_total - 1.0) > 1e-6:
            raise ValueError(f"image-quality sub-weights must sum to 1, got {iq_total:.6f}")


@dataclass
class PerceptionEntityConfig:
    """Task-specific success criterion and aggregation weight for one entity."""

    entity_id: str
    role: str
    output_type: str
    importance: float = 1.0
    critical: bool = False
    success_metric: str = "iou"
    success_threshold: float = 0.5

    def validate(self) -> None:
        if not self.entity_id or not self.role or not self.output_type:
            raise ValueError("perception entities need entity_id, role, and output_type")
        if self.importance < 0:
            raise ValueError(f"importance must be non-negative for {self.entity_id!r}")
        if self.success_threshold < 0:
            raise ValueError(f"success_threshold must be non-negative for {self.entity_id!r}")
        if self.success_metric.lower() in {"iou", "accuracy", "probability", "success_rate"} and self.success_threshold > 1:
            raise ValueError(f"{self.success_metric} threshold must lie in [0,1] for {self.entity_id!r}")


@dataclass
class ConfidenceConfig:
    """Perception confidence calibration and task-level aggregation."""

    enabled: bool = False
    calibrator: str = "platt"
    min_samples: int = 20
    min_per_class: int = 2
    camera_aggregation: str = "mean"
    model_id: str = "unknown"
    model_version: str = "unknown"
    predictions_path: str = "outputs/confidence/predictions__{dataset}.csv"
    labels_path: str = "outputs/confidence/labels__{dataset}.csv"
    observations_path: str = "outputs/confidence/observations__{dataset}.jsonl"
    entities: list[PerceptionEntityConfig] = field(default_factory=list)

    def validate(self) -> None:
        if self.calibrator not in {"platt", "isotonic"}:
            raise ValueError("confidence.calibrator must be 'platt' or 'isotonic'")
        if self.min_samples < 2 or self.min_per_class < 1:
            raise ValueError("confidence calibration sample limits must be positive")
        if self.camera_aggregation not in {"mean", "minimum", "maximum"}:
            raise ValueError("confidence.camera_aggregation must be mean/minimum/maximum")
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate perception entity ids: {ids}")
        for entity in self.entities:
            entity.validate()


# --------------------------------------------------------------------------- #
# reliability
# --------------------------------------------------------------------------- #


@dataclass
class ReliabilityConfig:
    """Components of the recorded-action reliability score c_t."""

    w_state_consistency: float = 0.35
    w_synchronization: float = 0.25
    w_residual_outlier: float = 0.30
    w_continuity: float = 0.10  # auxiliary cue only, deliberately small

    #: "arithmetic" (weighted mean) or "geometric" (weighted geometric mean,
    #: which lets a single near-zero component dominate).
    combine: str = "arithmetic"
    #: Mirrors ``VisibilityConfig.missing_policy`` for ``unavailable`` parts.
    missing_policy: str = "renormalize"
    missing_value: float = 0.0
    all_missing_value: float = 0.0

    #: Consistency: residual between the action at t and the realised state
    #: change, scaled by a robust (MAD) scale. ``consistency_tolerance``
    #: robust units map to score exp(-1) ~ 0.37.
    consistency_tolerance: float = 2.0
    #: Control latency in frames assumed between a command and its effect.
    #: ``1`` means action[t] drives state[t+1].
    control_delay_frames: int = 1

    #: Synchronisation: offsets searched, in frames.
    sync_max_offset_frames: int = 5
    sync_window_frames: int = 25
    #: Penalty per frame of detected mis-alignment.
    sync_offset_penalty: float = 0.25
    #: Correlation at/below this gives no synchronisation evidence.
    sync_min_correlation: float = 0.1
    #: What counts as "in sync". Real datasets are not aligned at offset 0:
    #: about 2 frames later, so penalising deviation from 0 would mark the
    #: whole clean dataset as desynchronised.
    #: "auto"  -- modal offset pooled over the whole scored population;
    #: "zero"  -- assume perfect alignment;
    #: "fixed" -- use ``sync_reference_offset``.
    sync_reference_mode: str = "auto"
    sync_reference_offset: float = 0.0

    #: Residual outlier: robust z above this scores 0.
    outlier_z_cutoff: float = 6.0
    #: Group residuals by these keys before computing the robust scale.
    outlier_group_by: list[str] = field(
        default_factory=lambda: ["task_phase", "state_bin"]
    )
    outlier_state_bins: int = 4
    outlier_min_group_size: int = 24

    #: Continuity: jerk tolerance in robust units. Sharp-but-explained actions
    #: (contact, direction reversal, release) are relieved when the observed
    #: state response matches, so a legitimate spike is not called an error.
    continuity_tolerance: float = 4.0
    continuity_explained_relief: float = 0.75

    def weights(self) -> dict[str, float]:
        return {
            "state_consistency": self.w_state_consistency,
            "synchronization": self.w_synchronization,
            "residual_outlier": self.w_residual_outlier,
            "continuity": self.w_continuity,
        }

    def validate(self) -> None:
        total = sum(self.weights().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reliability weights must sum to 1, got {total:.6f}")
        if any(v < 0 for v in self.weights().values()):
            raise ValueError("reliability weights must be non-negative")
        if self.combine not in {"arithmetic", "geometric"}:
            raise ValueError(f"unknown combine mode: {self.combine!r}")
        if self.missing_policy not in {"renormalize", "floor", "error"}:
            raise ValueError(f"unknown missing_policy: {self.missing_policy!r}")
        if self.control_delay_frames < 0:
            raise ValueError("control_delay_frames must be non-negative")
        if self.sync_reference_mode not in {"auto", "zero", "fixed"}:
            raise ValueError(f"unknown sync_reference_mode {self.sync_reference_mode!r}")
        if self.sync_max_offset_frames < 0:
            raise ValueError("sync_max_offset_frames must be non-negative")


# --------------------------------------------------------------------------- #
# vlm (optional)
# --------------------------------------------------------------------------- #


@dataclass
class VLMConfig:
    """Optional VLM metadata adapter. Never a label source for c_t."""

    enabled: bool = False
    adapter: str = "null"  # "null" | "replay"
    model_id: str | None = None
    prompt_version: str = "v1"
    #: Keyframe sampling stride in frames (sparse by design).
    keyframe_stride: int = 25
    cache_dir: str = "outputs/vlm_cache"
    #: Hard guard: VLM output may only feed task-phase grouping / event priors.
    allow_as_reliability_label: bool = False
    params: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# weighting
# --------------------------------------------------------------------------- #


@dataclass
class WeightingConfig:
    """w_t = clip(alpha*v_t + (1-alpha)*c_t, w_min, 1)."""

    #: Also supports perception_only, visibility_perception,
    #: perception_reliability, and combined_three_way.
    method: str = "combined"
    alpha: float = 0.5
    beta_visibility: float = 0.34
    beta_perception: float = 0.33
    beta_reliability: float = 0.33
    w_min: float = 0.15
    w_max: float = 1.0
    #: hard_filter only: transitions with raw trust below this are dropped.
    hard_filter_threshold: float = 0.5

    #: How a per-transition score maps onto a SmolVLA action chunk.
    #: "anchor" (default): the chunk inherits the score of its anchor
    #: transition t. "chunk_mean"/"chunk_min": aggregate over the chunk's
    #: non-padding steps. Documented in README.
    chunk_alignment: str = "anchor"
    #: How padding enters the outer average.
    #: "per_sample" (default) averages each sample's own mean, which is what
    #:   LeRobot's weighted path does.
    #: "per_element" additionally scales each sample weight by its number of
    #:   valid action steps, which makes the uniform case reduce *exactly* to
    #:   LeRobot's unweighted `reduction="mean"` loss even when samples have
    #:   different numbers of padded steps. See README "Chunk and padding
    #:   alignment".
    padding_normalization: str = "per_sample"
    #: What to do when one of v/c is unavailable for a transition.
    #: "renormalize" (default): use whichever score *is* available, with alpha
    #:   renormalised onto it -- the same rule the scorers use internally for
    #:   their own missing components. Flattening these to the floor instead
    #:   destroys all ranking information for them, which matters whenever
    #:   visibility is computed on a strided subset of frames.
    #: "w_min" | "one" | "drop" force a fixed outcome.
    missing_handling: str = "renormalize"
    #: Rescale batch weights to mean 1 before the weighted mean. Pure
    #: book-keeping: the normalised loss is invariant to it.
    normalize_batch_weights: bool = False

    def validate(self) -> None:
        known = {
            "uniform",
            "visibility_only",
            "reliability_only",
            "combined",
            "hard_filter",
            "perception_only",
            "visibility_perception",
            "perception_reliability",
            "combined_three_way",
        }
        if self.method not in known:
            raise ValueError(
                f"unknown weighting method {self.method!r}; expected one of {sorted(known)}"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0,1], got {self.alpha}")
        if self.method in {"visibility_perception", "perception_reliability", "combined_three_way"}:
            betas = (self.beta_visibility, self.beta_perception, self.beta_reliability)
            if any(v < 0.0 or v > 1.0 for v in betas):
                raise ValueError("beta coefficients must lie in [0,1]")
            beta_sum = sum(betas)
            if abs(beta_sum - 1.0) > 1e-6:
                raise ValueError(f"beta coefficients must sum to 1, got {beta_sum:.6f}")
            if self.method == "visibility_perception" and self.beta_visibility + self.beta_perception <= 0:
                raise ValueError("visibility_perception requires a positive visibility or perception beta")
            if self.method == "perception_reliability" and self.beta_perception + self.beta_reliability <= 0:
                raise ValueError("perception_reliability requires a positive perception or reliability beta")
        if not 0.0 <= self.w_min <= self.w_max <= 1.0:
            raise ValueError(
                f"require 0 <= w_min <= w_max <= 1, got {self.w_min}, {self.w_max}"
            )
        if self.chunk_alignment not in {"anchor", "chunk_mean", "chunk_min"}:
            raise ValueError(f"unknown chunk_alignment {self.chunk_alignment!r}")
        if self.missing_handling not in {"renormalize", "w_min", "one", "drop"}:
            raise ValueError(f"unknown missing_handling {self.missing_handling!r}")
        if self.padding_normalization not in {"per_sample", "per_element"}:
            raise ValueError(
                f"unknown padding_normalization {self.padding_normalization!r}; "
                "expected 'per_sample' or 'per_element'"
            )

    def effective_alpha(self) -> float:
        """Alpha implied by ``method`` (so ablations need one knob, not two)."""
        if self.method == "visibility_only":
            return 1.0
        if self.method == "reliability_only":
            return 0.0
        return self.alpha


# --------------------------------------------------------------------------- #
# corruption / split
# --------------------------------------------------------------------------- #


@dataclass
class CorruptionSpec:
    """One controlled corruption condition, possibly compound."""

    name: str
    #: Each entry: ``{"type": "blur", "strength": 2, ...}``. A list longer than
    #: one entry is a *compound* condition applied in order.
    ops: list[dict[str, Any]] = field(default_factory=list)
    #: Fraction of a selected episode's frames that are affected.
    frame_fraction: float = 1.0
    #: Fraction of eligible episodes that receive a corrupted copy.
    episode_fraction: float = 0.5
    seed: int = 0


@dataclass
class SplitConfig:
    """Episode-level split. Always applied *before* corruption copies."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    #: Corruption copies are confined to these splits.
    corrupt_splits: list[str] = field(
        default_factory=lambda: ["train", "val", "test"]
    )
    #: Test corruption copies exist for evaluation only; clean test episodes are
    #: never used for training or threshold tuning.
    test_corruption_is_eval_only: bool = True

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1, got {total:.6f}")
        for name, v in (("train", self.train), ("val", self.val), ("test", self.test)):
            if v < 0:
                raise ValueError(f"split fraction {name} must be non-negative")
        unknown = sorted(set(self.corrupt_splits) - {"train", "val", "test"})
        if unknown:
            raise ValueError(f"unknown split name(s) in corrupt_splits: {unknown}")


# --------------------------------------------------------------------------- #
# policy / training / eval
# --------------------------------------------------------------------------- #


@dataclass
class PolicyConfig:
    """SmolVLA checkpoint identity and the knobs we are allowed to touch.

    SmolVLA's own preprocessing / normalisation / action-chunk handling is left
    to LeRobot's processor pipeline; we only read ``chunk_size`` to align
    weights with chunks.
    """

    type: str = "smolvla"
    pretrained_path: str = "lerobot/smolvla_base"
    revision: str | None = None
    #: Read from the checkpoint's ``config.json`` when None. Verified value for
    #: ``lerobot/smolvla_base`` is 50.
    chunk_size: int | None = None
    n_action_steps: int | None = None
    device: str = "auto"
    use_amp: bool = False
    #: Extra PreTrainedConfig overrides, forwarded to LeRobot verbatim.
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 8
    num_workers: int = 2
    lr: float | None = None  # None -> checkpoint default
    seed: int = 1000
    log_freq: int = 100
    save_freq: int = 5_000
    output_dir: str = "outputs/train"
    resume: bool = False
    #: Policy backend. "lerobot" is the real path; "reference" is the
    #: torch-free numpy trainer used by the smoke test.
    backend: str = "lerobot"
    wandb: bool = False

    def validate(self) -> None:
        if self.backend not in {"lerobot", "reference"}:
            raise ValueError(f"unknown train backend {self.backend!r}")


@dataclass
class EvalConfig:
    batch_size: int = 16
    num_workers: int = 0
    seeds: list[int] = field(default_factory=lambda: [1000])
    #: Rollout is opt-in and only for datasets with a verified simulator.
    rollout_enabled: bool = False
    rollout_env_id: str | None = None
    rollout_episodes: int = 20
    output_dir: str = "outputs/eval"


@dataclass
class PathsConfig:
    data_root: str = "data"
    outputs_root: str = "outputs"
    sidecar_root: str = "outputs/sidecars"
    manifest_dir: str = "outputs"
    cache_dir: str = ".cache/svdt"


@dataclass
class Config:
    """Top-level config."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    corruptions: list[CorruptionSpec] = field(default_factory=list)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    #: Free-form note stored in run manifests.
    note: str = ""

    def validate(self) -> "Config":
        self.dataset.validate()
        self.visibility.validate()
        self.reliability.validate()
        self.confidence.validate()
        self.weighting.validate()
        self.split.validate()
        self.train.validate()
        names = [c.name for c in self.corruptions]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate corruption names: {names}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# loading / merging
# --------------------------------------------------------------------------- #


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. Lists are replaced, not concatenated."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_TYPE_TABLE: dict[str, Any] = {}


def _type_table() -> dict[str, Any]:
    if not _TYPE_TABLE:
        for obj in list(globals().values()):
            if isinstance(obj, type) and is_dataclass(obj):
                _TYPE_TABLE[obj.__name__] = obj
    return _TYPE_TABLE


def _resolve(annotation: Any) -> Any:
    """Resolve a (possibly stringified) annotation to something coercible."""
    if not isinstance(annotation, str):
        return annotation
    table = _type_table()
    text = annotation.strip()
    if text in table:
        return table[text]
    if text.startswith("list[") and text.endswith("]"):
        inner = text[5:-1].strip()
        if inner in table:
            return list[table[inner]]  # type: ignore[index]
    return Any


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    """Build the annotated type from plain YAML data."""
    tp = _resolve(annotation)
    if isinstance(tp, type) and is_dataclass(tp) and isinstance(value, Mapping):
        return from_dict(tp, value, path=path)
    origin = get_origin(tp)
    if origin in (list, set, tuple) and isinstance(value, (list, tuple)):
        args = get_args(tp)
        inner = args[0] if args else Any
        seq = [_coerce(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(seq) if origin is tuple else origin(seq)
    return value


def from_dict(cls: type, data: Mapping[str, Any], *, path: str = "") -> Any:
    """Instantiate a dataclass from a mapping; unknown keys raise."""
    if not (isinstance(cls, type) and is_dataclass(cls)):  # pragma: no cover
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        where = path or cls.__name__
        raise ValueError(
            f"unknown config key(s) at {where}: {unknown}. "
            f"Known keys: {sorted(known)}. "
            "Unsupported settings are rejected rather than silently ignored."
        )
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        child = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(f.type, data[name], child)
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML config, resolving its ``defaults:`` list first."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"config root must be a mapping: {p}")
    data = dict(data)
    defaults = data.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for d in defaults:
        merged = deep_merge(merged, load_yaml((p.parent / d).resolve()))
    return deep_merge(merged, data)


def parse_override(token: str) -> tuple[list[str], Any]:
    """``a.b.c=1`` -> (["a", "b", "c"], 1), values parsed as YAML scalars."""
    if "=" not in token:
        raise ValueError(f"override must be key=value, got {token!r}")
    key, raw = token.split("=", 1)
    keys = [k for k in key.strip().split(".") if k]
    if not keys:
        raise ValueError(f"empty override key in {token!r}")
    return keys, yaml.safe_load(raw)


def apply_overrides(data: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(dict(data), default=str)) if data else {}
    for token in overrides:
        keys, value = parse_override(token)
        node: dict[str, Any] = out
        for k in keys[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        node[keys[-1]] = value
    return out


def load_config(
    paths: list[str] | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Merge YAML files left-to-right, then apply dotted CLI overrides."""
    data: dict[str, Any] = {}
    for p in paths or []:
        data = deep_merge(data, load_yaml(p))
    data = apply_overrides(data, overrides or [])
    return from_dict(Config, data).validate()


def dump_config(cfg: Config, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return p
