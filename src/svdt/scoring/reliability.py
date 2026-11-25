r"""Recorded-action reliability score :math:`c_t`.

What it measures
----------------
Whether the action stored at time ``t`` is a *trustworthy record*: temporally
consistent with the observation/state that follows it, and plausible for this
dataset, task phase and state. It is not a measure of whether the action was
*optimal*.

Four components, each in ``[0, 1]`` and each independently preserved:

``state_consistency``
    Residual between what the action predicts for the next state and what the
    next state actually was. The prediction depends on what an action
    physically *is* (``dataset.action_semantics``) and on the control delay
    (``control_delay_frames``) -- neither is guessed.

``synchronization``
    Searches small time offsets between the action-change signal and the
    observed motion signal. A dataset whose actions were shifted against its
    observations shows a non-zero best offset.

``residual_outlier``
    Robust (MAD-based) z-score of the residual *within* comparable transitions
    -- same dataset, task phase and state bin -- so "unusual" means unusual for
    this situation, not merely large in absolute terms.

``continuity``
    Auxiliary only (default weight 0.10). A sharp action is normal at contact,
    at a direction reversal, or at release, so a spike whose state response
    matches the command is relieved instead of being called an error.

A single action-norm test is never used to decide reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import DatasetConfig, ReliabilityConfig
from ..data.backends import LowDimEpisode
from ..data.schema import DatasetSchema, state_action_index_map
from ..ids import make_keys
from .combine import combine_components

#: Scale factor making the median absolute deviation a consistent estimator of
#: the standard deviation for normal data.
MAD_TO_SIGMA = 1.4826

COMPONENTS = (
    "state_consistency",
    "synchronization",
    "residual_outlier",
    "continuity",
)


def robust_scale(values: np.ndarray, *, floor: float = 1e-9) -> float:
    """MAD-based scale, immune to the outliers we are trying to detect.

    The fallback chain matters. When the MAD is zero, at least half the values
    are identical and any departure from them *is* the anomaly -- so the
    fallback must stay small. Falling back to the standard deviation would be
    self-defeating: one extreme outlier inflates the std, which then divides its
    own z-score back down to nothing and hides it.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return floor
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = MAD_TO_SIGMA * mad
    if scale > floor:
        return scale
    # Near-constant centre: try the (still robust) interquartile spread.
    iqr = abs(float(np.subtract(*np.percentile(v, [75, 25]))))
    if iqr > floor:
        return max(iqr / 1.349, floor)
    # Fully degenerate: use a small fraction of the central magnitude so that a
    # genuine departure gets a large z instead of being normalised away.
    return max(1e-3 * max(abs(med), 1.0), floor)


def robust_z(values: np.ndarray, *, center: float | None = None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    med = float(np.median(finite)) if finite.size else 0.0
    if center is not None:
        med = center
    scale = robust_scale(v)
    return np.abs(v - med) / scale


@dataclass
class EpisodeReliabilityRaw:
    """Per-frame raw signals for one episode, before dataset-level scaling."""

    episode_index: int
    frame_index: np.ndarray
    #: ``||predicted next state - observed next state||`` on the mapped dims.
    residual: np.ndarray
    residual_valid: np.ndarray
    #: ``||action[t+1] - action[t]||``, the action-change magnitude.
    action_delta: np.ndarray
    #: Observed motion magnitude used for synchronisation.
    motion: np.ndarray
    motion_source: str
    #: Second difference of the action, the continuity cue.
    jerk: np.ndarray
    jerk_valid: np.ndarray
    #: Best time offset found locally, in frames (NaN where inconclusive).
    best_offset: np.ndarray
    offset_correlation: np.ndarray
    #: Grouping keys for the outlier component.
    task_phase: np.ndarray
    state_bin: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.frame_index)


@dataclass
class ReliabilityResult:
    keys: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    c: np.ndarray
    #: Per-transition dataset id; differs within one result when clean episodes
    #: and corrupted variants are scored together (as they must be, so the
    #: robust scales are shared).
    dataset_ids: np.ndarray | None = None
    components: dict[str, np.ndarray] = field(default_factory=dict)
    availability: dict[str, np.ndarray] = field(default_factory=dict)
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    unavailable: np.ndarray | None = None
    n_components_used: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)


# --------------------------------------------------------------------------- #
# per-episode signal extraction
# --------------------------------------------------------------------------- #


def predict_next_state(
    episode: LowDimEpisode,
    *,
    semantics: str,
    delay: int,
    state_idx: Sequence[int],
    action_idx: Sequence[int],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(prediction, observation, valid)`` on the comparable dimensions.

    ``valid`` is False for the last ``delay`` frames of the episode, which have
    no observed successor -- those transitions get no consistency evidence
    instead of a fabricated residual.
    """
    s = np.asarray(episode.state, dtype=np.float64)[:, list(state_idx)]
    a = np.asarray(episode.action, dtype=np.float64)[:, list(action_idx)]
    t = s.shape[0]
    pred = np.full_like(s, np.nan)
    obs = np.full_like(s, np.nan)
    valid = np.zeros(t, dtype=bool)
    if delay < 0:
        raise ValueError("control_delay_frames must be >= 0")
    upper = t - delay if delay > 0 else t
    if upper <= 0:
        return pred, obs, valid

    idx = np.arange(upper)
    nxt = idx + delay
    if semantics == "absolute_position_target":
        pred[idx] = a[idx]
        obs[idx] = s[nxt]
    elif semantics == "delta_position":
        pred[idx] = a[idx]
        obs[idx] = s[nxt] - s[idx]
    elif semantics == "velocity":
        pred[idx] = a[idx]
        obs[idx] = (s[nxt] - s[idx]) / max(dt * max(delay, 1), 1e-12)
    else:
        return pred, obs, valid
    valid[idx] = True
    if delay == 0:
        # action[t] is claimed to describe state[t] itself; still a valid test.
        valid[:] = True
    return pred, obs, valid


def _finite_diff(arr: np.ndarray) -> np.ndarray:
    """Forward difference magnitude, last entry repeated to keep length."""
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[0] < 2:
        return np.zeros(a.shape[0])
    d = np.linalg.norm(np.diff(a, axis=0), axis=-1)
    return np.concatenate([d, d[-1:]])


def _normalized(x: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance, or zeros for a constant signal."""
    v = np.asarray(x, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    sd = v.std()
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def local_time_offset(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    *,
    max_offset: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame best lag between two 1-D signals and its correlation.

    For each ``t`` a window of ``window`` frames is cross-correlated against
    ``signal_b`` shifted by every offset in ``[-max_offset, max_offset]``; the
    offset with the highest correlation is reported. A genuine
    action-observation shift shows up as a consistent non-zero offset, while
    noise gives a near-zero correlation and therefore no evidence.
    """
    a = _normalized(signal_a)
    b = _normalized(signal_b)
    n = a.size
    best = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    if n < 4 or max_offset < 0:
        return best, corr
    half = max(2, window // 2)
    offsets = np.arange(-max_offset, max_offset + 1)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        seg_a = a[lo:hi]
        if seg_a.size < 4 or seg_a.std() < 1e-12:
            continue
        scores = np.full(offsets.size, -np.inf)
        for i, off in enumerate(offsets):
            jlo, jhi = lo + off, hi + off
            if jlo < 0 or jhi > n:
                continue
            seg_b = b[jlo:jhi]
            if seg_b.size != seg_a.size or seg_b.std() < 1e-12:
                continue
            scores[i] = float(
                np.mean(
                    (seg_a - seg_a.mean())
                    * (seg_b - seg_b.mean())
                )
                / (seg_a.std() * seg_b.std())
            )
        if not np.isfinite(scores).any():
            continue
        k = int(np.argmax(scores))
        best[t] = float(offsets[k])
        corr[t] = float(scores[k])
    return best, corr


def default_task_phase(num_frames: int, *, bins: int = 3) -> np.ndarray:
    """Episode-progress phase, used when no VLM/annotation phase is supplied.

    A crude but honest stand-in: it groups transitions by how far through the
    episode they are, which is enough to stop early approach frames from being
    compared against late fine-adjust frames. Recorded as
    ``task_phase_source="episode_progress"`` in the sidecar.
    """
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int16)
    progress = np.arange(num_frames, dtype=np.float64) / max(num_frames - 1, 1)
    return np.clip((progress * bins).astype(np.int16), 0, bins - 1)


def state_bins(state: np.ndarray, *, bins: int) -> np.ndarray:
    """Quantile bins of the state magnitude, for conditional outlier scaling."""
    mag = np.linalg.norm(np.asarray(state, dtype=np.float64), axis=-1)
    if bins <= 1 or mag.size == 0:
        return np.zeros(mag.size, dtype=np.int16)
    edges = np.quantile(mag, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(edges, mag, side="right").astype(np.int16)


class ReliabilityScorer:
    """Turns low-dimensional episode data into :math:`c_t`."""

    def __init__(
        self,
        cfg: ReliabilityConfig,
        schema: DatasetSchema,
        dataset_cfg: DatasetConfig,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.schema = schema
        self.dataset_cfg = dataset_cfg
        self.state_idx, self.action_idx = state_action_index_map(schema, dataset_cfg)
        self._warnings: list[str] = []
        if not self.state_idx:
            self._warnings.append(
                "no comparable state/action dimensions; the state-consistency and "
                "synchronization components are unavailable"
            )

    # -- stage 1 ----------------------------------------------------------- #

    def extract(
        self,
        episode: LowDimEpisode,
        *,
        motion: np.ndarray | None = None,
        motion_source: str = "state_delta",
        task_phase: np.ndarray | None = None,
    ) -> EpisodeReliabilityRaw:
        cfg = self.cfg
        n = episode.num_frames
        dt = 1.0 / float(self.schema.fps)

        pred, obs, valid = predict_next_state(
            episode,
            semantics=self.dataset_cfg.action_semantics,
            delay=cfg.control_delay_frames,
            state_idx=self.state_idx,
            action_idx=self.action_idx,
            dt=dt,
        )
        residual = np.full(n, np.nan)
        if valid.any():
            diff = pred[valid] - obs[valid]
            residual[valid] = np.linalg.norm(diff, axis=-1)

        action = np.asarray(episode.action, dtype=np.float64)
        action_delta = _finite_diff(action)
        jerk = np.full(n, np.nan)
        jerk_valid = np.zeros(n, dtype=bool)
        if n >= 3:
            second = np.linalg.norm(np.diff(action, n=2, axis=0), axis=-1)
            jerk[1 : 1 + second.size] = second
            jerk_valid[1 : 1 + second.size] = True

        if motion is None:
            motion = _finite_diff(np.asarray(episode.state, dtype=np.float64))
            motion_source = "state_delta"
        motion = np.asarray(motion, dtype=np.float64).reshape(-1)
        if motion.size != n:
            raise ValueError(
                f"motion signal has {motion.size} entries, expected {n} "
                f"for episode {episode.episode_index}"
            )

        best_offset, offset_corr = local_time_offset(
            action_delta,
            motion,
            max_offset=cfg.sync_max_offset_frames,
            window=cfg.sync_window_frames,
        )

        phase = (
            np.asarray(task_phase, dtype=np.int16).reshape(-1)
            if task_phase is not None
            else default_task_phase(n)
        )
        if phase.size != n:
            raise ValueError(
                f"task_phase has {phase.size} entries, expected {n} "
                f"for episode {episode.episode_index}"
            )

        return EpisodeReliabilityRaw(
            episode_index=episode.episode_index,
            frame_index=np.asarray(episode.frame_index, dtype=np.int64),
            residual=residual,
            residual_valid=valid,
            action_delta=action_delta,
            motion=motion,
            motion_source=motion_source,
            jerk=jerk,
            jerk_valid=jerk_valid,
            best_offset=best_offset,
            offset_correlation=offset_corr,
            task_phase=phase,
            state_bin=state_bins(episode.state, bins=cfg.outlier_state_bins),
        )

    # -- stage 2 ----------------------------------------------------------- #

    def score(
        self,
        episodes: Sequence[LowDimEpisode],
        *,
        motion: Mapping[int, np.ndarray] | None = None,
        motion_source: str = "state_delta",
        task_phase: Mapping[int, np.ndarray] | None = None,
    ) -> ReliabilityResult:
        if not episodes:
            raise ValueError("no episodes to score")
        cfg = self.cfg
        raws = [
            self.extract(
                ep,
                motion=(motion or {}).get(ep.episode_index),
                motion_source=motion_source if motion else "state_delta",
                task_phase=(task_phase or {}).get(ep.episode_index),
            )
            for ep in episodes
        ]

        residual = np.concatenate([r.residual for r in raws])
        residual_valid = np.concatenate([r.residual_valid for r in raws])
        jerk = np.concatenate([r.jerk for r in raws])
        jerk_valid = np.concatenate([r.jerk_valid for r in raws])
        best_offset = np.concatenate([r.best_offset for r in raws])
        offset_corr = np.concatenate([r.offset_correlation for r in raws])
        phase = np.concatenate([r.task_phase for r in raws])
        sbin = np.concatenate([r.state_bin for r in raws])
        episode_index = np.concatenate(
            [np.full(r.num_frames, r.episode_index, dtype=np.int64) for r in raws]
        )
        frame_index = np.concatenate([r.frame_index for r in raws])
        # Keys come from each episode's *own* dataset_id, not the scorer's: a
        # corrupted variant and its clean parent share episode/frame indices, so
        # stamping one scorer-wide id would collide the two identities.
        keys = np.concatenate(
            [
                make_keys(
                    ep.dataset_id,
                    np.full(ep.num_frames, ep.episode_index, dtype=np.int64),
                    ep.frame_index,
                )
                for ep in episodes
            ]
        )
        dataset_ids = np.concatenate(
            [np.full(ep.num_frames, ep.dataset_id, dtype=object) for ep in episodes]
        )
        n = residual.size

        semantics_ok = self.dataset_cfg.action_semantics != "unknown" and bool(
            self.state_idx
        )

        # --- state consistency -------------------------------------------- #
        cons = np.zeros(n)
        cons_avail = residual_valid & np.isfinite(residual) & semantics_ok
        res_scale = (
            robust_scale(residual[cons_avail]) if cons_avail.any() else 1.0
        )
        if cons_avail.any():
            med = float(np.median(residual[cons_avail]))
            # A residual at the dataset median is a *normal* tracking error for
            # this controller, not evidence of a bad record; only excess over
            # the median is penalised.
            excess = np.maximum(residual - med, 0.0) / (
                max(cfg.consistency_tolerance, 1e-9) * res_scale
            )
            cons = np.where(cons_avail, np.exp(-np.clip(excess, 0.0, 50.0)), 0.0)

        # --- synchronization ---------------------------------------------- #
        # The reference offset is a property of the *dataset*, not of a frame:
        # frames later, so "aligned" means "at the population's own offset".
        # Pooling across episodes is what makes a shift applied to a subset of
        # episodes detectable; a shift applied uniformly to every episode moves
        # the reference with it and is, by construction, not detectable from
        # internal evidence alone (documented in README).
        sync_avail = (
            np.isfinite(best_offset)
            & np.isfinite(offset_corr)
            & (offset_corr > cfg.sync_min_correlation)
        )
        if cfg.sync_reference_mode == "zero":
            reference_offset = 0.0
            reference_source = "zero"
        elif cfg.sync_reference_mode == "fixed":
            reference_offset = float(cfg.sync_reference_offset)
            reference_source = "fixed"
        elif sync_avail.any():
            vals = best_offset[sync_avail]
            uniq, counts = np.unique(vals, return_counts=True)
            reference_offset = float(uniq[int(np.argmax(counts))])
            reference_source = "auto_modal"
        else:
            reference_offset = 0.0
            reference_source = "auto_no_evidence"

        sync = np.zeros(n)
        if sync_avail.any():
            deviation = np.abs(np.nan_to_num(best_offset) - reference_offset)
            penalty = cfg.sync_offset_penalty * deviation
            sync = np.where(sync_avail, np.clip(1.0 - penalty, 0.0, 1.0), 0.0)

        # --- residual outlier (grouped robust z) --------------------------- #
        outlier = np.zeros(n)
        outlier_avail = cons_avail.copy()
        group_info: dict[str, Any] = {"groups": 0, "fallback_to_global": 0}
        if outlier_avail.any():
            keys_for_group: list[np.ndarray] = []
            if "task_phase" in cfg.outlier_group_by:
                keys_for_group.append(phase.astype(np.int64))
            if "state_bin" in cfg.outlier_group_by:
                keys_for_group.append(sbin.astype(np.int64))
            if "episode" in cfg.outlier_group_by:
                keys_for_group.append(episode_index)
            if keys_for_group:
                stacked = np.stack(keys_for_group, axis=1)
                _, group_id = np.unique(stacked, axis=0, return_inverse=True)
            else:
                group_id = np.zeros(n, dtype=np.int64)
            z = np.full(n, np.nan)
            global_scale = robust_scale(residual[outlier_avail])
            global_med = float(np.median(residual[outlier_avail]))
            for g in np.unique(group_id):
                sel = (group_id == g) & outlier_avail
                count = int(sel.sum())
                if count == 0:
                    continue
                group_info["groups"] += 1
                if count < cfg.outlier_min_group_size:
                    # Too few comparable transitions for a trustworthy local
                    # scale: use the dataset-wide scale and record that we did.
                    group_info["fallback_to_global"] += 1
                    z[sel] = np.abs(residual[sel] - global_med) / global_scale
                else:
                    z[sel] = robust_z(residual[sel])
            outlier = np.where(
                outlier_avail,
                np.clip(1.0 - np.nan_to_num(z, nan=0.0) / max(cfg.outlier_z_cutoff, 1e-9), 0.0, 1.0),
                0.0,
            )

        # --- continuity (auxiliary) ---------------------------------------- #
        cont = np.zeros(n)
        cont_avail = jerk_valid & np.isfinite(jerk)
        if cont_avail.any():
            jerk_scale = robust_scale(jerk[cont_avail])
            jerk_med = float(np.median(jerk[cont_avail]))
            excess = np.maximum(jerk - jerk_med, 0.0) / (
                max(cfg.continuity_tolerance, 1e-9) * jerk_scale
            )
            base = np.clip(1.0 - excess, 0.0, 1.0)
            # Relief: a sharp command that the state actually followed is a
            # legitimate contact/reversal/release event, not a recording error.
            explained = cons if cons_avail.any() else np.zeros(n)
            relieved = base + cfg.continuity_explained_relief * (1.0 - base) * explained
            cont = np.where(cont_avail, np.clip(relieved, 0.0, 1.0), 0.0)

        components = {
            "state_consistency": cons.astype(np.float32),
            "synchronization": sync.astype(np.float32),
            "residual_outlier": outlier.astype(np.float32),
            "continuity": cont.astype(np.float32),
        }
        availability = {
            "state_consistency": cons_avail,
            "synchronization": sync_avail,
            "residual_outlier": outlier_avail,
            "continuity": cont_avail,
        }

        combined = combine_components(
            components,
            availability,
            cfg.weights(),
            policy=cfg.missing_policy,
            missing_value=cfg.missing_value,
            all_missing_value=cfg.all_missing_value,
            mode=cfg.combine,
            label="reliability",
        )

        warnings = list(self._warnings)
        if not semantics_ok:
            warnings.append(
                f"action_semantics={self.dataset_cfg.action_semantics!r}: the "
                "state-consistency and residual-outlier components are unavailable "
                "and their weight is renormalised away"
            )

        return ReliabilityResult(
            keys=keys,
            episode_index=episode_index,
            frame_index=frame_index,
            dataset_ids=dataset_ids,
            c=combined.value,
            components={f"c_{k}": v for k, v in components.items()},
            availability={f"avail_{k}": v for k, v in availability.items()},
            raw={
                "residual": residual.astype(np.float32),
                "jerk": jerk.astype(np.float32),
                "action_delta": np.concatenate([r.action_delta for r in raws]).astype(
                    np.float32
                ),
                "motion": np.concatenate([r.motion for r in raws]).astype(np.float32),
                "best_offset": best_offset.astype(np.float32),
                "offset_correlation": offset_corr.astype(np.float32),
                "task_phase": phase.astype(np.int16),
                "state_bin": sbin.astype(np.int16),
            },
            unavailable=combined.all_missing,
            n_components_used=combined.n_available,
            meta={
                "weights": cfg.weights(),
                "combine": cfg.combine,
                "missing_policy": cfg.missing_policy,
                "action_semantics": self.dataset_cfg.action_semantics,
                "control_delay_frames": cfg.control_delay_frames,
                "state_to_action_index_map": {
                    "state": list(self.state_idx),
                    "action": list(self.action_idx),
                    "source": (
                        "config"
                        if self.dataset_cfg.state_to_action_indices is not None
                        else "identity_prefix"
                    ),
                },
                "residual_robust_scale": float(res_scale),
                "residual_median": (
                    float(np.median(residual[cons_avail])) if cons_avail.any() else None
                ),
                "motion_source": raws[0].motion_source if raws else None,
                "sync_reference_offset_frames": reference_offset,
                "sync_reference_source": reference_source,
                "task_phase_source": "provided" if task_phase else "episode_progress",
                "outlier_grouping": {
                    "group_by": list(cfg.outlier_group_by),
                    "state_bins": cfg.outlier_state_bins,
                    "min_group_size": cfg.outlier_min_group_size,
                    **group_info,
                },
                "component_summary": combined.summary(),
                "warnings": warnings,
            },
        )


def offset_report(result: ReliabilityResult) -> dict[str, Any]:
    """Aggregate detected action/observation time offsets.

    A dataset-wide modal offset of k frames is the signature of a temporal
    shift between the action stream and the observations.
    """
    off = np.asarray(result.raw.get("best_offset", np.array([])), dtype=np.float64)
    corr = np.asarray(result.raw.get("offset_correlation", np.array([])), dtype=np.float64)
    ok = np.isfinite(off) & np.isfinite(corr)
    if not ok.any():
        return {"status": "no_evidence"}
    vals = off[ok]
    counts = {int(v): int((vals == v).sum()) for v in np.unique(vals)}
    modal = max(counts, key=lambda k: counts[k])
    return {
        "status": "ok",
        "n_with_evidence": int(ok.sum()),
        "modal_offset_frames": modal,
        "modal_fraction": round(counts[modal] / int(ok.sum()), 4),
        "mean_offset_frames": round(float(vals.mean()), 4),
        "offset_histogram": counts,
        "mean_correlation": round(float(corr[ok].mean()), 4),
    }
