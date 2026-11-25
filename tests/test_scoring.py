"""Visibility and reliability scoring, including the missing-evidence policy."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from svdt.config import (
    ImageQualityConfig,
    ReliabilityConfig,
    SegmentationConfig,
    VisibilityConfig,
)
from svdt.scoring import image_quality as iq
from svdt.scoring.combine import combine_components, renormalized_weight_table
from svdt.scoring.reliability import (
    ReliabilityScorer,
    default_task_phase,
    local_time_offset,
    offset_report,
    predict_next_state,
    robust_scale,
)
from svdt.scoring.segmentation import make_segmentation_backend
from svdt.scoring.visibility import (
    VisibilityScorer,
    compensation_report,
    estimate_expected_area,
)


# --------------------------------------------------------------------------- #
# component combination and the missing policy
# --------------------------------------------------------------------------- #


def test_renormalize_drops_missing_components_and_rescales():
    components = {"a": np.array([1.0, 1.0]), "b": np.array([0.0, 0.0])}
    availability = {"a": np.array([True, True]), "b": np.array([True, False])}
    out = combine_components(
        components, availability, {"a": 0.5, "b": 0.5}, policy="renormalize"
    )
    assert out.value[0] == pytest.approx(0.5)  # both present
    assert out.value[1] == pytest.approx(1.0)  # b dropped, a renormalised to 1
    assert out.n_available.tolist() == [2, 1]


def test_missing_evidence_never_becomes_a_high_score():
    """Everything missing must yield the configured value plus a flag."""
    components = {"a": np.array([0.9]), "b": np.array([0.9])}
    availability = {"a": np.array([False]), "b": np.array([False])}
    out = combine_components(
        components,
        availability,
        {"a": 0.5, "b": 0.5},
        policy="renormalize",
        all_missing_value=0.0,
    )
    assert out.value[0] == 0.0
    assert bool(out.all_missing[0])


def test_floor_policy_substitutes_the_missing_value():
    out = combine_components(
        {"a": np.array([1.0]), "b": np.array([1.0])},
        {"a": np.array([True]), "b": np.array([False])},
        {"a": 0.5, "b": 0.5},
        policy="floor",
        missing_value=0.0,
    )
    assert out.value[0] == pytest.approx(0.5)


def test_error_policy_refuses_partial_evidence():
    with pytest.raises(ValueError, match="missing_policy='error'"):
        combine_components(
            {"a": np.array([1.0])},
            {"a": np.array([False])},
            {"a": 1.0},
            policy="error",
        )


def test_components_outside_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"must lie in \[0,1\]"):
        combine_components(
            {"a": np.array([1.4])}, {"a": np.array([True])}, {"a": 1.0}
        )


def test_renormalized_weight_table():
    table = renormalized_weight_table({"a": 0.3, "b": 0.2, "c": 0.5}, ["a", "b"])
    assert table["a"] == pytest.approx(0.6)
    assert table["b"] == pytest.approx(0.4)
    assert table["c"] == 0.0


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #


def test_null_backend_reports_no_evidence_rather_than_zero_area():
    backend = make_segmentation_backend(
        SegmentationConfig(backend="null"), dataset_id="d"
    )
    results = backend.segment_batch(np.zeros((2, 8, 8, 3), np.uint8), "cam")
    for result in results:
        for role in backend.roles:
            assert result.get(role).evidence is False


def test_unknown_segmentation_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown segmentation backend"):
        make_segmentation_backend(SegmentationConfig(backend="magic"), dataset_id="d")


# --------------------------------------------------------------------------- #
# image quality
# --------------------------------------------------------------------------- #


def test_blur_lowers_the_sharpness_subscore():
    import cv2

    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(1, 48, 48, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp[0], (9, 9), 3.0)[None]
    cfg = ImageQualityConfig()
    raw_sharp, _ = iq.measure_batch(sharp, np.array([0]), cfg)
    raw_blur, _ = iq.measure_batch(blurred, np.array([0]), cfg)
    reference = iq.sharpness_reference(raw_sharp.laplacian_var, cfg)
    assert (
        iq.normalize(raw_blur, cfg, reference=reference).sharpness[0]
        < iq.normalize(raw_sharp, cfg, reference=reference).sharpness[0]
    )


def test_clipped_exposure_lowers_the_exposure_subscore():
    cfg = ImageQualityConfig()
    mid = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
    dark = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    reference = 1.0
    raw_mid, _ = iq.measure_batch(mid, np.array([0]), cfg)
    raw_dark, _ = iq.measure_batch(dark, np.array([0]), cfg)
    assert iq.normalize(raw_mid, cfg, reference=reference).exposure[0] == pytest.approx(1.0)
    assert iq.normalize(raw_dark, cfg, reference=reference).exposure[0] == pytest.approx(0.0)


def test_first_frame_motion_is_backfilled_not_assumed_still():
    cfg = ImageQualityConfig()
    frames = np.stack(
        [np.full((16, 16, 3), v, dtype=np.uint8) for v in (10, 60, 110)]
    )[None].reshape(3, 16, 16, 3)
    raw, _ = iq.measure_batch(frames, np.arange(3), cfg)
    assert raw.motion_valid.tolist() == [False, True, True]
    scores = iq.normalize(raw, cfg, reference=1.0)
    assert np.isfinite(scores.camera_motion).all()


def test_sharpness_reference_uses_a_percentile_not_the_max():
    cfg = ImageQualityConfig(sharpness_reference_percentile=90.0)
    values = np.concatenate([np.full(99, 1.0), np.array([1000.0])])
    assert iq.sharpness_reference(values, cfg) < 10.0


# --------------------------------------------------------------------------- #
# visibility
# --------------------------------------------------------------------------- #


def test_expected_area_is_unavailable_without_enough_detections():
    cfg = SegmentationConfig(min_frames_for_expected_area=8)
    areas = np.array([100.0, 100.0, np.nan])
    evidence = np.array([True, True, False])
    value, provenance = estimate_expected_area(areas, evidence, cfg, "object")
    assert value is None and provenance == "insufficient_detections"


def test_configured_expected_area_takes_precedence():
    cfg = SegmentationConfig(expected_area_px={"object": 250.0})
    value, provenance = estimate_expected_area(
        np.array([10.0]), np.array([True]), cfg, "object"
    )
    assert value == 250.0 and provenance == "config"


def test_visibility_without_segmentation_evidence_flags_missing(source):
    cfg = VisibilityConfig(segmentation=SegmentationConfig(backend="null"))
    scorer = VisibilityScorer(cfg, source.schema)
    raws = scorer.collect(source, [0], stride=4)
    result = scorer.score(raws)
    assert result.missing_evidence.all()
    # Only the image-quality component survives, so v_t equals it exactly.
    assert result.v == pytest.approx(result.components["q_image_quality"], abs=1e-6)
    assert result.meta["lambdas"]["object"] == 0.30  # config untouched


def test_visibility_scores_are_in_range_and_components_preserved(source):
    cfg = VisibilityConfig(segmentation=SegmentationConfig(backend="null"))
    scorer = VisibilityScorer(cfg, source.schema)
    result = scorer.score(scorer.collect(source, [0, 1], stride=2))
    assert 0.0 <= result.v.min() and result.v.max() <= 1.0
    for name in ("q_object", "q_gripper", "q_segmentation", "q_image_quality"):
        assert name in result.components
        assert 0.0 <= result.components[name].min() <= result.components[name].max() <= 1.0
    # Raw measurements are kept for sensitivity analysis.
    assert any(k.startswith("raw_laplacian_var") for k in result.raw)


def test_visibility_keys_carry_each_sources_dataset_id(source, corruption_specs):
    from svdt.data.corrupt import CorruptedFrameSource

    variant = CorruptedFrameSource(source, corruption_specs[1])
    cfg = VisibilityConfig(segmentation=SegmentationConfig(backend="null"))
    scorer = VisibilityScorer(cfg, source.schema)
    raws = scorer.collect(source, source.episode_indices(), stride=5)
    raws += scorer.collect(variant, variant.episode_indices(), stride=5)
    result = scorer.score(raws)
    ids = {str(k).split("|")[0] for k in result.keys}
    assert source.dataset_id in ids and variant.dataset_id in ids
    assert len(set(result.keys)) == len(result.keys), "identities must be unique"


def test_compensation_report_quantifies_component_masking(source):
    cfg = VisibilityConfig(segmentation=SegmentationConfig(backend="null"))
    scorer = VisibilityScorer(cfg, source.schema)
    result = scorer.score(scorer.collect(source, [0], stride=4))
    report = compensation_report(result)
    assert {"occluded_but_sharp", "visible_but_degraded", "mean_v"} <= set(report)


# --------------------------------------------------------------------------- #
# reliability
# --------------------------------------------------------------------------- #


def test_robust_scale_is_not_inflated_by_an_outlier():
    """A single extreme value must not normalise itself away."""
    base = np.ones(100)
    with_outlier = np.concatenate([base, np.array([1e6])])
    scale = robust_scale(with_outlier)
    assert scale < 1.0
    # The outlier must still read as extreme after scaling.
    assert abs(1e6 - np.median(with_outlier)) / scale > 100


def test_robust_scale_matches_mad_for_normal_data():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 3.0, size=5000)
    assert robust_scale(values) == pytest.approx(3.0, rel=0.1)


def test_predict_next_state_marks_the_episode_tail_invalid(source):
    episode = source.read_lowdim(0)
    _, _, valid = predict_next_state(
        episode,
        semantics="absolute_position_target",
        delay=1,
        state_idx=[0, 1],
        action_idx=[0, 1],
        dt=0.1,
    )
    assert valid[:-1].all() and not valid[-1]


def test_unknown_action_semantics_make_consistency_unavailable(source, dataset_cfg):
    cfg = copy.deepcopy(dataset_cfg)
    cfg.action_semantics = "unknown"
    scorer = ReliabilityScorer(ReliabilityConfig(), source.schema, cfg)
    result = scorer.score([source.read_lowdim(0)])
    assert not result.availability["avail_state_consistency"].any()
    assert any("unavailable" in w for w in result.meta["warnings"])
    # The remaining components still produce a usable score.
    assert np.isfinite(result.c).all()


def test_reliability_scores_are_in_range(source, dataset_cfg):
    scorer = ReliabilityScorer(ReliabilityConfig(), source.schema, dataset_cfg)
    result = scorer.score([source.read_lowdim(e) for e in source.episode_indices()])
    assert 0.0 <= result.c.min() and result.c.max() <= 1.0
    for name, values in result.components.items():
        assert 0.0 <= values.min() <= values.max() <= 1.0, name


def test_action_outliers_lower_the_reliability_score(source, dataset_cfg):
    clean = [source.read_lowdim(e) for e in source.episode_indices()]
    dirty = copy.deepcopy(clean[-1])
    dirty.dataset_id = "synthetic__outlier"
    rng = np.random.default_rng(0)
    hit = rng.choice(dirty.num_frames, size=4, replace=False)
    dirty.action[hit] += 200.0

    scorer = ReliabilityScorer(ReliabilityConfig(), source.schema, dataset_cfg)
    result = scorer.score(clean + [dirty])
    is_dirty = np.array([str(k).startswith("synthetic__outlier") for k in result.keys])
    corrupted_frames = np.isin(result.frame_index, dirty.frame_index[hit]) & is_dirty
    assert result.c[corrupted_frames].mean() < result.c[~is_dirty].mean()


def test_synchronization_reference_is_estimated_not_assumed_zero():
    """Real datasets lag; penalising deviation from 0 would flag clean data."""
    n = 200
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    lagged = np.roll(signal, 2)
    offsets, corr = local_time_offset(signal, lagged, max_offset=5, window=40)
    ok = np.isfinite(offsets) & (corr > 0.5)
    assert ok.any()
    # lagged[t + 2] == signal[t], so the best alignment offset is +2 -- a real
    # dataset's non-zero lag, which is why the reference must be estimated.
    assert int(np.median(offsets[ok])) == 2

    aligned_offsets, aligned_corr = local_time_offset(
        signal, signal, max_offset=5, window=40
    )
    good = np.isfinite(aligned_offsets) & (aligned_corr > 0.5)
    assert int(np.median(aligned_offsets[good])) == 0


def test_offset_report_summarises_the_detected_lag(source, dataset_cfg):
    scorer = ReliabilityScorer(ReliabilityConfig(), source.schema, dataset_cfg)
    result = scorer.score([source.read_lowdim(e) for e in source.episode_indices()])
    report = offset_report(result)
    assert report["status"] in {"ok", "no_evidence"}
    if report["status"] == "ok":
        assert "modal_offset_frames" in report


def test_default_task_phase_groups_by_episode_progress():
    phases = default_task_phase(9, bins=3)
    assert phases.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_reliability_keys_carry_each_episodes_dataset_id(source, dataset_cfg):
    clean = source.read_lowdim(0)
    variant = copy.deepcopy(clean)
    variant.dataset_id = "synthetic__variant"
    scorer = ReliabilityScorer(ReliabilityConfig(), source.schema, dataset_cfg)
    result = scorer.score([clean, variant])
    ids = {str(k).split("|")[0] for k in result.keys}
    assert ids == {"synthetic", "synthetic__variant"}
    assert len(set(result.keys)) == len(result.keys)
