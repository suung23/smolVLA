"""Score sidecars, the sample weighter, VLM metadata, and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from svdt.config import (
    ReliabilityConfig,
    SegmentationConfig,
    VisibilityConfig,
    VLMConfig,
    WeightingConfig,
)
from svdt.eval.metrics import (
    action_chunk_error,
    action_mae,
    auprc,
    auroc,
    degradation_table,
    distribution_summary,
    seed_statistics,
)
from svdt.eval.rollout import check_rollout_support, rollout_placeholder
from svdt.eval.score_eval import evaluate_scores, hard_filter_tradeoff
from svdt.ids import make_keys
from svdt.scoring.reliability import ReliabilityScorer
from svdt.scoring.sidecar import (
    ScoreSidecar,
    build_sidecar,
    load_sidecar,
    save_sidecar,
    sidecar_is_current,
)
from svdt.scoring.visibility import VisibilityScorer
from svdt.vlm import (
    NullVLMAdapter,
    ReplayVLMAdapter,
    SchemaError,
    VLMCache,
    assert_not_label_source,
    make_vlm_adapter,
    phases_from_annotation,
    validate_annotation,
)
from svdt.weighting.sample_weighter import TrustSampleWeighter


def _sidecar(n: int = 6, dataset_id: str = "d") -> ScoreSidecar:
    keys = make_keys(dataset_id, [0] * n, range(n))
    rng = np.random.default_rng(0)
    return ScoreSidecar(
        dataset_id=dataset_id,
        keys=keys,
        episode_index=np.zeros(n, dtype=np.int64),
        frame_index=np.arange(n, dtype=np.int64),
        v=rng.uniform(0, 1, n).astype(np.float32),
        c=rng.uniform(0, 1, n).astype(np.float32),
        components={"q_object": rng.uniform(0, 1, n).astype(np.float32)},
        flags={"missing_evidence": np.zeros(n, dtype=bool)},
        raw={"residual": rng.normal(size=n).astype(np.float32)},
        meta={"fingerprint": "abc123"},
    )


# --------------------------------------------------------------------------- #
# sidecar
# --------------------------------------------------------------------------- #


def test_sidecar_round_trips_through_disk(tmp_path):
    original = _sidecar()
    save_sidecar(original, tmp_path, input_fingerprint="abc123")
    restored = load_sidecar(tmp_path, "d")
    assert restored.keys.tolist() == original.keys.tolist()
    assert restored.v == pytest.approx(original.v)
    assert restored.c == pytest.approx(original.c)
    assert restored.components["q_object"] == pytest.approx(original.components["q_object"])
    assert restored.flags["missing_evidence"].tolist() == [False] * 6
    assert "residual" in restored.raw


def test_sidecar_rejects_duplicate_identities():
    keys = np.array(["d|000000|0000000", "d|000000|0000000"], dtype=object)
    with pytest.raises(ValueError, match="duplicate transition key"):
        ScoreSidecar(
            dataset_id="d",
            keys=keys,
            episode_index=np.zeros(2, dtype=np.int64),
            frame_index=np.zeros(2, dtype=np.int64),
        )


def test_sidecar_fingerprint_drives_idempotency(tmp_path):
    sidecar = _sidecar()
    save_sidecar(sidecar, tmp_path, input_fingerprint="fp-1")
    assert sidecar_is_current(tmp_path, "d", "fp-1")
    assert not sidecar_is_current(tmp_path, "d", "fp-2")


def test_sidecar_aligns_by_key_not_row_order():
    sidecar = _sidecar()
    shuffled = list(reversed(sidecar.keys.tolist()))
    positions = sidecar.align_to(shuffled)
    assert positions.tolist() == list(reversed(range(len(sidecar))))


def test_missing_sidecar_names_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="svdt score"):
        load_sidecar(tmp_path, "absent")


def test_build_sidecar_outer_joins_and_flags_each_score(source, dataset_cfg):
    """Visibility is often strided, so the two scorers cover different frames."""
    vis_scorer = VisibilityScorer(
        VisibilityConfig(segmentation=SegmentationConfig(backend="null")),
        source.schema,
    )
    visibility = vis_scorer.score(vis_scorer.collect(source, [0], stride=4))
    reliability = ReliabilityScorer(
        ReliabilityConfig(), source.schema, dataset_cfg
    ).score([source.read_lowdim(0)])

    sidecar = build_sidecar("d", visibility=visibility, reliability=reliability)
    assert len(sidecar) == len(reliability)  # the superset
    assert sidecar.flags["v_present"].sum() == len(visibility)
    assert sidecar.flags["c_present"].all()
    # Transitions without a v keep their c and are marked, never defaulted.
    missing_v = ~sidecar.flags["v_present"]
    assert missing_v.any()
    assert np.isnan(sidecar.v[missing_v]).all()
    assert np.isfinite(sidecar.c[missing_v]).all()


def test_sidecar_distribution_summary():
    summary = _sidecar().summary()
    assert summary["n_transitions"] == 6
    assert "v" in summary["distributions"] and "c" in summary["distributions"]
    assert len(summary["distributions"]["v"]["histogram"]) == 10


# --------------------------------------------------------------------------- #
# sample weighter
# --------------------------------------------------------------------------- #


def _batch(episodes, frames, horizon=3, pad=None):
    n = len(episodes)
    return {
        "episode_index": np.asarray(episodes),
        "frame_index": np.asarray(frames),
        "action": np.zeros((n, horizon, 2)),
        "action_is_pad": (
            np.zeros((n, horizon), dtype=bool) if pad is None else np.asarray(pad)
        ),
    }


def test_weighter_resolves_weights_by_transition_identity():
    sidecar = _sidecar(n=6)
    cfg = WeightingConfig(method="combined", alpha=0.5, w_min=0.0)
    weighter = TrustSampleWeighter(sidecar, cfg, dataset_id="d", chunk_size=3)
    weights, stats = weighter.compute_batch_weights(_batch([0, 0], [4, 1]))
    expected = 0.5 * sidecar.v[[4, 1]] + 0.5 * sidecar.c[[4, 1]]
    assert np.asarray(weights) == pytest.approx(expected, abs=1e-5)
    assert stats["n_missing_score"] == 0


def test_weighter_requires_identity_columns():
    weighter = TrustSampleWeighter(_sidecar(), WeightingConfig(), dataset_id="d")
    with pytest.raises(KeyError, match="cannot fall back to row order"):
        weighter.compute_batch_weights({"action": np.zeros((2, 3, 2))})


def test_weighter_zeroes_fully_padded_samples():
    sidecar = _sidecar(n=6)
    weighter = TrustSampleWeighter(
        sidecar, WeightingConfig(method="combined"), dataset_id="d", chunk_size=3
    )
    pad = np.array([[True, True, True], [False, False, False]])
    weights, _ = weighter.compute_batch_weights(_batch([0, 0], [0, 1], pad=pad))
    assert float(np.asarray(weights)[0]) == 0.0
    assert float(np.asarray(weights)[1]) > 0.0


def test_weighter_chunk_min_uses_the_chunk_not_just_the_anchor():
    sidecar = _sidecar(n=6)
    cfg = WeightingConfig(method="combined", chunk_alignment="chunk_min", w_min=0.0)
    weighter = TrustSampleWeighter(sidecar, cfg, dataset_id="d", chunk_size=3)
    weights, _ = weighter.compute_batch_weights(_batch([0], [1], horizon=3))
    trust = 0.5 * sidecar.v + 0.5 * sidecar.c
    assert float(np.asarray(weights)[0]) == pytest.approx(float(trust[1:4].min()), abs=1e-5)


def test_weighter_raises_when_the_sidecar_does_not_match_the_dataset():
    weighter = TrustSampleWeighter(
        _sidecar(n=4),
        WeightingConfig(method="combined"),
        dataset_id="d",
        max_missing_fraction=0.25,
    )
    with pytest.raises(KeyError, match="not the same variant"):
        weighter.compute_batch_weights(_batch([9, 9, 9, 9], [0, 1, 2, 3]))


def test_weighter_stats_report_applied_transitions():
    weighter = TrustSampleWeighter(
        _sidecar(), WeightingConfig(method="combined"), dataset_id="d"
    )
    weighter.compute_batch_weights(_batch([0, 0], [0, 1]))
    stats = weighter.get_stats()
    assert stats["samples_seen"] == 2
    assert stats["sidecar_dataset_id"] == "d"
    assert "retained_fraction" in stats


# --------------------------------------------------------------------------- #
# VLM
# --------------------------------------------------------------------------- #


def _annotation(**patch):
    base = {
        "version": 1,
        "dataset_id": "d",
        "episode_index": 0,
        "model": "test-vlm",
        "prompt_version": "v1",
        "phases": [
            {"name": "approach", "start_frame": 0, "end_frame": 4},
            {"name": "push", "start_frame": 5, "end_frame": 9},
        ],
        "events": [{"kind": "contact", "frame": 5, "confidence": 0.8}],
        "overall_confidence": 0.7,
    }
    base.update(patch)
    return base


def test_valid_annotation_passes():
    assert validate_annotation(_annotation(), dataset_id="d", episode_index=0)


def test_annotation_rejects_unknown_fields():
    with pytest.raises(SchemaError, match="unexpected key"):
        validate_annotation(_annotation(surprise=1))


def test_annotation_rejects_overlapping_phases():
    with pytest.raises(SchemaError, match="phases overlap"):
        validate_annotation(
            _annotation(
                phases=[
                    {"name": "a", "start_frame": 0, "end_frame": 5},
                    {"name": "b", "start_frame": 3, "end_frame": 8},
                ]
            )
        )


def test_annotation_rejects_frames_past_the_episode_end():
    with pytest.raises(SchemaError, match="exceeds the episode length"):
        validate_annotation(_annotation(), num_frames=6)


def test_annotation_rejects_an_unknown_event_kind():
    with pytest.raises(SchemaError, match="must be one of"):
        validate_annotation(_annotation(events=[{"kind": "teleport", "frame": 1}]))


def test_annotation_rejects_a_mismatched_episode():
    with pytest.raises(SchemaError, match="is for episode"):
        validate_annotation(_annotation(), episode_index=3)


def test_vlm_output_cannot_become_a_reliability_label_by_default():
    with pytest.raises(PermissionError, match="must not be used as a reliability label"):
        assert_not_label_source(VLMConfig())
    assert_not_label_source(VLMConfig(allow_as_reliability_label=True))


def test_phases_from_annotation_marks_uncovered_frames_distinctly():
    phases, source = phases_from_annotation(_annotation(), num_frames=12)
    assert source == "vlm"
    assert phases[:5].tolist() == [0] * 5
    assert phases[5:10].tolist() == [1] * 5
    assert phases[10:].tolist() == [-1, -1]  # outside every declared phase


def test_core_pipeline_runs_without_a_vlm():
    adapter = make_vlm_adapter(VLMConfig(enabled=False))
    assert isinstance(adapter, NullVLMAdapter)
    assert adapter.annotate_episode("d", 0, num_frames=10) is None
    phases, source = phases_from_annotation(None, num_frames=4)
    assert source == "none" and phases.tolist() == [0, 0, 0, 0]


def test_vlm_cache_key_includes_the_prompt_version(tmp_path):
    cache = VLMCache(tmp_path)
    assert cache.key("d", 0, "m", "v1") != cache.key("d", 0, "m", "v2")
    cache.put(_annotation())
    assert cache.get("d", 0, "test-vlm", "v1") is not None
    assert cache.get("d", 0, "test-vlm", "v2") is None  # prompt change invalidates


def test_replay_adapter_validates_what_it_reads(tmp_path):
    from svdt.io_utils import write_json

    write_json(tmp_path / "d" / "ep000000.json", _annotation(events=[{"kind": "nope", "frame": 1}]))
    adapter = ReplayVLMAdapter(tmp_path, strict=True)
    with pytest.raises(SchemaError):
        adapter.annotate_episode("d", 0, num_frames=12)
    assert ReplayVLMAdapter(tmp_path, strict=False).annotate_episode(
        "d", 0, num_frames=12
    ) is None


def test_unknown_vlm_adapter_is_rejected():
    with pytest.raises(ValueError, match="unknown vlm adapter"):
        make_vlm_adapter(VLMConfig(enabled=True, adapter="gpt"))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_auroc_is_one_for_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    positive = np.array([True, True, False, False])
    assert auroc(scores, positive, lower_is_positive=True) == 1.0
    assert auroc(scores, positive, lower_is_positive=False) == 0.0


def test_auroc_is_half_for_all_ties():
    assert auroc(np.full(6, 0.5), np.array([True] * 3 + [False] * 3)) == pytest.approx(0.5)


def test_auroc_is_none_without_both_classes():
    assert auroc(np.array([0.1, 0.2]), np.array([True, True])) is None


def test_auprc_rewards_ranking_positives_first():
    scores = np.array([0.1, 0.2, 0.9])
    positive = np.array([True, True, False])
    assert auprc(scores, positive) == pytest.approx(1.0)


def test_action_mae_ignores_padding():
    pred = np.zeros((2, 3, 2))
    target = np.zeros((2, 3, 2))
    target[0, 2] = 100.0  # padded
    pad = np.array([[False, False, True], [False, False, False]])
    assert action_mae(pred, target, pad)["action_mae"] == pytest.approx(0.0)


def test_action_chunk_error_reports_per_horizon_step():
    pred = np.zeros((2, 3, 1))
    target = np.ones((2, 3, 1))
    out = action_chunk_error(pred, target)
    assert out["per_step_mae"] == [1.0, 1.0, 1.0]
    assert out["chunk_size"] == 3


def test_degradation_table_sign_convention():
    table = degradation_table(1.0, {"blur": 1.5}, higher_is_better=False)
    assert table["conditions"]["blur"]["relative_degradation"] == pytest.approx(0.5)
    table = degradation_table(0.8, {"blur": 0.4}, higher_is_better=True)
    assert table["conditions"]["blur"]["relative_degradation"] == pytest.approx(0.5)


def test_seed_statistics_reports_no_std_for_a_single_seed():
    assert seed_statistics({1000: 0.5})["std"] is None
    # Reported values are rounded to 6 dp for readable manifests.
    assert seed_statistics({1000: 0.4, 1001: 0.6})["std"] == pytest.approx(
        float(np.std([0.4, 0.6], ddof=1)), abs=1e-6
    )


def test_distribution_summary_handles_non_finite_values():
    summary = distribution_summary(np.array([0.1, np.nan, 0.9]))
    assert summary["n"] == 2 and summary["n_non_finite"] == 1


def test_evaluate_scores_against_injected_labels():
    sidecar = _sidecar(n=20)
    # Make the first ten clearly worse so the ranking is meaningful.
    sidecar.v[:10] = 0.1
    sidecar.c[:10] = 0.1
    sidecar.v[10:] = 0.9
    sidecar.c[10:] = 0.9
    labels = {str(k): i < 10 for i, k in enumerate(sidecar.keys)}
    out = evaluate_scores(sidecar, labels, weighting=WeightingConfig(method="combined"))
    assert out["n_corrupted"] == 10
    assert out["scores"]["v"]["auroc"] == 1.0
    assert out["scores"]["w"]["auroc"] == 1.0
    assert "weighting" in out and out["weighting"]["n_applied"] == 20


def test_hard_filter_tradeoff_reports_the_cost_of_discarding():
    sidecar = _sidecar(n=20)
    sidecar.v[:10] = 0.1
    sidecar.c[:10] = 0.1
    sidecar.v[10:] = 0.9
    sidecar.c[10:] = 0.9
    labels = {str(k): i < 10 for i, k in enumerate(sidecar.keys)}
    table = hard_filter_tradeoff(sidecar, labels, thresholds=(0.5,))
    row = table["rows"][0]
    assert row["corrupted_removed_fraction"] == pytest.approx(1.0)
    assert row["clean_removed_fraction"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# rollout support
# --------------------------------------------------------------------------- #


def test_cabinet_rollout_is_reported_as_not_available():
    """gym-aloha has no mobile-cabinet environment; claiming one would be false."""
    support = check_rollout_support("lerobot/aloha_mobile_cabinet")
    assert support.status == "not_available"
    assert not support.runnable
    assert "gym-aloha" in support.reason
    record = rollout_placeholder("lerobot/aloha_mobile_cabinet")
    assert record["rollout_success_rate"] is None
    assert record["rollout_success_rate_status"] == "not_available"


def test_unknown_dataset_rollout_is_not_invented():
    support = check_rollout_support("someone/unknown-dataset")
    assert support.status == "not_available"
    assert "no verified simulator" in support.reason


\n