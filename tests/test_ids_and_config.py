"""Stable transition identity and config validation."""

from __future__ import annotations

import pytest

from svdt.config import Config, VisibilityConfig, WeightingConfig, from_dict, parse_override
from svdt.ids import (
    TransitionID,
    align_to,
    build_index,
    make_keys,
    parent_episode_key,
    sanitize_dataset_id,
)


def test_transition_id_round_trip():
    tid = TransitionID("sample", 3, 17)
    assert tid.key == "sample|000003|0000017"
    assert TransitionID.parse(tid.key) == tid


def test_keys_sort_lexicographically_in_numeric_order():
    """Zero padding is what lets a plain string sort match numeric order."""
    keys = sorted(make_keys("d", [0, 0, 0], [2, 10, 100]))
    assert [TransitionID.parse(k).frame_index for k in keys] == [2, 10, 100]


def test_dataset_id_rejects_separator():
    with pytest.raises(ValueError, match="must not contain"):
        TransitionID("bad|id", 0, 0)


def test_sanitize_dataset_id():
    assert sanitize_dataset_id("lerobot/sample") == "lerobot__sample"
    with pytest.raises(ValueError):
        sanitize_dataset_id("  ")


def test_build_index_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate transition key"):
        build_index(["a|000000|0000000", "a|000000|0000000"])


def test_align_to_matches_by_identity_not_row_order():
    source = make_keys("d", [0, 0, 1], [0, 1, 0])
    # Deliberately shuffled relative to the source.
    target = ["d|000001|0000000", "d|000000|0000000"]
    assert align_to(target, source).tolist() == [2, 0]


def test_align_to_reports_missing():
    source = make_keys("d", [0], [0])
    with pytest.raises(KeyError, match="missing from source"):
        align_to(["d|000009|0000009"], source)
    assert align_to(["d|000009|0000009"], source, required=False).tolist() == [-1]


def test_variant_and_parent_share_episode_but_not_identity():
    parent = TransitionID("sample", 4, 9)
    variant = TransitionID("dataset__blur_seed1", 4, 9)
    assert parent.key != variant.key
    assert parent_episode_key("sample", 4) == "sample|000004"


def test_make_keys_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        make_keys("d", [0, 1], [0])


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_default_config_validates_and_matches_the_specification():
    cfg = Config().validate()
    assert cfg.weighting.alpha == 0.5
    assert cfg.weighting.w_min == 0.15
    assert cfg.visibility.lambdas() == {
        "object": 0.30,
        "gripper": 0.25,
        "segmentation": 0.20,
        "image_quality": 0.25,
    }
    assert sum(cfg.visibility.lambdas().values()) == pytest.approx(1.0)
    assert sum(cfg.reliability.weights().values()) == pytest.approx(1.0)


def test_visibility_lambdas_must_sum_to_one():
    cfg = VisibilityConfig(lambda_object=0.5)
    with pytest.raises(ValueError, match="must sum to 1"):
        cfg.validate()


def test_unknown_config_keys_are_rejected_not_ignored():
    with pytest.raises(ValueError, match="unknown config key"):
        from_dict(Config, {"weighting": {"alhpa": 0.3}})


def test_nested_dataclass_lists_are_coerced():
    cfg = from_dict(Config, {"dataset": {"cameras": [{"key": "observation.image"}]}})
    assert cfg.dataset.cameras[0].key == "observation.image"
    assert cfg.dataset.cameras[0].role == "scene"


def test_effective_alpha_per_method():
    assert WeightingConfig(method="visibility_only").effective_alpha() == 1.0
    assert WeightingConfig(method="reliability_only").effective_alpha() == 0.0
    assert WeightingConfig(method="combined", alpha=0.3).effective_alpha() == 0.3


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"alpha": 1.5}, "alpha must lie"),
        ({"w_min": 0.9, "w_max": 0.5}, "w_min <= w_max"),
        ({"chunk_alignment": "nope"}, "unknown chunk_alignment"),
        ({"padding_normalization": "nope"}, "unknown padding_normalization"),
        ({"method": "nope"}, "unknown weighting method"),
    ],
)
def test_weighting_config_validation(patch, message):
    cfg = WeightingConfig(**patch)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_parse_override_uses_yaml_scalars():
    assert parse_override("weighting.alpha=0.7") == (["weighting", "alpha"], 0.7)
    assert parse_override("vlm.enabled=true") == (["vlm", "enabled"], True)
    assert parse_override("policy.revision=null") == (["policy", "revision"], None)


def test_config_round_trips_through_dict():
    cfg = Config().validate()
    restored = from_dict(Config, cfg.to_dict()).validate()
    assert restored.to_dict() == cfg.to_dict()
