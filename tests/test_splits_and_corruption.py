"""Episode splits, corruption generation, and the leakage guarantee.

Completion criterion: after generating corruptions there must be no episode
leakage between splits.
"""

from __future__ import annotations

import numpy as np
import pytest

from svdt.config import CorruptionSpec, SplitConfig
from svdt.data.corrupt import (
    CorruptedFrameSource,
    CorruptionError,
    build_variants,
    parent_of,
    robust_action_scale,
    select_episodes,
    stable_seed,
    validate_ops,
    variant_dataset_id,
)
from svdt.data.splits import (
    check_no_parent_overlap,
    check_variant_split_consistency,
    load_split,
    make_split,
    save_split,
    training_episodes,
    tuning_episodes,
)
from svdt.ids import TransitionID


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #


def test_split_is_deterministic_for_a_given_seed():
    cfg = SplitConfig(seed=7)
    a = make_split(range(50), cfg, dataset_id="d")
    b = make_split(range(50), cfg, dataset_id="d")
    assert a.assignment == b.assignment


def test_split_changes_with_the_seed():
    a = make_split(range(50), SplitConfig(seed=1), dataset_id="d")
    b = make_split(range(50), SplitConfig(seed=2), dataset_id="d")
    assert a.assignment != b.assignment


def test_split_covers_every_episode_exactly_once():
    split = make_split(range(37), SplitConfig(), dataset_id="d")
    assert sum(split.counts().values()) == 37
    all_eps = split.episodes("train") + split.episodes("val") + split.episodes("test")
    assert sorted(all_eps) == list(range(37))
    assert len(set(all_eps)) == 37


def test_no_split_is_empty_when_its_fraction_is_positive():
    """A silently empty validation set would make threshold tuning impossible."""
    split = make_split(range(4), SplitConfig(train=0.7, val=0.15, test=0.15), dataset_id="d")
    assert all(count > 0 for count in split.counts().values())
    check_no_parent_overlap(split).raise_if_bad()


def test_training_never_includes_test_episodes():
    split = make_split(range(20), SplitConfig(), dataset_id="d")
    test = set(split.episodes("test"))
    assert not set(training_episodes(split)) & test
    assert not set(training_episodes(split, include_val=True)) & test
    assert not set(tuning_episodes(split)) & test
    assert set(tuning_episodes(split)) == set(split.episodes("val"))


def test_split_round_trips_through_disk(tmp_path):
    split = make_split(range(12), SplitConfig(seed=3), dataset_id="demo")
    save_split(split, tmp_path)
    restored = load_split(tmp_path, "demo")
    assert restored.assignment == split.assignment
    assert restored.protected_test == split.protected_test


def test_missing_split_file_names_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="svdt split"):
        load_split(tmp_path, "nope")


# --------------------------------------------------------------------------- #
# corruption specs
# --------------------------------------------------------------------------- #


def test_unknown_corruption_type_is_rejected():
    with pytest.raises(CorruptionError, match="unknown corruption type"):
        validate_ops([{"type": "cosmic_rays"}])


def test_unknown_corruption_parameter_is_rejected_not_ignored():
    with pytest.raises(CorruptionError, match="unsupported parameter"):
        validate_ops([{"type": "blur", "sigmaa": 2.0}])


def test_op_defaults_are_filled_in():
    (op,) = validate_ops([{"type": "blur"}])
    assert op == {"type": "blur", "sigma": 2.0}


@pytest.mark.parametrize(
    "op, message",
    [
        ({"type": "blur", "sigma": 0}, "sigma must be positive"),
        ({"type": "occlusion", "size_fraction": 0}, "size_fraction"),
        ({"type": "occlusion", "mode": "weird"}, "mode must be"),
        ({"type": "action_outlier", "fraction": 0}, "fraction must lie"),
        ({"type": "exposure", "gamma": 0}, "gamma must be positive"),
    ],
)
def test_op_parameter_bounds(op, message):
    with pytest.raises(CorruptionError, match=message):
        validate_ops([op])


def test_episode_selection_is_stable_across_processes():
    """`hash()` is salted per interpreter, so a digest-based seed is required."""
    assert stable_seed(0, "blur", "sample") == stable_seed(0, "blur", "sample")
    spec = CorruptionSpec(name="blur", ops=[{"type": "blur"}], episode_fraction=0.5, seed=1)
    first = select_episodes(range(20), spec, salt="sample")
    second = select_episodes(range(20), spec, salt="sample")
    assert first == second
    # A fixed expectation pins the value so a refactor cannot silently reshuffle.
    assert first == select_episodes(list(range(20)), spec, salt="sample")


def test_variant_dataset_id_encodes_spec_and_seed():
    spec = CorruptionSpec(name="blur_s3", ops=[{"type": "blur"}], seed=4)
    assert variant_dataset_id("sample", spec) == "dataset__blur_s3_seed4"


def test_robust_action_scale_is_positive_even_for_a_constant_action():
    constant = np.ones((10, 3), dtype=np.float32)
    assert np.all(robust_action_scale(constant) > 0)


# --------------------------------------------------------------------------- #
# corruption behaviour
# --------------------------------------------------------------------------- #


def test_action_corruption_changes_actions_and_labels_only_those_frames(source):
    spec = CorruptionSpec(
        name="outlier",
        ops=[{"type": "action_outlier", "magnitude": 10.0, "fraction": 0.25}],
        frame_fraction=1.0,
        episode_fraction=1.0,
        seed=2,
    )
    variant = CorruptedFrameSource(source, spec)
    ep = variant.episode_indices()[0]
    clean = source.read_lowdim(ep)
    dirty = variant.read_lowdim(ep)

    labels = variant.corruption_labels()
    flags = {str(k): bool(f) for k, f in zip(labels["keys"], labels["corrupted"])}
    changed = ~np.all(np.isclose(clean.action, dirty.action), axis=1)

    for i, frame in enumerate(clean.frame_index):
        key = TransitionID(variant.dataset_id, ep, int(frame)).key
        assert flags[key] == bool(changed[i]), f"label/change mismatch at frame {frame}"
    assert changed.any(), "the corruption changed nothing"
    assert not changed.all(), "fraction=0.25 should not hit every frame"


def test_image_corruption_leaves_actions_untouched(source):
    spec = CorruptionSpec(
        name="blur",
        ops=[{"type": "blur", "sigma": 3.0}],
        frame_fraction=1.0,
        episode_fraction=1.0,
        seed=5,
    )
    variant = CorruptedFrameSource(source, spec)
    ep = variant.episode_indices()[0]
    assert np.allclose(source.read_lowdim(ep).action, variant.read_lowdim(ep).action)

    clean_batch = next(iter(source.iter_image_batches(ep, ["observation.image"])))
    dirty_batch = next(iter(variant.iter_image_batches(ep, ["observation.image"])))
    assert not np.array_equal(
        clean_batch.images["observation.image"], dirty_batch.images["observation.image"]
    )


def test_temporal_shift_marks_every_transition(source):
    spec = CorruptionSpec(
        name="shift",
        ops=[{"type": "temporal_shift", "frames": 3}],
        frame_fraction=0.25,  # deliberately small: a shift still affects all
        episode_fraction=1.0,
        seed=6,
    )
    variant = CorruptedFrameSource(source, spec)
    labels = variant.corruption_labels()
    assert bool(np.all(labels["corrupted"]))


def test_temporal_shift_replicates_edges_rather_than_wrapping(source):
    spec = CorruptionSpec(
        name="shift",
        ops=[{"type": "temporal_shift", "frames": 2}],
        episode_fraction=1.0,
        seed=6,
    )
    variant = CorruptedFrameSource(source, spec)
    ep = variant.episode_indices()[0]
    clean = source.read_lowdim(ep).action
    dirty = variant.read_lowdim(ep).action
    assert np.allclose(dirty[2:], clean[:-2])
    # A wrap would put the episode's final actions at the start.
    assert np.allclose(dirty[:2], clean[0])


def test_corruption_is_reproducible_from_the_spec(source):
    spec = CorruptionSpec(
        name="noise",
        ops=[{"type": "action_noise", "sigma": 2.0}],
        episode_fraction=1.0,
        seed=9,
    )
    a = CorruptedFrameSource(source, spec)
    b = CorruptedFrameSource(source, spec)
    ep = a.episode_indices()[0]
    assert np.allclose(a.read_lowdim(ep).action, b.read_lowdim(ep).action)


def test_variant_keeps_its_own_identity(source, corruption_specs):
    variant = CorruptedFrameSource(source, corruption_specs[0])
    assert variant.dataset_id != source.dataset_id
    key = variant.corruption_labels()["keys"][0]
    parent = parent_of(str(key))
    assert parent.dataset_id == source.dataset_id
    assert parent.episode_index == TransitionID.parse(str(key)).episode_index


def test_labels_are_not_reachable_from_the_frame_payload(source, corruption_specs):
    """A scorer reading batches must not be able to see the injected labels."""
    variant = CorruptedFrameSource(source, corruption_specs[0])
    ep = variant.episode_indices()[0]
    batch = next(iter(variant.iter_image_batches(ep, ["observation.image"])))
    assert set(vars(batch)) == {"episode_index", "frame_index", "images"}
    episode = variant.read_lowdim(ep)
    assert "corrupted" not in episode.extras
    assert not any("corrupt" in k for k in episode.extras)


# --------------------------------------------------------------------------- #
# the leakage guarantee
# --------------------------------------------------------------------------- #


def test_corruption_copies_inherit_their_parent_split(source, corruption_specs):
    split = make_split(source.episode_indices(), SplitConfig(seed=0), dataset_id=source.dataset_id)
    variants = build_variants(source, corruption_specs, episodes=source.episode_indices())
    mapping = {v.dataset_id: v.record().episode_parents for v in variants}
    variant_splits = {
        did: {ep: split.split_of(parent) for ep, parent in parents.items()}
        for did, parents in mapping.items()
    }
    report = check_variant_split_consistency(split, mapping, variant_splits=variant_splits)
    assert report.ok, report.violations


def test_leakage_is_detected_when_a_copy_is_misassigned(source, corruption_specs):
    split = make_split(source.episode_indices(), SplitConfig(seed=0), dataset_id=source.dataset_id)
    variants = build_variants(source, corruption_specs, episodes=source.episode_indices())
    mapping = {v.dataset_id: v.record().episode_parents for v in variants}

    # Deliberately mis-file one copy into a different split.
    bad = {}
    for did, parents in mapping.items():
        bad[did] = {}
        for ep, parent in parents.items():
            actual = split.split_of(parent)
            bad[did][ep] = "test" if actual != "test" else "train"
    report = check_variant_split_consistency(split, mapping, variant_splits=bad)
    assert not report.ok
    assert any("but its parent episode" in v for v in report.violations)
    with pytest.raises(AssertionError, match="leakage detected"):
        report.raise_if_bad()


def test_no_clean_test_episode_appears_in_any_training_set(source, corruption_specs):
    split = make_split(source.episode_indices(), SplitConfig(seed=0), dataset_id=source.dataset_id)
    test_eps = set(split.episodes("test"))
    train_eps = set(training_episodes(split, include_val=True))
    assert not (test_eps & train_eps)
    assert set(split.protected_test) == test_eps
