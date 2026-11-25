"""The weighting formula, chunk alignment, padding, and the weighted loss.

These are the completion criteria that must hold on any machine: uniform vs
weighted equivalence, chunk/padding alignment, and numpy/torch parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_torch

from svdt.config import WeightingConfig
from svdt.ids import make_keys
from svdt.weighting.loss import (
    effective_sample_weights,
    per_sample_losses,
    uniform_chunk_loss,
    weighted_chunk_loss,
    weighted_per_sample_loss,
)
from svdt.weighting.weights import (
    chunk_weights,
    compute_weights,
    hard_filter_mask,
    weight_report,
)

KEYS = make_keys("d", [0] * 8, list(range(8)))


# --------------------------------------------------------------------------- #
# w_t = clip(alpha*v + (1-alpha)*c, w_min, 1)
# --------------------------------------------------------------------------- #


def test_combined_formula_matches_the_specification():
    v = np.array([1.0, 0.8, 0.0, 0.5])
    c = np.array([1.0, 0.2, 0.0, 0.5])
    cfg = WeightingConfig(method="combined", alpha=0.5, w_min=0.15)
    out = compute_weights(cfg, keys=KEYS[:4], v=v, c=c)
    expected = np.clip(0.5 * v + 0.5 * c, 0.15, 1.0)
    assert out.w == pytest.approx(expected, abs=1e-6)


def test_w_min_floor_and_upper_clip_apply():
    cfg = WeightingConfig(method="combined", alpha=0.5, w_min=0.15)
    out = compute_weights(
        cfg, keys=KEYS[:2], v=np.array([0.0, 1.0]), c=np.array([0.0, 1.0])
    )
    assert out.w[0] == pytest.approx(0.15)
    assert out.w[1] == pytest.approx(1.0)
    assert out.w.min() >= 0.15 and out.w.max() <= 1.0


def test_uniform_is_exactly_ones_and_needs_no_scores():
    out = compute_weights(WeightingConfig(method="uniform"), keys=KEYS[:4])
    assert np.all(out.w == 1.0)
    assert not out.missing.any() and not out.dropped.any()


def test_visibility_only_ignores_c_entirely():
    v = np.array([0.9, 0.2])
    cfg = WeightingConfig(method="visibility_only", w_min=0.0)
    a = compute_weights(cfg, keys=KEYS[:2], v=v, c=np.array([0.0, 1.0]))
    b = compute_weights(cfg, keys=KEYS[:2], v=v, c=np.array([1.0, 0.0]))
    assert a.w == pytest.approx(b.w)
    assert a.w == pytest.approx(v)


def test_reliability_only_ignores_v_entirely():
    c = np.array([0.9, 0.2])
    cfg = WeightingConfig(method="reliability_only", w_min=0.0)
    a = compute_weights(cfg, keys=KEYS[:2], v=np.array([0.0, 1.0]), c=c)
    assert a.w == pytest.approx(c)


def test_hard_filter_drops_below_threshold():
    cfg = WeightingConfig(method="hard_filter", hard_filter_threshold=0.5, alpha=0.5)
    v = np.array([0.9, 0.9, 0.1, 0.1])
    c = np.array([0.9, 0.1, 0.9, 0.1])
    out = compute_weights(cfg, keys=KEYS[:4], v=v, c=c)
    assert out.w.tolist() == [1.0, 1.0, 1.0, 0.0]  # trust 0.9, 0.5, 0.5, 0.1
    assert hard_filter_mask(out).tolist() == [True, True, True, False]
    assert weight_report(out)["retained_fraction"] == pytest.approx(0.75)


def test_missing_scores_are_flagged_and_resolved_by_policy():
    v = np.array([0.8, np.nan])
    c = np.array([0.8, 0.8])
    floor = compute_weights(
        WeightingConfig(method="combined", missing_handling="w_min"),
        keys=KEYS[:2],
        v=v,
        c=c,
    )
    assert floor.missing.tolist() == [False, True]
    assert floor.w[1] == pytest.approx(0.15)

    dropped = compute_weights(
        WeightingConfig(method="combined", missing_handling="drop"),
        keys=KEYS[:2],
        v=v,
        c=c,
    )
    assert dropped.w[1] == 0.0 and dropped.dropped[1]

    kept = compute_weights(
        WeightingConfig(method="combined", missing_handling="one"),
        keys=KEYS[:2],
        v=v,
        c=c,
    )
    assert kept.w[1] == pytest.approx(1.0)


def test_scores_outside_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"must lie in \[0,1\]"):
        compute_weights(
            WeightingConfig(), keys=KEYS[:2], v=np.array([0.5, 1.4]), c=np.array([0.5, 0.5])
        )


def test_weight_report_accounting():
    out = compute_weights(
        WeightingConfig(method="combined"),
        keys=KEYS[:4],
        v=np.array([1.0, 1.0, 0.0, 0.0]),
        c=np.array([1.0, 0.0, 1.0, 0.0]),
    )
    report = weight_report(out, n_total=10)
    assert report["n_transitions_scored"] == 4
    assert report["n_transitions_total"] == 10
    assert report["weight_sum"] == pytest.approx(float(np.sum(out.w)), abs=1e-5)
    assert report["effective_sample_size"] <= 4.0


# --------------------------------------------------------------------------- #
# chunk alignment and padding
# --------------------------------------------------------------------------- #


def test_anchor_alignment_uses_the_anchor_weight():
    anchor = np.array([0.2, 0.9])
    members = np.array([[0.2, 1.0], [0.9, 0.1]])
    out = chunk_weights(anchor, alignment="anchor", chunk_member_weights=members)
    assert out == pytest.approx(anchor)


def test_chunk_mean_and_min_ignore_padding_steps():
    members = np.array([[1.0, 0.2, 0.0], [0.5, 0.5, 0.5]])
    pad = np.array([[False, False, True], [False, False, False]])
    mean = chunk_weights(
        np.array([1.0, 0.5]),
        alignment="chunk_mean",
        chunk_member_weights=members,
        pad_mask=pad,
    )
    # First sample averages 1.0 and 0.2 only; the padded 0.0 must not count.
    assert mean[0] == pytest.approx(0.6)
    assert mean[1] == pytest.approx(0.5)

    minimum = chunk_weights(
        np.array([1.0, 0.5]),
        alignment="chunk_min",
        chunk_member_weights=members,
        pad_mask=pad,
    )
    assert minimum[0] == pytest.approx(0.2)


def test_fully_padded_sample_gets_zero_weight_in_every_alignment():
    pad = np.array([[True, True], [False, False]])
    members = np.array([[0.9, 0.9], [0.4, 0.6]])
    for alignment in ("anchor", "chunk_mean", "chunk_min"):
        out = chunk_weights(
            np.array([0.9, 0.5]),
            alignment=alignment,
            chunk_member_weights=members,
            pad_mask=pad,
        )
        assert out[0] == 0.0, alignment
        assert out[1] > 0.0, alignment


def test_chunk_alignment_requires_member_weights():
    with pytest.raises(ValueError, match="needs chunk_member_weights"):
        chunk_weights(np.array([1.0]), alignment="chunk_mean")


# --------------------------------------------------------------------------- #
# weighted loss
# --------------------------------------------------------------------------- #


def test_padding_is_excluded_from_the_per_sample_loss_and_its_denominator(chunk_batch):
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    per_sample, valid_steps = per_sample_losses(pred, target, pad)
    assert valid_steps.tolist() == [3, 5, 1, 5, 5, 0]

    # Changing a padded element must not change the loss at all.
    mutated = pred.copy()
    mutated[0, 4] += 1000.0
    other, _ = per_sample_losses(mutated, target, pad)
    assert per_sample == pytest.approx(other)


def test_uniform_equals_weighted_with_per_element_normalization(chunk_batch):
    """The uniform control must reproduce the unweighted loss *exactly*."""
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    baseline = uniform_chunk_loss(pred, target, pad)
    weighted = weighted_chunk_loss(
        pred, target, np.ones(len(pred)), pad, padding_normalization="per_element"
    )
    assert weighted.loss == baseline.loss


def test_per_sample_normalization_differs_when_valid_counts_differ(chunk_batch):
    """Documented, deliberate: the two reductions are different functions."""
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    baseline = uniform_chunk_loss(pred, target, pad)
    per_sample = weighted_chunk_loss(
        pred, target, np.ones(len(pred)), pad, padding_normalization="per_sample"
    )
    assert per_sample.loss != pytest.approx(baseline.loss, rel=1e-9)

    # With equal valid-step counts they coincide again.
    equal_pad = np.zeros_like(pad)
    equal_pad[:, 4:] = True
    assert weighted_chunk_loss(
        pred, target, np.ones(len(pred)), equal_pad
    ).loss == pytest.approx(uniform_chunk_loss(pred, target, equal_pad).loss)


def test_fully_padded_sample_leaves_numerator_and_denominator(chunk_batch):
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    weights = np.full(len(pred), 0.5)
    out = weighted_chunk_loss(pred, target, weights, pad)
    assert out.sample_weights[5] == 0.0
    assert out.weight_sum == pytest.approx(0.5 * 5)  # 5 of 6 samples contribute
    assert out.n_contributing == 5


def test_weighted_loss_is_the_normalised_weighted_mean(chunk_batch):
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.2, 1.0, size=len(pred))
    out = weighted_chunk_loss(pred, target, weights, pad)
    eff = effective_sample_weights(weights, out.valid_steps)
    manual = float((out.per_sample_loss * eff).sum() / eff.sum())
    assert out.loss == pytest.approx(manual, rel=1e-12)


def test_zero_weight_sample_contributes_nothing(chunk_batch):
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    weights = np.ones(len(pred))
    weights[1] = 0.0
    with_zero = weighted_chunk_loss(pred, target, weights, pad).loss
    without = weighted_chunk_loss(
        np.delete(pred, 1, axis=0),
        np.delete(target, 1, axis=0),
        np.delete(weights, 1),
        np.delete(pad, 1, axis=0),
    ).loss
    assert with_zero == pytest.approx(without, rel=1e-9)


def test_all_zero_weights_do_not_divide_by_zero(chunk_batch):
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    out = weighted_chunk_loss(pred, target, np.zeros(len(pred)), pad)
    assert np.isfinite(out.loss) and out.loss == 0.0


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_per_sample_loss(np.array([1.0, 1.0]), np.array([1.0, -0.5]))


def test_eps_mode_lerobot_matches_upstream_formula():
    losses = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 1.0, 0.25])
    guard, _ = weighted_per_sample_loss(losses, weights, eps_mode="guard")
    upstream, _ = weighted_per_sample_loss(losses, weights, eps_mode="lerobot")
    assert guard == pytest.approx(float((losses * weights).sum() / weights.sum()))
    assert upstream == pytest.approx(
        float((losses * weights).sum() / (weights.sum() + 1e-6))
    )


def test_mismatched_shapes_are_rejected(chunk_batch):
    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    with pytest.raises(ValueError, match="pad_mask must be"):
        weighted_chunk_loss(pred, target, np.ones(len(pred)), pad[:, :2])
    with pytest.raises(ValueError, match="must match"):
        weighted_chunk_loss(pred, target, np.ones(len(pred) + 1), pad)


# --------------------------------------------------------------------------- #
# torch parity (skipped where torch cannot be imported)
# --------------------------------------------------------------------------- #


@requires_torch
@pytest.mark.needs_torch
def test_torch_matches_the_numpy_reference(chunk_batch):
    import torch

    from svdt.weighting.loss import torch_uniform_chunk_loss, torch_weighted_chunk_loss

    pred, target, pad = chunk_batch["pred"], chunk_batch["target"], chunk_batch["pad"]
    rng = np.random.default_rng(2)
    weights = rng.uniform(0.1, 1.0, size=len(pred))

    tp = torch.tensor(pred, dtype=torch.float64)
    tt = torch.tensor(target, dtype=torch.float64)
    tpad = torch.tensor(pad)
    tw = torch.tensor(weights, dtype=torch.float64)

    for normalization in ("per_sample", "per_element"):
        np_loss = weighted_chunk_loss(
            pred, target, weights, pad, padding_normalization=normalization
        ).loss
        torch_loss, _ = torch_weighted_chunk_loss(
            tp, tt, tw, tpad, padding_normalization=normalization
        )
        assert float(torch_loss) == pytest.approx(np_loss, rel=1e-12)

    assert float(torch_uniform_chunk_loss(tp, tt, tpad)) == pytest.approx(
        uniform_chunk_loss(pred, target, pad).loss, rel=1e-12
    )


@requires_torch
@pytest.mark.needs_torch
def test_torch_weighted_loss_is_differentiable(chunk_batch):
    import torch

    from svdt.weighting.loss import torch_weighted_chunk_loss

    pred = torch.tensor(chunk_batch["pred"], dtype=torch.float64, requires_grad=True)
    target = torch.tensor(chunk_batch["target"], dtype=torch.float64)
    pad = torch.tensor(chunk_batch["pad"])
    weights = torch.full((pred.shape[0],), 0.5, dtype=torch.float64)

    loss, _ = torch_weighted_chunk_loss(pred, target, weights, pad)
    loss.backward()
    assert pred.grad is not None
    # Padded positions must receive no gradient at all.
    assert torch.allclose(pred.grad[0, 3:], torch.zeros_like(pred.grad[0, 3:]))
    assert torch.allclose(pred.grad[5], torch.zeros_like(pred.grad[5]))
