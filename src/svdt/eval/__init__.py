"""Offline metrics, score-quality metrics, and an optional rollout adapter."""

from .metrics import (
    action_chunk_error,
    action_mae,
    aggregate_over_seeds,
    auprc,
    auroc,
    degradation_table,
    distribution_summary,
    ranking_metrics,
    seed_statistics,
)
from .rollout import RolloutSupport, check_rollout_support, rollout_placeholder, run_rollout
from .score_eval import component_contribution, evaluate_scores, hard_filter_tradeoff

__all__ = [
    "RolloutSupport",
    "action_chunk_error",
    "action_mae",
    "aggregate_over_seeds",
    "auprc",
    "auroc",
    "check_rollout_support",
    "component_contribution",
    "degradation_table",
    "distribution_summary",
    "evaluate_scores",
    "hard_filter_tradeoff",
    "ranking_metrics",
    "rollout_placeholder",
    "run_rollout",
    "seed_statistics",
]
