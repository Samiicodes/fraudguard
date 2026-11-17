'''Tests for metrics, cost-based threshold selection, and uncertainty estimates.'''

import numpy as np
import pandas as pd
import pytest

from fraudguard.config import ThresholdConfig
from fraudguard.evaluation import (
    bootstrap_confidence_intervals,
    build_threshold_table,
    compute_calibration_table,
    compute_ranking_metrics,
    compute_threshold_metrics,
    cost_without_model,
    select_threshold,
)

COSTS = ThresholdConfig(
    false_positive_cost=5.0, false_negative_cost=150.0, max_false_positive_rate=0.01
)
TARGET = np.array([0, 0, 0, 0, 1, 1])
SCORES = np.array([0.1, 0.2, 0.35, 0.6, 0.55, 0.9])


def candidate_table(false_positive_rate: list[float], expected_cost: list[float]) -> pd.DataFrame:
    thresholds = [round(0.1 * (position + 1), 1) for position in range(len(expected_cost))]
    return pd.DataFrame(
        {
            'threshold': thresholds,
            'false_positive_rate': false_positive_rate,
            'expected_cost': expected_cost,
            'f1': [0.5, 0.6, 0.7, 0.4, 0.3][: len(expected_cost)],
        }
    )


def test_counts_rates_and_cost_at_a_threshold() -> None:
    metrics = compute_threshold_metrics(TARGET, SCORES, 0.5, COSTS)

    # Flagged: 0.6 (legitimate) plus 0.55 and 0.9 (fraud)
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 3
    assert metrics.false_negatives == 0
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.false_positive_rate == pytest.approx(1 / 4)
    assert metrics.expected_cost == pytest.approx(5.0)
    assert metrics.cost_per_10k == pytest.approx(5.0 / 6 * 10_000)


def test_score_equal_to_the_threshold_is_flagged() -> None:
    assert compute_threshold_metrics(TARGET, SCORES, 0.55, COSTS).true_positives == 2


def test_flagging_nothing_gives_zero_precision_and_full_fraud_cost() -> None:
    metrics = compute_threshold_metrics(TARGET, SCORES, 0.95, COSTS)

    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert metrics.expected_cost == pytest.approx(2 * 150.0)
    # Accuracy still looks respectable, which is why it is never used for decisions
    assert metrics.accuracy == pytest.approx(4 / 6)


def test_threshold_table_is_ordered_and_monotonic() -> None:
    table = build_threshold_table(TARGET, SCORES, COSTS)

    assert table['threshold'].is_monotonic_increasing
    assert table['recall'].is_monotonic_decreasing
    assert table['false_positive_rate'].is_monotonic_decreasing


def test_selection_takes_the_cheapest_threshold_within_the_review_budget() -> None:
    table = candidate_table(
        false_positive_rate=[0.05, 0.02, 0.008, 0.001], expected_cost=[100.0, 120.0, 150.0, 300.0]
    )

    selection = select_threshold(table, COSTS)

    assert selection.threshold == 0.3
    assert selection.budget_met
    assert selection.unconstrained_threshold == 0.1
    assert selection.max_f1_threshold == 0.3


def test_selection_falls_back_to_the_lowest_false_positive_rate() -> None:
    table = candidate_table(
        false_positive_rate=[0.05, 0.04, 0.03, 0.02], expected_cost=[100.0, 120.0, 150.0, 300.0]
    )

    selection = select_threshold(table, COSTS)

    assert selection.threshold == 0.4
    assert not selection.budget_met


def test_tied_costs_resolve_to_the_middle_of_the_plateau() -> None:
    table = candidate_table(
        false_positive_rate=[0.0] * 5, expected_cost=[200.0, 100.0, 100.0, 100.0, 300.0]
    )

    assert select_threshold(table, COSTS).threshold == 0.3


def test_ranking_metrics_for_perfect_and_random_scores() -> None:
    perfect = compute_ranking_metrics(TARGET, TARGET.astype(float))
    rng = np.random.default_rng(0)
    target = (rng.random(20_000) < 0.01).astype(int)
    random_scores = compute_ranking_metrics(target, rng.random(20_000))

    assert perfect == {'roc_auc': 1.0, 'pr_auc': 1.0, 'brier_score': 0.0}
    assert random_scores['roc_auc'] == pytest.approx(0.5, abs=0.05)
    # Random scores give a PR-AUC near the fraud rate, far below the ROC-AUC
    assert random_scores['pr_auc'] < 0.05


def test_cost_without_model_counts_every_fraud_as_missed() -> None:
    assert cost_without_model(TARGET, COSTS) == pytest.approx(2 * 150.0 / 6 * 10_000)


def test_calibration_table_reports_occupied_bins_only() -> None:
    table = compute_calibration_table(
        np.array([0, 0, 1, 1]), np.array([0.05, 0.15, 0.85, 0.95]), bin_count=10
    )

    assert table['bin_lower'].tolist() == pytest.approx([0.0, 0.1, 0.8, 0.9])
    assert table['observed_rate'].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert table['count'].sum() == 4


def test_bootstrap_intervals_contain_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    target = (rng.random(3000) < 0.05).astype(int)
    scores = np.clip(target * 0.6 + rng.random(3000) * 0.5, 0.0, 1.0)

    intervals = bootstrap_confidence_intervals(
        target, scores, 0.5, COSTS, sample_count=100, random_seed=0
    )

    low, high = intervals['pr_auc']
    assert low <= compute_ranking_metrics(target, scores)['pr_auc'] <= high
    assert set(intervals) == {'pr_auc', 'roc_auc', 'precision', 'recall', 'cost_per_10k'}
