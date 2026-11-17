'''Metrics, cost-based evaluation, and threshold selection for imbalanced classification.

Accuracy is reported for reference only. With roughly 0.2% fraud, a model that flags
nothing is more than 99.8% accurate, so accuracy must never drive a decision here.
'''

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraudguard.config import ThresholdConfig

# Candidate decision thresholds. A fixed grid is easier to communicate and less
# prone to chasing noise than using every distinct score as a cut-off.
THRESHOLD_GRID = np.round(np.arange(0.005, 1.0, 0.005), 3)
COST_BASIS = 10_000


@dataclass(frozen=True)
class ThresholdMetrics:
    '''Confusion counts, rates, and cost at one decision threshold.'''

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    expected_cost: float
    cost_per_10k: float

    def to_dict(self) -> dict[str, float]:
        '''Return the metrics as a flat dictionary.'''
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    '''Outcome of the threshold-selection rule, with reference points for the report.'''

    threshold: float
    budget_met: bool
    unconstrained_threshold: float
    max_f1_threshold: float


def compute_ranking_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    '''Compute threshold-free metrics.

    PR-AUC is average precision, which does not interpolate between points and
    therefore does not flatter the curve.
    '''
    return {
        'roc_auc': float(roc_auc_score(target, scores)),
        'pr_auc': float(average_precision_score(target, scores)),
        'brier_score': float(brier_score_loss(target, scores)),
    }


def build_threshold_table(
    target: np.ndarray,
    scores: np.ndarray,
    config: ThresholdConfig,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    '''Compute confusion counts, rates, and expected cost for each threshold.

    A transaction is flagged when its score is greater than or equal to the threshold.
    Precision is reported as 0 when nothing is flagged.
    '''
    is_fraud = np.asarray(target) == 1
    flagged = np.asarray(scores)[np.newaxis, :] >= np.asarray(thresholds)[:, np.newaxis]
    true_positives = (flagged & is_fraud).sum(axis=1)
    false_positives = (flagged & ~is_fraud).sum(axis=1)
    false_negatives = int(is_fraud.sum()) - true_positives
    true_negatives = int((~is_fraud).sum()) - false_positives
    flagged_count = true_positives + false_positives

    precision = _safe_divide(true_positives, flagged_count)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    expected_cost = (
        false_positives * config.false_positive_cost + false_negatives * config.false_negative_cost
    )
    return pd.DataFrame(
        {
            'threshold': np.asarray(thresholds, dtype=np.float64),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': _safe_divide(2 * precision * recall, precision + recall),
            'false_positive_rate': _safe_divide(false_positives, false_positives + true_negatives),
            'false_negative_rate': _safe_divide(false_negatives, true_positives + false_negatives),
            'accuracy': (true_positives + true_negatives) / is_fraud.size,
            'expected_cost': expected_cost,
            'cost_per_10k': expected_cost / is_fraud.size * COST_BASIS,
        }
    )


def compute_threshold_metrics(
    target: np.ndarray, scores: np.ndarray, threshold: float, config: ThresholdConfig
) -> ThresholdMetrics:
    '''Compute confusion counts, rates, and cost at a single threshold.'''
    table = build_threshold_table(target, scores, config, np.array([threshold]))
    return metrics_at_threshold(table, threshold)


def metrics_at_threshold(table: pd.DataFrame, threshold: float) -> ThresholdMetrics:
    '''Return the row of a threshold table for one of its thresholds.'''
    row = table.loc[table['threshold'] == threshold].iloc[0]
    return ThresholdMetrics(
        threshold=float(row['threshold']),
        true_positives=int(row['true_positives']),
        false_positives=int(row['false_positives']),
        true_negatives=int(row['true_negatives']),
        false_negatives=int(row['false_negatives']),
        precision=float(row['precision']),
        recall=float(row['recall']),
        f1=float(row['f1']),
        false_positive_rate=float(row['false_positive_rate']),
        false_negative_rate=float(row['false_negative_rate']),
        accuracy=float(row['accuracy']),
        expected_cost=float(row['expected_cost']),
        cost_per_10k=float(row['cost_per_10k']),
    )


def select_threshold(table: pd.DataFrame, config: ThresholdConfig) -> ThresholdSelection:
    '''Choose the lowest-cost threshold whose false-positive rate fits the review budget.

    The rule encodes two business assumptions: a missed fraud costs far more than a
    false alert, and the review team can only absorb a limited rate of false alerts.
    Maximising F1 would ignore both, so it is reported for comparison only.

    If no threshold fits the budget, the one with the lowest false-positive rate is
    returned and budget_met is False.
    '''
    within_budget = table[table['false_positive_rate'] <= config.max_false_positive_rate]
    if within_budget.empty:
        fallback = table.sort_values(['false_positive_rate', 'expected_cost']).iloc[0]
        threshold, budget_met = float(fallback['threshold']), False
    else:
        threshold, budget_met = _lowest_cost_threshold(within_budget), True

    return ThresholdSelection(
        threshold=threshold,
        budget_met=budget_met,
        unconstrained_threshold=_lowest_cost_threshold(table),
        max_f1_threshold=float(table['threshold'].iloc[int(np.argmax(table['f1'].to_numpy()))]),
    )


def cost_without_model(target: np.ndarray, config: ThresholdConfig) -> float:
    '''Return the cost per 10,000 transactions when nothing is flagged.'''
    is_fraud = np.asarray(target) == 1
    return float(is_fraud.sum() * config.false_negative_cost / is_fraud.size * COST_BASIS)


def compute_calibration_table(
    target: np.ndarray, scores: np.ndarray, bin_count: int
) -> pd.DataFrame:
    '''Compare predicted probability with observed fraud rate in equal-width bins.

    Empty bins are omitted. Bins with few rows are noisy, so the count is included.
    '''
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_index = np.clip(np.digitize(scores, edges[1:-1]), 0, bin_count - 1)
    frame = pd.DataFrame({'bin': bin_index, 'score': scores, 'target': target})
    grouped = frame.groupby('bin').agg(
        mean_predicted=('score', 'mean'), observed_rate=('target', 'mean'), count=('target', 'size')
    )
    grouped.insert(0, 'bin_lower', edges[grouped.index])
    grouped.insert(1, 'bin_upper', edges[grouped.index + 1])
    return grouped.reset_index(drop=True)


def bootstrap_confidence_intervals(
    target: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    config: ThresholdConfig,
    sample_count: int,
    random_seed: int,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    '''Estimate percentile bootstrap intervals for the headline metrics.

    The test split holds few fraud cases, so point estimates alone would overstate
    how precisely performance is known.
    '''
    target, scores = np.asarray(target), np.asarray(scores)
    rng = np.random.default_rng(random_seed)
    samples: dict[str, list[float]] = {
        'pr_auc': [],
        'roc_auc': [],
        'precision': [],
        'recall': [],
        'cost_per_10k': [],
    }
    for _ in range(sample_count):
        rows = rng.integers(0, target.size, size=target.size)
        resampled_target, resampled_scores = target[rows], scores[rows]
        # Ranking metrics are undefined when a resample holds a single class
        if resampled_target.min() == resampled_target.max():
            continue
        ranking = compute_ranking_metrics(resampled_target, resampled_scores)
        at_threshold = compute_threshold_metrics(
            resampled_target, resampled_scores, threshold, config
        ).to_dict()
        for name, values in samples.items():
            values.append(ranking[name] if name in ranking else at_threshold[name])

    tail = (1 - confidence) / 2 * 100
    return {
        name: (float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail)))
        for name, values in samples.items()
        if values
    }


def _lowest_cost_threshold(table: pd.DataFrame) -> float:
    '''Return the cheapest threshold, taking the middle of any tied range.'''
    cheapest = table[table['expected_cost'] == table['expected_cost'].min()]
    # The middle of a cost plateau tolerates small score shifts better than its edges
    return float(cheapest['threshold'].iloc[len(cheapest) // 2])


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    '''Divide element-wise, returning 0 where the denominator is 0.'''
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result: np.ndarray = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return result
