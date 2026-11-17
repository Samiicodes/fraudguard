'''Tests for drift detection and prediction monitoring.'''

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudguard.config import MonitoringConfig
from fraudguard.data_ingestion import INPUT_COLUMNS
from fraudguard.monitoring import (
    PredictionMonitor,
    build_reference_profile,
    compute_drift_report,
    population_stability_index,
)


@pytest.fixture
def monitoring_config(tmp_path: Path) -> MonitoringConfig:
    return MonitoringConfig(
        prediction_log_path=tmp_path / 'monitoring' / 'predictions.jsonl',
        window_size=1000,
        min_drift_samples=100,
        psi_bins=10,
        psi_warning_threshold=0.1,
        psi_alert_threshold=0.25,
    )


def as_inputs(frame: pd.DataFrame) -> list[dict[str, float]]:
    return [
        {str(name): float(value) for name, value in row.items()}
        for _, row in frame[INPUT_COLUMNS].iterrows()
    ]


def test_psi_is_zero_for_identical_distributions() -> None:
    shares = np.array([0.2, 0.3, 0.5])

    assert population_stability_index(shares, shares) == pytest.approx(0.0)


def test_psi_stays_finite_when_a_bin_is_empty() -> None:
    value = population_stability_index(np.array([0.5, 0.5, 0.0]), np.array([0.4, 0.4, 0.2]))

    assert np.isfinite(value)
    assert value > 0


def test_reference_profile_uses_equal_share_bins(transactions: pd.DataFrame) -> None:
    profile = build_reference_profile(transactions, bin_count=10)

    shares = profile['features']['v1']['expected']
    assert len(shares) == 10
    assert shares == pytest.approx([0.1] * 10, abs=0.01)


def test_data_from_the_same_distribution_is_stable(
    transactions: pd.DataFrame, monitoring_config: MonitoringConfig
) -> None:
    profile = build_reference_profile(transactions.iloc[:2000], bin_count=10)

    report = compute_drift_report(profile, transactions.iloc[2000:], monitoring_config)

    assert report['status'] == 'stable'
    assert report['drifted_features'] == {}


def test_shifted_amounts_raise_an_alert(
    transactions: pd.DataFrame, monitoring_config: MonitoringConfig
) -> None:
    profile = build_reference_profile(transactions.iloc[:2000], bin_count=10)
    recent = transactions.iloc[2000:].copy()
    recent['amount'] *= 20

    report = compute_drift_report(profile, recent, monitoring_config)

    assert report['status'] == 'alert'
    assert next(iter(report['drifted_features'])) == 'amount'


def test_drift_waits_for_enough_samples(
    transactions: pd.DataFrame, monitoring_config: MonitoringConfig
) -> None:
    profile = build_reference_profile(transactions, bin_count=10)

    report = compute_drift_report(profile, transactions.head(50), monitoring_config)

    assert report == {'status': 'insufficient_data', 'sample_count': 50, 'required_samples': 100}


def test_monitor_tracks_counts_rates_and_latency(
    transactions: pd.DataFrame, monitoring_config: MonitoringConfig
) -> None:
    monitor = PredictionMonitor(monitoring_config)

    for index, inputs in enumerate(as_inputs(transactions.head(4))):
        monitor.record(
            request_id=f'request-{index}',
            inputs=inputs,
            fraud_probability=0.9 if index == 0 else 0.1,
            is_flagged=index == 0,
            latency_ms=10.0 + index,
            model_version='1',
        )
    summary = monitor.summary()

    assert summary['prediction_count'] == 4
    assert summary['flagged_count'] == 1
    assert summary['flagged_rate'] == pytest.approx(0.25)
    assert summary['mean_fraud_probability'] == pytest.approx(0.3)
    assert summary['latency_ms']['max'] == pytest.approx(13.0)
    assert set(summary['input_statistics']) >= {'amount', 'v1', 'v28'}
    assert 'drift' not in summary
    json.dumps(summary)


def test_monitor_appends_every_prediction_to_the_log(
    transactions: pd.DataFrame, monitoring_config: MonitoringConfig
) -> None:
    monitor = PredictionMonitor(monitoring_config)

    for index, inputs in enumerate(as_inputs(transactions.head(3))):
        monitor.record(f'request-{index}', inputs, 0.2, False, 5.0, '1')

    lines = monitoring_config.prediction_log_path.read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['request_id'] for line in lines] == [
        'request-0',
        'request-1',
        'request-2',
    ]


def test_monitor_reports_drift_against_the_reference_profile(
    transactions: pd.DataFrame, monitoring_config: MonitoringConfig
) -> None:
    monitor = PredictionMonitor(monitoring_config, build_reference_profile(transactions, 10))

    for index, inputs in enumerate(as_inputs(transactions.head(150))):
        monitor.record(f'request-{index}', inputs, 0.1, False, 5.0, '1')

    drift = monitor.summary()['drift']
    assert drift['sample_count'] == 150
    assert drift['status'] in {'stable', 'warning', 'alert'}


def test_summary_before_any_prediction(monitoring_config: MonitoringConfig) -> None:
    summary = PredictionMonitor(monitoring_config).summary()

    assert summary['prediction_count'] == 0
    assert summary['flagged_rate'] is None
