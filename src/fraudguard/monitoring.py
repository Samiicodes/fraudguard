'''Prediction logging, service metrics, and input drift detection.

State lives in memory and in a local JSON Lines file. docs/architecture.md
describes how the same signals map to Prometheus and Grafana in production.
'''

import json
import logging
import threading
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from fraudguard.config import MonitoringConfig
from fraudguard.data_ingestion import AMOUNT_COLUMN, PCA_COLUMNS

logger = logging.getLogger(__name__)

# The raw time column is excluded: it only ever increases, so it always looks drifted
MONITORED_FEATURES = [AMOUNT_COLUMN, *PCA_COLUMNS]
_PSI_FLOOR = 1e-4


def build_reference_profile(frame: pd.DataFrame, bin_count: int) -> dict[str, Any]:
    '''Summarise the training distribution of each monitored feature.

    Bin edges are training quantiles, so every bin holds a similar share of the
    training rows and no sparse bin dominates the population stability index.
    '''
    inner_quantiles = np.linspace(0, 1, bin_count + 1)[1:-1]
    features: dict[str, Any] = {}
    for name in MONITORED_FEATURES:
        values = frame[name].dropna().to_numpy(dtype=np.float64)
        # Repeated values, common for amounts, can make neighbouring quantiles equal
        edges = np.unique(np.quantile(values, inner_quantiles))
        features[name] = {
            'edges': edges.tolist(),
            'expected': _bin_fractions(values, edges).tolist(),
            'mean': float(values.mean()),
            'std': float(values.std()),
        }
    return {'row_count': len(frame), 'features': features}


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    '''Return the PSI between two binned distributions given as fractions.

    Common reading: below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 major shift.
    '''
    # Floor empty bins so the logarithm stays finite
    expected = np.clip(np.asarray(expected, dtype=np.float64), _PSI_FLOOR, None)
    actual = np.clip(np.asarray(actual, dtype=np.float64), _PSI_FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_drift_report(
    reference_profile: dict[str, Any], recent: pd.DataFrame, config: MonitoringConfig
) -> dict[str, Any]:
    '''Compare recent inputs with the training profile, feature by feature.

    PSI on a small sample is noisy, so nothing is reported until min_drift_samples
    rows are available.
    '''
    if len(recent) < config.min_drift_samples:
        return {
            'status': 'insufficient_data',
            'sample_count': len(recent),
            'required_samples': config.min_drift_samples,
        }

    feature_psi = {
        name: population_stability_index(
            np.asarray(reference['expected']),
            _bin_fractions(recent[name].to_numpy(dtype=np.float64), np.asarray(reference['edges'])),
        )
        for name, reference in reference_profile['features'].items()
    }
    max_psi = max(feature_psi.values())
    if max_psi >= config.psi_alert_threshold:
        status = 'alert'
    elif max_psi >= config.psi_warning_threshold:
        status = 'warning'
    else:
        status = 'stable'
    drifted = {
        name: round(value, 4)
        for name, value in sorted(feature_psi.items(), key=lambda item: -item[1])
        if value >= config.psi_warning_threshold
    }
    return {
        'status': status,
        'sample_count': len(recent),
        'max_psi': round(max_psi, 4),
        'drifted_features': drifted,
    }


class PredictionMonitor:
    '''Collects prediction statistics and appends every prediction to a log file.

    Counters cover the lifetime of the process. Latency, input statistics, and
    drift are computed over a rolling window of the most recent predictions.
    '''

    def __init__(
        self, config: MonitoringConfig, reference_profile: dict[str, Any] | None = None
    ) -> None:
        self._config = config
        self._reference_profile = reference_profile
        self._lock = threading.Lock()
        self._window: deque[dict[str, float]] = deque(maxlen=config.window_size)
        self._prediction_count = 0
        self._flagged_count = 0
        self._started_at = datetime.now(UTC)
        config.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        request_id: str,
        inputs: dict[str, float],
        fraud_probability: float,
        is_flagged: bool,
        latency_ms: float,
        model_version: str,
    ) -> None:
        '''Record one prediction in memory and in the prediction log.'''
        entry = {
            'timestamp': datetime.now(UTC).isoformat(),
            'request_id': request_id,
            'model_version': model_version,
            'fraud_probability': fraud_probability,
            'is_flagged': is_flagged,
            'latency_ms': latency_ms,
            'inputs': inputs,
        }
        observation = {name: inputs[name] for name in MONITORED_FEATURES}
        observation.update(fraud_probability=fraud_probability, latency_ms=latency_ms)
        with self._lock:
            self._prediction_count += 1
            self._flagged_count += int(is_flagged)
            self._window.append(observation)
            self._append_to_log(entry)

    def summary(self) -> dict[str, Any]:
        '''Return counters, latency, input statistics, and drift status.'''
        with self._lock:
            window = pd.DataFrame(list(self._window))
            prediction_count, flagged_count = self._prediction_count, self._flagged_count

        summary: dict[str, Any] = {
            'started_at': self._started_at.isoformat(),
            'prediction_count': prediction_count,
            'flagged_count': flagged_count,
            'flagged_rate': flagged_count / prediction_count if prediction_count else None,
            'window_size': len(window),
        }
        if window.empty:
            return summary

        latency = window['latency_ms']
        summary['mean_fraud_probability'] = float(window['fraud_probability'].mean())
        summary['latency_ms'] = {
            'mean': float(latency.mean()),
            'p50': float(latency.quantile(0.5)),
            'p95': float(latency.quantile(0.95)),
            'max': float(latency.max()),
        }
        statistics = window[MONITORED_FEATURES].agg(['mean', 'std', 'min', 'max']).round(4)
        # A single observation has no standard deviation; JSON has no NaN
        summary['input_statistics'] = (
            statistics.astype(object).where(statistics.notna(), None).to_dict()
        )
        if self._reference_profile is not None:
            summary['drift'] = compute_drift_report(self._reference_profile, window, self._config)
        return summary

    def _append_to_log(self, entry: dict[str, Any]) -> None:
        '''Append one JSON line. A failed write must not fail the prediction.'''
        try:
            with self._config.prediction_log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry) + '\n')
        except OSError:
            logger.exception('Could not write to the prediction log')


def _bin_fractions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    '''Return the share of values falling into each bin defined by the inner edges.'''
    counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
    return counts / max(int(counts.sum()), 1)
