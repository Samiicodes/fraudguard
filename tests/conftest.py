'''Shared fixtures: a small test configuration and a pipeline trained once on synthetic data.'''

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from fraudguard.config import AppConfig
from fraudguard.data_ingestion import RAW_COLUMN_MAPPING, generate_synthetic_transactions
from fraudguard.training import TrainingSummary, run_training_pipeline

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'


def build_test_config(root: Path) -> AppConfig:
    '''Return the project configuration with small models and every path under root.'''
    raw: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    raw['data'] = {
        'raw_path': str(root / 'raw' / 'creditcard.csv'),
        'sample_path': str(root / 'raw' / 'sample_transactions.csv'),
        'processed_dir': str(root / 'processed'),
    }
    for candidate in raw['training']['candidates']:
        if 'n_estimators' in candidate['params']:
            candidate['params']['n_estimators'] = 30
    raw['evaluation']['bootstrap_samples'] = 30
    raw['explainability'].update(sample_size=400, permutation_repeats=1)
    database = (root / 'mlflow.db').as_posix()
    raw['mlflow'].update(
        tracking_uri=f'sqlite:///{database}', artifact_dir=str(root / 'mlartifacts')
    )
    raw['reports']['output_dir'] = str(root / 'reports')
    raw['monitoring']['prediction_log_path'] = str(root / 'monitoring' / 'predictions.jsonl')
    return AppConfig.model_validate(raw)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    '''Test configuration whose paths live in a fresh temporary directory.'''
    return build_test_config(tmp_path)


@pytest.fixture
def transactions() -> pd.DataFrame:
    '''Synthetic transactions in the canonical schema, without duplicates.'''
    frame = generate_synthetic_transactions(
        n_rows=3000, fraud_rate=0.03, duplicate_rate=0.0, random_seed=7
    )
    return frame.rename(columns=RAW_COLUMN_MAPPING)


@pytest.fixture(scope='session')
def trained_pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[AppConfig, TrainingSummary]:
    '''Run the full training pipeline once on synthetic data and share the result.'''
    config = build_test_config(tmp_path_factory.mktemp('pipeline'))
    sample = generate_synthetic_transactions(n_rows=8000, fraud_rate=0.02, random_seed=11)
    config.data.sample_path.parent.mkdir(parents=True)
    sample.to_csv(config.data.sample_path, index=False)
    return config, run_training_pipeline(config, use_sample_data=True)
