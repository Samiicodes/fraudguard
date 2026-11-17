'''Tests for configuration loading, validation, and environment overrides.'''

from pathlib import Path

import pytest
from pydantic import ValidationError

from fraudguard.config import (
    CandidateConfig,
    EnvironmentSettings,
    FeatureConfig,
    SplitConfig,
    ThresholdConfig,
    TrainingConfig,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> EnvironmentSettings:
    '''Settings that ignore any local .env file and inherited overrides.'''
    for name in ('MLFLOW_TRACKING_URI', 'FRAUDGUARD_API_HOST', 'FRAUDGUARD_API_PORT'):
        monkeypatch.delenv(name, raising=False)
    return EnvironmentSettings(_env_file=None)


def test_project_config_is_valid(settings: EnvironmentSettings) -> None:
    config = load_config(CONFIG_PATH, settings)

    names = {candidate.name for candidate in config.training.candidates}
    assert {'logistic_regression', 'random_forest', 'xgboost'} <= names
    assert config.training.selection_metric == 'pr_auc'
    assert config.mlflow.tracking_uri == 'sqlite:///mlflow.db'


def test_environment_overrides_single_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    monkeypatch.setenv('FRAUDGUARD_API_PORT', '9000')

    config = load_config(CONFIG_PATH, EnvironmentSettings(_env_file=None))

    assert config.mlflow.tracking_uri == 'http://mlflow:5000'
    assert config.api.port == 9000
    # Settings that were not overridden keep their file values
    assert config.mlflow.experiment_name == 'fraudguard'
    assert config.api.host == '127.0.0.1'


def test_missing_config_file_raises(tmp_path: Path, settings: EnvironmentSettings) -> None:
    with pytest.raises(FileNotFoundError, match=r'missing\.yaml'):
        load_config(tmp_path / 'missing.yaml', settings)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match='false_postive_cost'):
        ThresholdConfig.model_validate(
            {
                'false_postive_cost': 5.0,
                'false_negative_cost': 150.0,
                'max_false_positive_rate': 0.001,
            }
        )


def test_class_weight_is_rejected_for_xgboost() -> None:
    with pytest.raises(ValidationError, match='scale_pos_weight instead'):
        CandidateConfig(name='booster', algorithm='xgboost', imbalance_strategy='class_weight')


def test_scale_pos_weight_requires_xgboost() -> None:
    with pytest.raises(ValidationError, match='only supported by xgboost'):
        CandidateConfig(
            name='forest', algorithm='random_forest', imbalance_strategy='scale_pos_weight'
        )


def test_candidate_names_must_be_unique() -> None:
    candidate = CandidateConfig(
        name='baseline', algorithm='logistic_regression', imbalance_strategy='class_weight'
    )

    with pytest.raises(ValidationError, match='unique'):
        TrainingConfig(
            random_seed=0,
            selection_metric='pr_auc',
            smote_sampling_ratio=0.1,
            candidates=(candidate, candidate),
        )


def test_splits_must_leave_rows_for_training() -> None:
    with pytest.raises(ValidationError, match='less than 1'):
        SplitConfig(validation_size=0.5, test_size=0.5, random_seed=0)


def test_amount_bucket_edges_must_increase() -> None:
    with pytest.raises(ValidationError, match='strictly increasing'):
        FeatureConfig(amount_bucket_edges=(10.0, 5.0))
