'''Typed configuration loaded from config/config.yaml and environment variables.'''

import logging
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Algorithm = Literal['logistic_regression', 'random_forest', 'xgboost']
ImbalanceStrategy = Literal['none', 'class_weight', 'scale_pos_weight', 'smote']
SelectionMetric = Literal['pr_auc', 'roc_auc']

CLI_LOG_FORMAT = '%(message)s'
SERVICE_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'

# Third-party loggers that print progress lines unrelated to the pipeline stages
_NOISY_LOGGERS = ('alembic', 'mlflow', 'httpx', 'matplotlib')


class StrictModel(BaseModel):
    '''Base class that rejects unknown keys so typos in the YAML file fail fast.'''

    model_config = ConfigDict(extra='forbid', frozen=True)


class DataConfig(StrictModel):
    '''Locations of raw and processed data.'''

    raw_path: Path
    sample_path: Path
    processed_dir: Path


class ValidationConfig(StrictModel):
    '''Tolerances used by the data validation checks.'''

    max_missing_fraction: float = Field(ge=0, le=1)
    max_invalid_fraction: float = Field(ge=0, le=1)
    min_fraud_rows: int = Field(ge=1)
    min_fraud_rate: float = Field(ge=0, le=1)
    max_fraud_rate: float = Field(ge=0, le=1)


class SplitConfig(StrictModel):
    '''Stratified split proportions.'''

    validation_size: float = Field(gt=0, lt=1)
    test_size: float = Field(gt=0, lt=1)
    random_seed: int

    @model_validator(mode='after')
    def _leave_room_for_training(self) -> Self:
        if self.validation_size + self.test_size >= 1:
            raise ValueError('validation_size and test_size must sum to less than 1')
        return self


class FeatureConfig(StrictModel):
    '''Settings for engineered features.'''

    amount_bucket_edges: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def _edges_increase(self) -> Self:
        edges = self.amount_bucket_edges
        if edges[0] <= 0 or any(low >= high for low, high in pairwise(edges)):
            raise ValueError('amount_bucket_edges must be positive and strictly increasing')
        return self


class CandidateConfig(StrictModel):
    '''One model candidate in the comparison.'''

    name: str = Field(min_length=1)
    algorithm: Algorithm
    imbalance_strategy: ImbalanceStrategy
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _strategy_matches_algorithm(self) -> Self:
        if self.imbalance_strategy == 'scale_pos_weight' and self.algorithm != 'xgboost':
            raise ValueError('scale_pos_weight is only supported by xgboost')
        if self.imbalance_strategy == 'class_weight' and self.algorithm == 'xgboost':
            raise ValueError('use scale_pos_weight instead of class_weight for xgboost')
        return self


class TrainingConfig(StrictModel):
    '''Model candidates and the rule used to rank them.'''

    random_seed: int
    selection_metric: SelectionMetric
    smote_sampling_ratio: float = Field(gt=0, le=1)
    candidates: tuple[CandidateConfig, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def _names_are_unique(self) -> Self:
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError('candidate names must be unique')
        return self


class ThresholdConfig(StrictModel):
    '''Business assumptions behind the decision threshold.'''

    false_positive_cost: float = Field(gt=0)
    false_negative_cost: float = Field(gt=0)
    max_false_positive_rate: float = Field(gt=0, le=1)


class EvaluationConfig(StrictModel):
    '''Settings for test-set evaluation.'''

    bootstrap_samples: int = Field(ge=0)
    calibration_bins: int = Field(ge=2)


class ExplainabilityConfig(StrictModel):
    '''Settings for SHAP and permutation importance.'''

    sample_size: int = Field(ge=10)
    top_features: int = Field(ge=1)
    permutation_repeats: int = Field(ge=1)


class MlflowConfig(StrictModel):
    '''Experiment tracking and model registry settings.'''

    tracking_uri: str
    artifact_dir: Path
    experiment_name: str
    registered_model_name: str
    champion_alias: str


class ReportsConfig(StrictModel):
    '''Location of generated reports and figures.'''

    output_dir: Path


class ApiConfig(StrictModel):
    '''Inference service settings.'''

    host: str
    port: int = Field(ge=1, le=65535)
    explanation_top_k: int = Field(ge=1)


class MonitoringConfig(StrictModel):
    '''Prediction logging and drift detection settings.'''

    prediction_log_path: Path
    window_size: int = Field(ge=1)
    min_drift_samples: int = Field(ge=1)
    psi_bins: int = Field(ge=2)
    psi_warning_threshold: float = Field(gt=0)
    psi_alert_threshold: float = Field(gt=0)


class AppConfig(StrictModel):
    '''Complete project configuration.'''

    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig
    features: FeatureConfig
    training: TrainingConfig
    threshold: ThresholdConfig
    evaluation: EvaluationConfig
    explainability: ExplainabilityConfig
    mlflow: MlflowConfig
    reports: ReportsConfig
    api: ApiConfig
    monitoring: MonitoringConfig


class EnvironmentSettings(BaseSettings):
    '''Deployment-specific overrides read from the environment or a .env file.'''

    model_config = SettingsConfigDict(
        env_prefix='FRAUDGUARD_', env_file='.env', extra='ignore', populate_by_name=True
    )

    config_path: Path = Path('config/config.yaml')
    log_level: str = 'INFO'
    # MLflow reads this variable itself, so it keeps its standard unprefixed name
    mlflow_tracking_uri: str | None = Field(default=None, validation_alias='MLFLOW_TRACKING_URI')
    mlflow_artifact_dir: Path | None = None
    api_host: str | None = None
    api_port: int | None = None
    prediction_log_path: Path | None = None

    def overrides(self) -> dict[str, dict[str, Any]]:
        '''Return the settings that were provided, grouped by config section.'''
        candidates: dict[str, dict[str, Any]] = {
            'mlflow': {
                'tracking_uri': self.mlflow_tracking_uri,
                'artifact_dir': self.mlflow_artifact_dir,
            },
            'api': {'host': self.api_host, 'port': self.api_port},
            'monitoring': {'prediction_log_path': self.prediction_log_path},
        }
        return {
            section: {key: value for key, value in values.items() if value is not None}
            for section, values in candidates.items()
        }


def load_config(
    config_path: Path | None = None, settings: EnvironmentSettings | None = None
) -> AppConfig:
    '''Load the YAML configuration and apply environment overrides.

    Args:
        config_path: YAML file to read. Defaults to the path in the environment settings.
        settings: Environment settings. Read from the process environment when omitted.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    '''
    settings = settings or EnvironmentSettings()
    path = config_path or settings.config_path
    if not path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    with path.open(encoding='utf-8') as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    for section, values in settings.overrides().items():
        raw[section] = {**raw.get(section, {}), **values}
    return AppConfig.model_validate(raw)


def configure_logging(level: str = 'INFO', log_format: str = CLI_LOG_FORMAT) -> None:
    '''Configure root logging and quieten chatty third-party loggers.'''
    logging.basicConfig(level=level.upper(), format=log_format, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # uv environments ship without pip, which MLflow warns about on every model save
    logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)
