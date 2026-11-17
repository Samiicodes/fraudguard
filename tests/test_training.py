'''Tests for model construction, imbalance handling, and the end-to-end training pipeline.'''

import json

import mlflow
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ResamplingPipeline
from mlflow import MlflowClient

from fraudguard.config import AppConfig, CandidateConfig
from fraudguard.data_ingestion import INPUT_COLUMNS, RAW_COLUMN_MAPPING, TARGET_COLUMN
from fraudguard.data_validation import DataValidationError
from fraudguard.feature_engineering import MODEL_FEATURES
from fraudguard.reporting import FIGURE_DIR_NAME, FIGURE_FILES
from fraudguard.training import TrainingSummary, build_classifier, prepare_data

TrainedPipeline = tuple[AppConfig, TrainingSummary]


@pytest.mark.parametrize(
    ('algorithm', 'strategy', 'parameter', 'expected'),
    [
        ('logistic_regression', 'class_weight', 'class_weight', 'balanced'),
        ('logistic_regression', 'none', 'class_weight', None),
        ('random_forest', 'class_weight', 'class_weight', 'balanced_subsample'),
        ('xgboost', 'scale_pos_weight', 'scale_pos_weight', 99.0),
        ('xgboost', 'none', 'scale_pos_weight', 1.0),
    ],
)
def test_imbalance_strategy_sets_the_matching_parameter(
    app_config: AppConfig, algorithm: str, strategy: str, parameter: str, expected: object
) -> None:
    candidate = CandidateConfig.model_validate(
        {'name': 'candidate', 'algorithm': algorithm, 'imbalance_strategy': strategy}
    )

    model = build_classifier(candidate, app_config.training, scale_pos_weight=99.0)

    assert model.get_params()[parameter] == expected


def test_smote_is_only_applied_while_fitting(
    app_config: AppConfig, transactions: pd.DataFrame
) -> None:
    candidate = CandidateConfig(
        name='smote',
        algorithm='xgboost',
        imbalance_strategy='smote',
        params={'n_estimators': 10},
    )
    features, target = transactions[INPUT_COLUMNS], transactions[TARGET_COLUMN]

    model = build_classifier(candidate, app_config.training, scale_pos_weight=1.0)
    model.fit(features, target)
    _, resampled_target = model.named_steps['smote'].fit_resample(features, target)

    assert isinstance(model, ResamplingPipeline)
    # The sampler adds synthetic fraud rows during fit, but prediction sees the input unchanged
    assert resampled_target.sum() > target.sum()
    assert model.predict_proba(features.head(50)).shape == (50, 2)


def test_pipeline_stops_when_validation_fails(
    app_config: AppConfig, transactions: pd.DataFrame
) -> None:
    raw_names = {canonical: raw for raw, canonical in RAW_COLUMN_MAPPING.items()}
    app_config.data.sample_path.parent.mkdir(parents=True)
    transactions.rename(columns=raw_names).drop(columns=['Amount']).to_csv(
        app_config.data.sample_path, index=False
    )

    with pytest.raises(DataValidationError, match='schema'):
        prepare_data(app_config, use_sample_data=True)


def test_champion_is_registered_with_its_alias_and_threshold(
    trained_pipeline: TrainedPipeline,
) -> None:
    config, summary = trained_pipeline
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)

    version = client.get_model_version_by_alias(
        config.mlflow.registered_model_name, config.mlflow.champion_alias
    )

    assert str(version.version) == summary.model_version
    assert version.tags['candidate_name'] == summary.champion_name
    assert float(version.tags['decision_threshold']) == pytest.approx(summary.decision_threshold)


def test_every_candidate_is_tracked_as_a_run(trained_pipeline: TrainedPipeline) -> None:
    config, summary = trained_pipeline
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    assert experiment is not None

    runs = {run.info.run_name: run for run in client.search_runs([experiment.experiment_id])}

    assert set(runs) == {candidate.name for candidate in config.training.candidates}
    for run in runs.values():
        assert {'validation_pr_auc', 'validation_recall'} <= set(run.data.metrics)
        assert 'imbalance_strategy' in run.data.params
    # Only the champion is ever scored on the test split
    test_scored = [name for name, run in runs.items() if 'test_pr_auc' in run.data.metrics]
    assert test_scored == [summary.champion_name]


def test_smote_candidate_is_served_without_the_sampler(trained_pipeline: TrainedPipeline) -> None:
    config, _ = trained_pipeline
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    assert experiment is not None
    smote_run = next(
        run
        for run in client.search_runs([experiment.experiment_id])
        if run.data.params['imbalance_strategy'] == 'smote'
    )

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    served = mlflow.sklearn.load_model(f'runs:/{smote_run.info.run_id}/model')

    assert [name for name, _ in served.steps] == ['preprocessor', 'classifier']


def test_report_figures_and_metrics_are_written(trained_pipeline: TrainedPipeline) -> None:
    config, summary = trained_pipeline
    report_dir = config.reports.output_dir

    report = summary.report_path.read_text(encoding='utf-8')
    metrics = json.loads((report_dir / 'metrics.json').read_text(encoding='utf-8'))

    assert 'synthetic sample data' in report
    assert '## Threshold selection' in report
    for file_name in FIGURE_FILES.values():
        assert (report_dir / FIGURE_DIR_NAME / file_name).is_file()
    assert metrics['champion']['name'] == summary.champion_name
    assert metrics['test']['pr_auc'] == pytest.approx(summary.test_pr_auc)
    assert 0.0 < metrics['champion']['decision_threshold'] < 1.0


def test_selected_threshold_fits_the_review_budget(trained_pipeline: TrainedPipeline) -> None:
    config, _ = trained_pipeline
    metrics = json.loads((config.reports.output_dir / 'metrics.json').read_text(encoding='utf-8'))

    assert metrics['champion']['review_budget_met']
    assert metrics['validation']['false_positive_rate'] <= config.threshold.max_false_positive_rate


def test_saved_splits_share_no_transactions(trained_pipeline: TrainedPipeline) -> None:
    config, _ = trained_pipeline
    splits = [
        pd.read_parquet(config.data.processed_dir / f'{name}.parquet')
        for name in ('train', 'validation', 'test')
    ]

    ids = [set(split.index) for split in splits]
    rows = [set(pd.util.hash_pandas_object(split[MODEL_FEATURES], index=False)) for split in splits]

    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert not ids[first] & ids[second]
        assert not rows[first] & rows[second]
