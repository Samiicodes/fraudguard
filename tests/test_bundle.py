'''Tests for exporting the champion to a bundle and serving the API from it.'''

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.models import Model

from fraudguard.api import create_app
from fraudguard.bundle import (
    DEMO_GROUPS,
    MODEL_DIR,
    choose_demo_examples,
    export_champion,
    load_model_bundle,
    read_demo_examples,
    read_manifest,
    write_demo_examples,
)
from fraudguard.config import AppConfig
from fraudguard.data_ingestion import (
    INPUT_COLUMNS,
    RAW_COLUMN_MAPPING,
    TARGET_COLUMN,
    generate_synthetic_transactions,
)
from fraudguard.inference import FraudModel, FraudPrediction, load_champion_model
from fraudguard.training import TrainingSummary

TrainedPipeline = tuple[AppConfig, TrainingSummary]


class FixedDecisionModel:
    '''Stands in for a model whose decisions are known in advance.'''

    def __init__(self, flagged: list[bool]) -> None:
        self.flagged = flagged

    def predict(
        self, transactions: pd.DataFrame, explanation_top_k: int = 0
    ) -> list[FraudPrediction]:
        return [
            FraudPrediction(fraud_probability=float(flag), is_fraud=flag, explanation=None)
            for flag in self.flagged
        ]


def labelled_transactions(is_fraud: list[bool]) -> pd.DataFrame:
    '''Distinct transactions in the canonical schema with the given labels.'''
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(len(is_fraud), len(INPUT_COLUMNS))), columns=INPUT_COLUMNS
    )
    frame[TARGET_COLUMN] = [int(label) for label in is_fraud]
    return frame


def serving_config(config: AppConfig, bundle_dir: Path, tmp_path: Path) -> AppConfig:
    '''Point the service at a bundle and at an empty registry it must not need.'''
    registry = (tmp_path / 'empty.db').as_posix()
    return config.model_copy(
        update={
            'api': config.api.model_copy(update={'model_bundle': bundle_dir}),
            'mlflow': config.mlflow.model_copy(update={'tracking_uri': f'sqlite:///{registry}'}),
        }
    )


@pytest.fixture(scope='module')
def bundle_dir(trained_pipeline: TrainedPipeline, tmp_path_factory: pytest.TempPathFactory) -> Path:
    '''The test champion exported with demo examples from synthetic transactions.'''
    config, _ = trained_pipeline
    directory = tmp_path_factory.mktemp('bundle')
    export_champion(config.mlflow, directory)
    sample = generate_synthetic_transactions(n_rows=2000, fraud_rate=0.05, random_seed=3)
    examples = choose_demo_examples(
        sample.rename(columns=RAW_COLUMN_MAPPING), load_model_bundle(directory)
    )
    write_demo_examples(examples, directory)
    return directory


def test_bundle_scores_like_the_registry_model(
    trained_pipeline: TrainedPipeline, bundle_dir: Path, transactions: pd.DataFrame
) -> None:
    config, _ = trained_pipeline
    registry_model = load_champion_model(config.mlflow)

    bundled_model = load_model_bundle(bundle_dir)

    def scores(model: FraudModel) -> list[float]:
        return [item.fraud_probability for item in model.predict(transactions[INPUT_COLUMNS])]

    assert scores(bundled_model) == scores(registry_model)
    assert bundled_model.decision_threshold == registry_model.decision_threshold
    assert bundled_model.model_version == registry_model.model_version
    assert bundled_model.reference_profile == registry_model.reference_profile


def test_manifest_records_the_version_and_test_metrics(
    trained_pipeline: TrainedPipeline, bundle_dir: Path
) -> None:
    _, summary = trained_pipeline

    manifest = read_manifest(bundle_dir)

    assert manifest.model_version == summary.model_version
    assert manifest.algorithm in {'logistic_regression', 'random_forest', 'xgboost'}
    assert manifest.dataset_sha256
    assert manifest.test_metrics['pr_auc'] == pytest.approx(summary.test_pr_auc)
    assert manifest.test_metrics['recall'] == pytest.approx(summary.test_recall)


def test_bundle_does_not_record_paths_from_the_exporting_machine(
    trained_pipeline: TrainedPipeline, bundle_dir: Path
) -> None:
    config, _ = trained_pipeline
    descriptor = (bundle_dir / MODEL_DIR / 'MLmodel').read_text(encoding='utf-8')

    assert Model.load(bundle_dir / MODEL_DIR).artifact_path == MODEL_DIR
    assert config.mlflow.artifact_dir.as_posix() not in descriptor


def test_export_does_not_overwrite_a_bundle(
    trained_pipeline: TrainedPipeline, bundle_dir: Path
) -> None:
    config, _ = trained_pipeline

    with pytest.raises(FileExistsError):
        export_champion(config.mlflow, bundle_dir)


def test_loading_a_directory_that_is_not_a_bundle_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='manifest'):
        load_model_bundle(tmp_path)


def test_demo_examples_cover_every_outcome_in_order() -> None:
    is_fraud = [True] * 6 + [False] * 6
    flagged = [True, False, True, False, True, True, False, True, False, False, True, False]
    transactions = labelled_transactions(is_fraud)

    examples = choose_demo_examples(transactions, cast(FraudModel, FixedDecisionModel(flagged)))

    expected_groups = [group for group, _, _, count in DEMO_GROUPS for _ in range(count)]
    assert [example.group for example in examples] == expected_groups
    outcomes = {group: (fraud, flag) for group, fraud, flag, _ in DEMO_GROUPS}
    for example in examples:
        row = next(
            index
            for index in range(len(transactions))
            if transactions[INPUT_COLUMNS].iloc[index].tolist()
            == [example.transaction[column] for column in INPUT_COLUMNS]
        )
        fraud, flag = outcomes[example.group]
        assert (is_fraud[row], flagged[row]) == (fraud, flag)
        assert example.actual_label == ('fraud' if fraud else 'legitimate')


def test_demo_examples_are_reproducible_and_skip_empty_groups() -> None:
    transactions = labelled_transactions([False] * 8)
    model = cast(FraudModel, FixedDecisionModel([False] * 8))

    first = choose_demo_examples(transactions, model, seed=5)
    second = choose_demo_examples(transactions, model, seed=5)

    assert first == second
    assert {example.group for example in first} == {'Legitimate payment, approved'}


def test_demo_examples_survive_a_round_trip(tmp_path: Path) -> None:
    transactions = labelled_transactions([True, False])
    examples = choose_demo_examples(
        transactions, cast(FraudModel, FixedDecisionModel([True, False]))
    )

    write_demo_examples(examples, tmp_path)

    assert read_demo_examples(tmp_path) == examples
    assert read_demo_examples(tmp_path / 'elsewhere') == []


def test_service_serves_the_bundle_without_the_registry(
    trained_pipeline: TrainedPipeline, bundle_dir: Path, tmp_path: Path
) -> None:
    config, summary = trained_pipeline
    app = create_app(serving_config(config, bundle_dir, tmp_path))

    with TestClient(app) as client:
        health = client.get('/health').json()
        demo = client.get('/demo/examples').json()
        predictions = [
            client.post('/predict', json=example['transaction']).json()
            for example in demo['examples']
        ]

    assert health['model_version'] == summary.model_version
    assert demo['algorithm'] == read_manifest(bundle_dir).algorithm
    assert demo['test_metrics']['pr_auc'] == pytest.approx(summary.test_pr_auc)
    assert demo['examples']
    # Each example was chosen for the decision the served model makes on it
    flags = {group: flag for group, _, flag, _ in DEMO_GROUPS}
    for example, prediction in zip(demo['examples'], predictions, strict=True):
        assert (prediction['predicted_label'] == 'fraud') == flags[example['group']]


def test_service_reports_a_missing_bundle(
    app_config: AppConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = serving_config(app_config, tmp_path / 'missing', tmp_path)

    with caplog.at_level(logging.ERROR), TestClient(create_app(config)) as client:
        health = client.get('/health')
        demo = client.get('/demo/examples').json()

    assert health.status_code == 503
    assert demo['examples'] == []
    assert 'Could not load the model bundle' in caplog.text


def test_demo_page_is_served(trained_pipeline: TrainedPipeline) -> None:
    config, _ = trained_pipeline

    with TestClient(create_app(config)) as client:
        page = client.get('/')
        demo = client.get('/demo/examples').json()
        schema = client.get('/openapi.json').json()

    assert page.status_code == 200
    assert page.headers['content-type'].startswith('text/html')
    assert 'demo/examples' in page.text
    # A model from the registry has no bundled examples, and the page is not an API route
    assert demo == {'algorithm': None, 'test_metrics': {}, 'examples': []}
    assert '/' not in schema['paths']
