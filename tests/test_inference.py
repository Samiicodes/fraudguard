'''Tests for loading the champion from the registry and scoring new transactions.'''

import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from fraudguard.config import AppConfig
from fraudguard.data_ingestion import (
    INPUT_COLUMNS,
    RAW_COLUMN_MAPPING,
    TARGET_COLUMN,
    generate_synthetic_transactions,
)
from fraudguard.feature_engineering import MODEL_FEATURES
from fraudguard.inference import FraudModel, load_champion_model
from fraudguard.training import TrainingSummary

TrainedPipeline = tuple[AppConfig, TrainingSummary]


@pytest.fixture(scope='module')
def champion(trained_pipeline: TrainedPipeline) -> FraudModel:
    config, _ = trained_pipeline
    return load_champion_model(config.mlflow)


@pytest.fixture(scope='module')
def unseen_transactions() -> pd.DataFrame:
    frame = generate_synthetic_transactions(
        n_rows=600, fraud_rate=0.1, duplicate_rate=0.0, random_seed=99
    )
    return frame.rename(columns=RAW_COLUMN_MAPPING)


def test_champion_loads_with_its_registry_metadata(
    champion: FraudModel, trained_pipeline: TrainedPipeline
) -> None:
    config, summary = trained_pipeline

    assert champion.model_name == config.mlflow.registered_model_name
    assert champion.model_version == summary.model_version
    assert champion.decision_threshold == pytest.approx(summary.decision_threshold)
    assert champion.reference_profile is not None
    assert {'amount', 'v14'} <= set(champion.reference_profile['features'])


def test_labels_follow_the_decision_threshold(
    champion: FraudModel, unseen_transactions: pd.DataFrame
) -> None:
    predictions = champion.predict(unseen_transactions[INPUT_COLUMNS])

    assert len(predictions) == len(unseen_transactions)
    for prediction in predictions:
        assert 0.0 <= prediction.fraud_probability <= 1.0
        assert prediction.is_fraud == (prediction.fraud_probability >= champion.decision_threshold)
        assert prediction.explanation is None


def test_scores_separate_fraud_from_legitimate_transactions(
    champion: FraudModel, unseen_transactions: pd.DataFrame
) -> None:
    predictions = champion.predict(unseen_transactions[INPUT_COLUMNS])
    scores = [prediction.fraud_probability for prediction in predictions]

    assert roc_auc_score(unseen_transactions[TARGET_COLUMN], scores) > 0.9


def test_contributions_add_up_to_the_model_output(
    champion: FraudModel, unseen_transactions: pd.DataFrame
) -> None:
    row = unseen_transactions[INPUT_COLUMNS].iloc[[0]]

    prediction = champion.predict(row, explanation_top_k=len(MODEL_FEATURES))[0]

    explanation = prediction.explanation
    assert explanation is not None
    assert len(explanation.contributions) == len(MODEL_FEATURES)
    total = explanation.base_value + sum(item.contribution for item in explanation.contributions)
    reconstructed = expit(total) if explanation.output_unit == 'log_odds' else total
    assert reconstructed == pytest.approx(prediction.fraud_probability, abs=1e-4)


def test_contributions_are_ordered_by_size(
    champion: FraudModel, unseen_transactions: pd.DataFrame
) -> None:
    row = unseen_transactions[INPUT_COLUMNS].iloc[[1]]

    explanation = champion.predict(row, explanation_top_k=5)[0].explanation

    assert explanation is not None
    sizes = [abs(item.contribution) for item in explanation.contributions]
    assert len(sizes) == 5
    assert sizes == sorted(sizes, reverse=True)
