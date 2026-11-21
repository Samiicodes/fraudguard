'''Load the registered champion model and score transactions.'''

from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import Model
from sklearn.pipeline import Pipeline

from fraudguard.config import MlflowConfig
from fraudguard.data_ingestion import INPUT_COLUMNS
from fraudguard.explainability import Explanation, ShapExplainer
from fraudguard.feature_engineering import MODEL_FEATURES

PREPROCESSOR_STEP = 'preprocessor'
CLASSIFIER_STEP = 'classifier'


@dataclass(frozen=True)
class FraudPrediction:
    '''Score, decision, and optional explanation for one transaction.'''

    fraud_probability: float
    is_fraud: bool
    explanation: Explanation | None


def build_serving_pipeline(preprocessor: Pipeline, classifier: Any) -> Pipeline:
    '''Chain a fitted preprocessor and classifier into the model that is served.'''
    return Pipeline([(PREPROCESSOR_STEP, preprocessor), (CLASSIFIER_STEP, classifier)])


class FraudModel:
    '''A serving pipeline with its decision threshold and registry metadata.'''

    def __init__(
        self,
        pipeline: Pipeline,
        decision_threshold: float,
        model_name: str,
        model_version: str,
        reference_profile: dict[str, Any] | None = None,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.model_name = model_name
        self.model_version = model_version
        self.reference_profile = reference_profile
        self._preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
        self._classifier = pipeline.named_steps[CLASSIFIER_STEP]
        # Building a tree explainer parses every tree, so it happens once at load time
        self._explainer = ShapExplainer(self._classifier, MODEL_FEATURES)

    def predict(
        self, transactions: pd.DataFrame, explanation_top_k: int = 0
    ) -> list[FraudPrediction]:
        '''Score transactions given in the canonical input schema.

        Args:
            transactions: One row per transaction with the columns in INPUT_COLUMNS.
            explanation_top_k: Feature contributions to return per transaction.
                Zero skips the explanation.
        '''
        features = self._preprocessor.transform(transactions[INPUT_COLUMNS])
        probabilities = self._classifier.predict_proba(features)[:, 1]
        return [
            FraudPrediction(
                fraud_probability=float(probability),
                # Same rule as threshold selection: flag when the score reaches the threshold
                is_fraud=bool(probability >= self.decision_threshold),
                explanation=self._explainer.explain_row(features.iloc[[row]], explanation_top_k)
                if explanation_top_k > 0
                else None,
            )
            for row, probability in enumerate(probabilities)
        ]


def load_champion_model(config: MlflowConfig) -> FraudModel:
    '''Load the model version that the champion alias points to.

    The alias is resolved to a version number before anything is downloaded, so
    the reported version matches the loaded artefact even if the alias moves.

    Raises:
        mlflow.exceptions.MlflowException: If the model or the alias does not exist.
    '''
    mlflow.set_tracking_uri(config.tracking_uri)
    version = MlflowClient().get_model_version_by_alias(
        config.registered_model_name, config.champion_alias
    )
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f'models:/{config.registered_model_name}/{version.version}'
    )
    return model_from_artifacts(local_path, config.registered_model_name, str(version.version))


def model_from_artifacts(local_path: str, model_name: str, model_version: str) -> FraudModel:
    '''Build a FraudModel from a downloaded MLflow model directory.

    The decision threshold and drift reference profile come from the model metadata
    written at registration time.
    '''
    metadata = Model.load(local_path).metadata or {}
    return FraudModel(
        pipeline=mlflow.sklearn.load_model(local_path),
        decision_threshold=float(metadata['decision_threshold']),
        model_name=model_name,
        model_version=model_version,
        reference_profile=metadata.get('reference_profile'),
    )
