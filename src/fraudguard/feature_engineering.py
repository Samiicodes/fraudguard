'''Feature engineering and preprocessing, fitted on the training split only.'''

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from fraudguard.config import FeatureConfig
from fraudguard.data_ingestion import (
    AMOUNT_COLUMN,
    INPUT_COLUMNS,
    PCA_COLUMNS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    TARGET_COLUMN,
    TIME_COLUMN,
    DatasetSplits,
)

ENGINEERED_COLUMNS = ['amount_log', 'amount_bucket', 'hour_sin', 'hour_cos']
MODEL_FEATURES = [*PCA_COLUMNS, *ENGINEERED_COLUMNS]
PREPROCESSOR_FILE_NAME = 'preprocessor.joblib'


class TransactionFeatureBuilder(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    '''Derive model features from the canonical transaction columns.

    The raw time column is dropped. It counts seconds since the first row of the
    dataset, so its absolute value carries no meaning for future transactions.
    Only the position within the day is kept.
    '''

    def __init__(self, amount_bucket_edges: Sequence[float] = (1, 10, 50, 100, 500, 1000)) -> None:
        self.amount_bucket_edges = amount_bucket_edges

    def fit(self, frame: pd.DataFrame, target: object = None) -> Self:
        '''Record the input columns. The transformation itself has no fitted state.'''
        _require_columns(frame)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        '''Return the PCA components plus amount and time-of-day features.'''
        check_is_fitted(self, 'n_features_in_')
        _require_columns(frame)
        amount = frame[AMOUNT_COLUMN]
        hour = (frame[TIME_COLUMN] % SECONDS_PER_DAY) / SECONDS_PER_HOUR
        angle = 2 * np.pi * hour / 24

        features = frame[PCA_COLUMNS].copy()
        features['amount_log'] = np.log1p(amount.clip(lower=0))
        features['amount_bucket'] = self._bucket(amount)
        # Sine and cosine keep 23:00 and 00:00 close together
        features['hour_sin'] = np.sin(angle)
        features['hour_cos'] = np.cos(angle)
        return features

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        '''Return the output column names, as required by the scikit-learn output API.'''
        return np.asarray(MODEL_FEATURES, dtype=object)

    def _bucket(self, amount: pd.Series) -> np.ndarray:
        '''Map each amount to the ordinal position of its bucket.'''
        bucket = np.searchsorted(self.amount_bucket_edges, amount, side='right').astype(np.float64)
        # Keep missing amounts missing so the imputer handles them
        bucket[amount.isna().to_numpy()] = np.nan
        return bucket


@dataclass(frozen=True)
class ModelData:
    '''Model-ready features and the matching target for one split.'''

    features: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class ProcessedSplits:
    '''Model-ready data for the train, validation, and test splits.'''

    train: ModelData
    validation: ModelData
    test: ModelData


def build_preprocessor(config: FeatureConfig) -> Pipeline:
    '''Build the unfitted preprocessing pipeline shared by all model candidates.'''
    steps = [
        ('feature_builder', TransactionFeatureBuilder(config.amount_bucket_edges)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    return Pipeline(steps).set_output(transform='pandas')


def fit_preprocessor(preprocessor: Pipeline, splits: DatasetSplits) -> ProcessedSplits:
    '''Fit the preprocessor on the training split and transform all three splits.'''
    # Fit on train only so validation and test statistics cannot leak into training
    train_features = preprocessor.fit_transform(splits.train[INPUT_COLUMNS])
    return ProcessedSplits(
        train=ModelData(train_features, splits.train[TARGET_COLUMN]),
        validation=_transform_split(preprocessor, splits.validation),
        test=_transform_split(preprocessor, splits.test),
    )


def save_processed_data(
    processed: ProcessedSplits, preprocessor: Pipeline, output_dir: Path
) -> None:
    '''Write the model-ready splits as Parquet and the fitted preprocessor as joblib.'''
    output_dir.mkdir(parents=True, exist_ok=True)
    named_splits = {
        'train': processed.train,
        'validation': processed.validation,
        'test': processed.test,
    }
    for name, split in named_splits.items():
        split.features.assign(**{TARGET_COLUMN: split.target}).to_parquet(
            output_dir / f'{name}.parquet'
        )
    joblib.dump(preprocessor, output_dir / PREPROCESSOR_FILE_NAME)


def _transform_split(preprocessor: Pipeline, split: pd.DataFrame) -> ModelData:
    return ModelData(preprocessor.transform(split[INPUT_COLUMNS]), split[TARGET_COLUMN])


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError('Missing input columns: ' + ', '.join(missing))
