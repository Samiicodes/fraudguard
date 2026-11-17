'''Tests for engineered features and leakage-free preprocessing.'''

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from fraudguard.config import FeatureConfig, SplitConfig
from fraudguard.data_ingestion import (
    INPUT_COLUMNS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    TARGET_COLUMN,
    DatasetSplits,
    split_transactions,
)
from fraudguard.feature_engineering import (
    MODEL_FEATURES,
    PREPROCESSOR_FILE_NAME,
    TransactionFeatureBuilder,
    build_preprocessor,
    fit_preprocessor,
    save_processed_data,
)

FEATURES = FeatureConfig(amount_bucket_edges=(1, 10, 50, 100, 500, 1000))
SPLIT = SplitConfig(validation_size=0.2, test_size=0.2, random_seed=0)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    builder = TransactionFeatureBuilder(FEATURES.amount_bucket_edges)
    return builder.fit(frame).transform(frame)


def test_builder_returns_model_features_without_raw_time(transactions: pd.DataFrame) -> None:
    features = build_features(transactions[INPUT_COLUMNS])

    assert list(features.columns) == MODEL_FEATURES
    assert 'time_seconds' not in features.columns
    assert len(features) == len(transactions)


def test_hour_encoding_wraps_around_midnight(transactions: pd.DataFrame) -> None:
    frame = transactions[INPUT_COLUMNS].head(4).copy()
    frame['time_seconds'] = [0.0, SECONDS_PER_DAY, SECONDS_PER_DAY - 1, 6 * SECONDS_PER_HOUR]

    encoded = build_features(frame)[['hour_sin', 'hour_cos']].to_numpy()
    midnight, next_midnight, just_before_midnight, six_am = encoded

    np.testing.assert_allclose(midnight, next_midnight, atol=1e-12)
    assert np.linalg.norm(just_before_midnight - midnight) < 0.01
    np.testing.assert_allclose(six_am, [1.0, 0.0], atol=1e-12)


def test_amount_buckets_follow_the_configured_edges(transactions: pd.DataFrame) -> None:
    frame = transactions[INPUT_COLUMNS].head(7).copy()
    frame['amount'] = [0.0, 0.99, 1.0, 9.99, 10.0, 999.99, 5000.0]

    buckets = build_features(frame)['amount_bucket'].tolist()

    assert buckets == [0, 0, 1, 1, 2, 5, 6]


def test_log_amount_is_finite_for_zero_amounts(transactions: pd.DataFrame) -> None:
    frame = transactions[INPUT_COLUMNS].head(2).copy()
    frame['amount'] = [0.0, 100.0]

    np.testing.assert_allclose(build_features(frame)['amount_log'], [0.0, np.log1p(100.0)])


def test_missing_amount_stays_missing_until_imputation(transactions: pd.DataFrame) -> None:
    frame = transactions[INPUT_COLUMNS].head(3).copy()
    frame.loc[frame.index[0], 'amount'] = np.nan

    features = build_features(frame)

    assert features[['amount_log', 'amount_bucket']].iloc[0].isna().all()
    assert features[['amount_log', 'amount_bucket']].iloc[1:].notna().all().all()


def test_builder_rejects_frames_without_the_input_columns(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match='amount'):
        build_features(transactions[INPUT_COLUMNS].drop(columns=['amount']))


def test_transform_before_fit_raises(transactions: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        TransactionFeatureBuilder().transform(transactions[INPUT_COLUMNS])


def test_preprocessor_statistics_come_from_the_training_split(
    transactions: pd.DataFrame,
) -> None:
    splits = split_transactions(transactions, SPLIT)
    # Held-out splits that look nothing like training must not move the fitted statistics
    shifted = DatasetSplits(
        train=splits.train,
        validation=splits.validation.assign(amount=splits.validation['amount'] * 100),
        test=splits.test.assign(v1=splits.test['v1'] + 50),
    )
    preprocessor = build_preprocessor(FEATURES)

    processed = fit_preprocessor(preprocessor, shifted)

    train_features = build_features(splits.train[INPUT_COLUMNS])
    scaler = preprocessor.named_steps['scaler']
    np.testing.assert_allclose(scaler.mean_, train_features.mean().to_numpy())
    np.testing.assert_allclose(processed.train.features.mean(), 0.0, atol=1e-9)
    assert processed.validation.features['amount_log'].mean() > 1.0
    assert processed.test.features['v1'].mean() > 10.0


def test_imputer_uses_training_medians(transactions: pd.DataFrame) -> None:
    splits = split_transactions(transactions, SPLIT)
    test = splits.test.copy()
    test.loc[test.index[0], 'v2'] = np.nan
    preprocessor = build_preprocessor(FEATURES)

    processed = fit_preprocessor(preprocessor, DatasetSplits(splits.train, splits.validation, test))

    imputer = preprocessor.named_steps['imputer']
    v2_median = imputer.statistics_[MODEL_FEATURES.index('v2')]
    assert v2_median == pytest.approx(splits.train['v2'].median())
    assert not processed.test.features.isna().any().any()


def test_processed_data_and_preprocessor_are_saved(
    transactions: pd.DataFrame, tmp_path: Path
) -> None:
    splits = split_transactions(transactions, SPLIT)
    preprocessor = build_preprocessor(FEATURES)
    processed = fit_preprocessor(preprocessor, splits)

    save_processed_data(processed, preprocessor, tmp_path)

    for name in ('train', 'validation', 'test'):
        saved = pd.read_parquet(tmp_path / f'{name}.parquet')
        assert list(saved.columns) == [*MODEL_FEATURES, TARGET_COLUMN]
    reloaded = joblib.load(tmp_path / PREPROCESSOR_FILE_NAME)
    pd.testing.assert_frame_equal(
        reloaded.transform(splits.test[INPUT_COLUMNS]), processed.test.features
    )
