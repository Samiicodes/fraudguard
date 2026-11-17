'''Load raw transactions, map them to the canonical schema, and split them.'''

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraudguard.config import SplitConfig

TIME_COLUMN = 'time_seconds'
AMOUNT_COLUMN = 'amount'
TARGET_COLUMN = 'is_fraud'
PCA_COLUMNS = [f'v{index}' for index in range(1, 29)]
INPUT_COLUMNS = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]
REQUIRED_COLUMNS = [*INPUT_COLUMNS, TARGET_COLUMN]

# Header of the public creditcard.csv file mapped to the internal column names
RAW_COLUMN_MAPPING = {
    'Time': TIME_COLUMN,
    **{f'V{index}': f'v{index}' for index in range(1, 29)},
    'Amount': AMOUNT_COLUMN,
    'Class': TARGET_COLUMN,
}

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400

# Rough spread of each PCA component in the public dataset
_COMPONENT_SCALES = np.linspace(1.9, 0.35, len(PCA_COLUMNS))
# Zero-based positions of the components where fraud differs most, with the mean shift
_FRAUD_COMPONENT_SHIFTS = {
    2: -4.0,
    3: 3.5,
    9: -4.5,
    10: 3.0,
    11: -5.0,
    13: -6.0,
    15: -3.5,
    16: -5.5,
}
# Relative volume of legitimate transactions per hour of day, with a dip at night
_HOURLY_ACTIVITY = np.array([
    0.5, 0.25, 0.15, 0.12, 0.12, 0.2, 0.4, 0.7, 1.0, 1.1, 1.1, 1.1,
    1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.8, 0.65,
])  # fmt: skip
_CARD_TESTING_SHARE = 0.4
_ZERO_AMOUNT_SHARE = 0.006
_MAX_AMOUNT = 25_000.0


@dataclass(frozen=True)
class DatasetSplits:
    '''Train, validation, and test partitions of the cleaned dataset.'''

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_raw_transactions(path: Path) -> pd.DataFrame:
    '''Read a raw transaction CSV and rename its columns to the canonical schema.

    The row position in the file becomes the transaction_id index, so any processed
    row can be traced back to its source line.

    Raises:
        FileNotFoundError: If the file does not exist.
    '''
    if not path.is_file():
        raise FileNotFoundError(
            f'Dataset not found: {path}. Download creditcard.csv as described in the README '
            'or generate sample data with scripts/generate_sample_data.py.'
        )
    frame = pd.read_csv(path).rename(columns=RAW_COLUMN_MAPPING)
    frame.index.name = 'transaction_id'
    return frame


def compute_file_sha256(path: Path) -> str:
    '''Return the SHA-256 digest of a file, used as a dataset fingerprint.'''
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def split_transactions(frame: pd.DataFrame, config: SplitConfig) -> DatasetSplits:
    '''Split into train, validation, and test partitions, stratified by the target.'''
    holdout_size = config.validation_size + config.test_size
    train, holdout = train_test_split(
        frame,
        test_size=holdout_size,
        stratify=frame[TARGET_COLUMN],
        random_state=config.random_seed,
    )
    validation, test = train_test_split(
        holdout,
        test_size=config.test_size / holdout_size,
        stratify=holdout[TARGET_COLUMN],
        random_state=config.random_seed,
    )
    return DatasetSplits(train=train, validation=validation, test=test)


def generate_synthetic_transactions(
    n_rows: int = 50_000,
    fraud_rate: float = 0.005,
    duplicate_rate: float = 0.004,
    random_seed: int = 42,
) -> pd.DataFrame:
    '''Generate artificial transactions with the same columns as creditcard.csv.

    The data exists so the pipeline, tests, and API can run without the real dataset.
    Metrics obtained on it say nothing about real fraud.

    Args:
        n_rows: Total number of rows, including duplicates.
        fraud_rate: Share of fraudulent rows among the unique rows.
        duplicate_rate: Share of rows that are exact copies of another row.
        random_seed: Seed for the random number generator.

    Returns:
        A frame in the raw schema (Time, V1 to V28, Amount, Class), ordered by time.
    '''
    if n_rows < 100:
        raise ValueError('n_rows must be at least 100')
    if not 0 < fraud_rate < 0.5:
        raise ValueError('fraud_rate must be between 0 and 0.5')
    if not 0 <= duplicate_rate < 0.5:
        raise ValueError('duplicate_rate must be between 0 and 0.5')

    rng = np.random.default_rng(random_seed)
    duplicate_count = round(n_rows * duplicate_rate)
    unique_count = n_rows - duplicate_count
    fraud_count = max(1, round(unique_count * fraud_rate))
    is_fraud = np.zeros(unique_count, dtype=np.int64)
    is_fraud[:fraud_count] = 1

    raw_columns = list(RAW_COLUMN_MAPPING)
    frame = pd.DataFrame(
        _sample_components(rng, unique_count, fraud_count), columns=raw_columns[1:-2]
    )
    frame.insert(0, raw_columns[0], _sample_times(rng, is_fraud))
    frame[raw_columns[-2]] = _sample_amounts(rng, is_fraud)
    frame[raw_columns[-1]] = is_fraud

    duplicates = frame.sample(n=duplicate_count, random_state=random_seed)
    combined = pd.concat([frame, duplicates])
    return combined.sort_values(raw_columns[0], kind='stable').reset_index(drop=True)


def _sample_components(rng: np.random.Generator, row_count: int, fraud_count: int) -> np.ndarray:
    '''Sample PCA-like components. The first fraud_count rows are fraudulent.'''
    components = rng.normal(0.0, _COMPONENT_SCALES, size=(row_count, len(PCA_COLUMNS)))
    # Intensity varies per fraud so that part of it overlaps with legitimate traffic
    intensity = rng.beta(2.0, 1.5, size=fraud_count)
    for position, shift in _FRAUD_COMPONENT_SHIFTS.items():
        components[:fraud_count, position] *= 1.0 + intensity
        components[:fraud_count, position] += shift * intensity
    return components


def _sample_times(rng: np.random.Generator, is_fraud: np.ndarray) -> np.ndarray:
    '''Sample seconds since the first transaction over a two-day window.'''
    hour = rng.choice(24, size=is_fraud.size, p=_HOURLY_ACTIVITY / _HOURLY_ACTIVITY.sum())
    # Fraud is spread evenly over the day, so its share rises during quiet hours
    fraud_rows = np.flatnonzero(is_fraud)
    hour[fraud_rows] = rng.integers(0, 24, size=fraud_rows.size)
    day = rng.integers(0, 2, size=is_fraud.size)
    second = rng.integers(0, SECONDS_PER_HOUR, size=is_fraud.size)
    return (day * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR + second).astype(np.float64)


def _sample_amounts(rng: np.random.Generator, is_fraud: np.ndarray) -> np.ndarray:
    '''Sample transaction amounts with a long right tail.'''
    amount = rng.lognormal(mean=3.1, sigma=1.6, size=is_fraud.size)
    amount[rng.random(is_fraud.size) < _ZERO_AMOUNT_SHARE] = 0.0

    # Fraud mixes small card-testing amounts with larger purchases
    fraud_rows = np.flatnonzero(is_fraud)
    card_testing = rng.random(fraud_rows.size) < _CARD_TESTING_SHARE
    amount[fraud_rows] = np.where(
        card_testing,
        rng.lognormal(mean=0.0, sigma=0.5, size=fraud_rows.size),
        rng.lognormal(mean=4.6, sigma=1.0, size=fraud_rows.size),
    )
    return np.round(np.clip(amount, 0.0, _MAX_AMOUNT), 2)
