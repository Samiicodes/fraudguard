'''Tests for data quality checks, cleaning, and the split leakage guard.'''

import numpy as np
import pandas as pd
import pytest

from fraudguard.config import SplitConfig, ValidationConfig
from fraudguard.data_ingestion import (
    INPUT_COLUMNS,
    TARGET_COLUMN,
    DatasetSplits,
    split_transactions,
)
from fraudguard.data_validation import (
    DataValidationError,
    assert_no_split_overlap,
    check_class_balance,
    check_duplicates,
    check_missing_values,
    check_schema,
    check_value_ranges,
    clean_transactions,
    count_split_overlap,
    find_invalid_rows,
    validate_transactions,
)

VALIDATION = ValidationConfig(
    max_missing_fraction=0.05,
    max_invalid_fraction=0.01,
    min_fraud_rows=30,
    min_fraud_rate=0.0005,
    max_fraud_rate=0.2,
)
SPLIT = SplitConfig(validation_size=0.2, test_size=0.2, random_seed=0)


def test_clean_dataset_passes_every_check(transactions: pd.DataFrame) -> None:
    report = validate_transactions(transactions, VALIDATION)

    assert report.passed
    assert [check.status for check in report.checks] == ['passed'] * 5


def test_missing_column_fails_schema(transactions: pd.DataFrame) -> None:
    report = validate_transactions(transactions.drop(columns=['v14']), VALIDATION)

    assert not report.passed
    assert 'v14' in report.failures[0].message


def test_non_numeric_column_fails_schema(transactions: pd.DataFrame) -> None:
    transactions['amount'] = transactions['amount'].astype(str)

    assert check_schema(transactions).status == 'failed'


def test_unexpected_column_is_only_a_warning(transactions: pd.DataFrame) -> None:
    transactions['merchant_note'] = 'gift card'

    result = check_schema(transactions)

    assert result.status == 'warning'
    assert 'merchant_note' in result.message


def test_missing_values_warn_below_the_limit_and_fail_above(transactions: pd.DataFrame) -> None:
    few = transactions.copy()
    few.loc[few.index[:5], 'v3'] = np.nan
    many = transactions.copy()
    many.loc[many.index[: len(many) // 10], 'v3'] = np.nan

    assert check_missing_values(few, max_missing_fraction=0.05).status == 'warning'
    assert check_missing_values(many, max_missing_fraction=0.05).status == 'failed'


def test_duplicates_are_counted(transactions: pd.DataFrame) -> None:
    result = check_duplicates(pd.concat([transactions, transactions.head(3)]))

    assert result.status == 'warning'
    assert result.message.startswith('3 duplicate rows')


def test_invalid_values_are_flagged(transactions: pd.DataFrame) -> None:
    transactions.loc[transactions.index[0], 'amount'] = -5.0
    transactions.loc[transactions.index[1], TARGET_COLUMN] = 2
    transactions.loc[transactions.index[2], 'v7'] = np.inf

    assert find_invalid_rows(transactions).sum() == 3
    assert check_value_ranges(transactions, max_invalid_fraction=0.01).status == 'warning'
    assert check_value_ranges(transactions, max_invalid_fraction=0.0001).status == 'failed'


def test_too_few_fraud_rows_fail_class_balance(transactions: pd.DataFrame) -> None:
    fraud_rows = transactions.index[transactions[TARGET_COLUMN] == 1]
    transactions.loc[fraud_rows[5:], TARGET_COLUMN] = 0

    assert check_class_balance(transactions, VALIDATION).status == 'failed'


def test_implausible_fraud_rate_is_a_warning(transactions: pd.DataFrame) -> None:
    strict = VALIDATION.model_copy(update={'max_fraud_rate': 0.01})

    assert check_class_balance(transactions, strict).status == 'warning'


def test_empty_dataset_fails(transactions: pd.DataFrame) -> None:
    report = validate_transactions(transactions.iloc[0:0], VALIDATION)

    assert not report.passed
    assert report.failures[0].name == 'row_count'


def test_raise_if_failed_names_the_failed_check(transactions: pd.DataFrame) -> None:
    report = validate_transactions(transactions.drop(columns=['amount']), VALIDATION)

    with pytest.raises(DataValidationError, match='schema'):
        report.raise_if_failed()


def test_cleaning_drops_invalid_duplicate_and_conflicting_rows(
    transactions: pd.DataFrame,
) -> None:
    transactions.loc[transactions.index[0], 'amount'] = -1.0
    duplicate = transactions.iloc[[1]]
    conflicting = transactions.iloc[[2]].copy()
    conflicting[TARGET_COLUMN] = 1 - conflicting[TARGET_COLUMN]
    frame = pd.concat([transactions, duplicate, conflicting], ignore_index=True)

    cleaned, summary = clean_transactions(frame)

    assert summary.invalid_rows_dropped == 1
    assert summary.duplicate_rows_dropped == 1
    # Both copies of the ambiguous transaction go
    assert summary.conflicting_rows_dropped == 2
    assert summary.output_rows == len(cleaned) == len(transactions) - 2
    assert not cleaned.duplicated(subset=INPUT_COLUMNS).any()


def test_missing_values_are_kept_for_imputation(transactions: pd.DataFrame) -> None:
    transactions.loc[transactions.index[:3], 'v5'] = np.nan

    cleaned, summary = clean_transactions(transactions)

    assert summary.output_rows == len(transactions)
    assert cleaned['v5'].isna().sum() == 3


def test_split_is_stratified_and_disjoint(transactions: pd.DataFrame) -> None:
    splits = split_transactions(transactions, SPLIT)
    parts = [splits.train, splits.validation, splits.test]

    assert sum(len(part) for part in parts) == len(transactions)
    assert len(splits.train) == pytest.approx(0.6 * len(transactions), abs=1)
    fraud_rate = transactions[TARGET_COLUMN].mean()
    for part in parts:
        assert part[TARGET_COLUMN].mean() == pytest.approx(fraud_rate, abs=0.005)
    train, validation, test = (set(part.index) for part in parts)
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_split_is_reproducible(transactions: pd.DataFrame) -> None:
    first = split_transactions(transactions, SPLIT)
    second = split_transactions(transactions, SPLIT)

    assert first.test.index.equals(second.test.index)


def test_overlap_guard_catches_a_row_shared_between_splits(transactions: pd.DataFrame) -> None:
    splits = split_transactions(transactions, SPLIT)
    leaked = DatasetSplits(
        train=splits.train,
        validation=splits.validation,
        test=pd.concat([splits.test, splits.train.head(1)]),
    )

    assert count_split_overlap(splits) == 0
    assert count_split_overlap(leaked) == 1
    with pytest.raises(DataValidationError, match='more than one split'):
        assert_no_split_overlap(leaked)
