'''Data quality checks, cleaning, and leakage guards for transaction data.'''

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraudguard.config import ValidationConfig
from fraudguard.data_ingestion import (
    AMOUNT_COLUMN,
    INPUT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    DatasetSplits,
)

CheckStatus = Literal['passed', 'warning', 'failed']


class DataValidationError(ValueError):
    '''Raised when the dataset fails a blocking validation check.'''


@dataclass(frozen=True)
class CheckResult:
    '''Outcome of one validation check.'''

    name: str
    status: CheckStatus
    message: str


@dataclass(frozen=True)
class ValidationReport:
    '''Results of all validation checks for one dataset.'''

    row_count: int
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        '''Whether no blocking check failed. Warnings do not block the pipeline.'''
        return not self.failures

    @property
    def failures(self) -> list[CheckResult]:
        '''Checks that block the pipeline.'''
        return [check for check in self.checks if check.status == 'failed']

    def to_dict(self) -> dict[str, Any]:
        '''Return a JSON-serialisable summary.'''
        return {
            'row_count': self.row_count,
            'passed': self.passed,
            'checks': [asdict(check) for check in self.checks],
        }

    def raise_if_failed(self) -> None:
        '''Raise DataValidationError listing every failed check.'''
        if self.failures:
            details = '; '.join(f'{check.name}: {check.message}' for check in self.failures)
            raise DataValidationError(f'Data validation failed. {details}')


@dataclass(frozen=True)
class CleaningSummary:
    '''Row counts removed by each cleaning step.'''

    input_rows: int
    invalid_rows_dropped: int
    duplicate_rows_dropped: int
    conflicting_rows_dropped: int
    output_rows: int


def check_schema(frame: pd.DataFrame) -> CheckResult:
    '''Check that every required column is present and numeric.'''
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return CheckResult('schema', 'failed', 'Missing columns: ' + ', '.join(missing))

    non_numeric = [
        column for column in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        return CheckResult('schema', 'failed', 'Non-numeric columns: ' + ', '.join(non_numeric))

    unexpected = [str(column) for column in frame.columns if column not in REQUIRED_COLUMNS]
    if unexpected:
        return CheckResult(
            'schema', 'warning', 'Unexpected columns ignored: ' + ', '.join(unexpected)
        )
    return CheckResult('schema', 'passed', f'All {len(REQUIRED_COLUMNS)} required columns present')


def check_missing_values(frame: pd.DataFrame, max_missing_fraction: float) -> CheckResult:
    '''Check the share of missing values in each input column.'''
    missing_fraction = frame[INPUT_COLUMNS].isna().mean()
    worst_column = str(missing_fraction.idxmax())
    worst_fraction = float(missing_fraction.max())
    if worst_fraction > max_missing_fraction:
        message = (
            f'{worst_column} is missing in {worst_fraction:.1%} of rows, '
            f'above the {max_missing_fraction:.1%} limit'
        )
        return CheckResult('missing_values', 'failed', message)

    affected_rows = int(frame[INPUT_COLUMNS].isna().any(axis=1).sum())
    if affected_rows:
        message = f'{affected_rows} rows have missing inputs and will be imputed with train medians'
        return CheckResult('missing_values', 'warning', message)
    return CheckResult('missing_values', 'passed', 'No missing input values')


def check_duplicates(frame: pd.DataFrame) -> CheckResult:
    '''Count exact duplicate rows, which would leak between splits if kept.'''
    duplicate_count = int(frame.duplicated(subset=REQUIRED_COLUMNS).sum())
    if duplicate_count:
        share = duplicate_count / len(frame)
        message = f'{duplicate_count} duplicate rows ({share:.2%}) will be dropped before splitting'
        return CheckResult('duplicates', 'warning', message)
    return CheckResult('duplicates', 'passed', 'No duplicate rows')


def find_invalid_rows(frame: pd.DataFrame) -> pd.Series:
    '''Flag rows with an unusable target, a negative amount or time, or infinite inputs.'''
    invalid_target = ~frame[TARGET_COLUMN].isin([0, 1])
    negative = (frame[AMOUNT_COLUMN] < 0) | (frame[TIME_COLUMN] < 0)
    infinite = frame[INPUT_COLUMNS].isin([np.inf, -np.inf]).any(axis=1)
    return invalid_target | negative | infinite


def check_value_ranges(frame: pd.DataFrame, max_invalid_fraction: float) -> CheckResult:
    '''Check the share of rows with values outside their valid range.'''
    invalid_count = int(find_invalid_rows(frame).sum())
    if not invalid_count:
        return CheckResult('value_ranges', 'passed', 'All values within valid ranges')

    share = invalid_count / len(frame)
    if share > max_invalid_fraction:
        limit = f'{max_invalid_fraction:.2%}'
        message = f'{invalid_count} invalid rows ({share:.2%}), above the {limit} limit'
        return CheckResult('value_ranges', 'failed', message)
    return CheckResult('value_ranges', 'warning', f'{invalid_count} invalid rows will be dropped')


def check_class_balance(frame: pd.DataFrame, config: ValidationConfig) -> CheckResult:
    '''Check that there is enough fraud to stratify and that its rate is plausible.'''
    fraud_count = int((frame[TARGET_COLUMN] == 1).sum())
    fraud_rate = fraud_count / len(frame)
    if fraud_count < config.min_fraud_rows:
        message = f'Only {fraud_count} fraud rows, at least {config.min_fraud_rows} are required'
        return CheckResult('class_balance', 'failed', message)

    summary = f'{fraud_count} fraud rows ({fraud_rate:.3%})'
    if not config.min_fraud_rate <= fraud_rate <= config.max_fraud_rate:
        expected = f'{config.min_fraud_rate:.2%} to {config.max_fraud_rate:.0%}'
        return CheckResult(
            'class_balance', 'warning', f'{summary}, outside the expected {expected}'
        )
    return CheckResult('class_balance', 'passed', summary)


def validate_transactions(frame: pd.DataFrame, config: ValidationConfig) -> ValidationReport:
    '''Run every validation check on a frame in the canonical schema.'''
    schema = check_schema(frame)
    if schema.status == 'failed':
        return ValidationReport(row_count=len(frame), checks=(schema,))
    if frame.empty:
        empty = CheckResult('row_count', 'failed', 'Dataset has no rows')
        return ValidationReport(row_count=0, checks=(schema, empty))

    checks = (
        schema,
        check_missing_values(frame, config.max_missing_fraction),
        check_duplicates(frame),
        check_value_ranges(frame, config.max_invalid_fraction),
        check_class_balance(frame, config),
    )
    return ValidationReport(row_count=len(frame), checks=checks)


def clean_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    '''Drop rows that are unusable or that would leak between splits.

    Missing input values are kept. They are imputed later with statistics
    fitted on the training split.
    '''
    valid = frame.loc[~find_invalid_rows(frame), REQUIRED_COLUMNS]
    deduplicated = valid.drop_duplicates()
    # Identical inputs with different labels are ambiguous and could straddle splits
    conflicting = deduplicated.duplicated(subset=INPUT_COLUMNS, keep=False)
    cleaned = deduplicated.loc[~conflicting].astype({TARGET_COLUMN: 'int64'})

    summary = CleaningSummary(
        input_rows=len(frame),
        invalid_rows_dropped=len(frame) - len(valid),
        duplicate_rows_dropped=len(valid) - len(deduplicated),
        conflicting_rows_dropped=len(deduplicated) - len(cleaned),
        output_rows=len(cleaned),
    )
    return cleaned, summary


def count_split_overlap(splits: DatasetSplits) -> int:
    '''Count input vectors that appear in more than one split.'''
    train, validation, test = (
        set(pd.util.hash_pandas_object(part[INPUT_COLUMNS], index=False))
        for part in (splits.train, splits.validation, splits.test)
    )
    return len((train & validation) | (train & test) | (validation & test))


def assert_no_split_overlap(splits: DatasetSplits) -> None:
    '''Raise DataValidationError if any input vector is shared between splits.'''
    overlap = count_split_overlap(splits)
    if overlap:
        raise DataValidationError(f'{overlap} input vectors appear in more than one split')
