'''SHAP and permutation explanations for the fitted classifiers.

The PCA components in the public dataset are anonymised, so explanations can say
which component drove a score but not what that component means in business terms.
'''

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

OutputUnit = Literal['log_odds', 'probability']


@dataclass(frozen=True)
class FeatureContribution:
    '''Contribution of one feature to a single prediction.'''

    feature: str
    # Standardised model input: distance from the training mean in standard deviations
    value: float
    contribution: float


@dataclass(frozen=True)
class Explanation:
    '''Local explanation of one prediction.

    The base value plus all contributions equals the raw model output, expressed
    in output_unit. Only the largest contributions are listed.
    '''

    base_value: float
    output_unit: OutputUnit
    contributions: tuple[FeatureContribution, ...]

    def to_frame(self) -> pd.DataFrame:
        '''Return the contributions as a table.'''
        return pd.DataFrame([asdict(item) for item in self.contributions])


class ShapExplainer:
    '''SHAP values for the fraud class of a fitted classifier.

    Tree models use the exact tree algorithm. Logistic regression uses the linear
    explainer with an all-zero background, which is the training mean because the
    model inputs are standardised.
    '''

    def __init__(self, classifier: Any, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.output_unit: OutputUnit = 'log_odds'
        if isinstance(classifier, LogisticRegression):
            background = np.zeros((1, len(feature_names)))
            self._explainer = shap.LinearExplainer(classifier, background)
        else:
            self._explainer = shap.TreeExplainer(classifier)
            if isinstance(classifier, RandomForestClassifier):
                self.output_unit = 'probability'

    def explain(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''Return SHAP values of shape (rows, features) and base values of shape (rows,).

        Both come from the same call, so for every row the base value plus the sum of
        its SHAP values equals the raw model output. The explainer's expected_value
        attribute is not used: for tree models SHAP replaces it during computation.
        '''
        result = self._explainer(features[self.feature_names])
        values, base_values = np.asarray(result.values), np.asarray(result.base_values)
        # Classifiers with one output per class keep the fraud class only
        if values.ndim == 3:
            values, base_values = values[:, :, 1], base_values[:, 1]
        return values, base_values

    def shap_values(self, features: pd.DataFrame) -> np.ndarray:
        '''Return SHAP values of shape (rows, features) for the fraud class.'''
        return self.explain(features)[0]

    def explain_row(self, features: pd.DataFrame, top_k: int) -> Explanation:
        '''Explain the first row of features with its top_k largest contributions.'''
        row = features[self.feature_names].iloc[[0]]
        values, base_values = self.explain(row)
        contributions = values[0]
        largest = np.argsort(-np.abs(contributions))[:top_k]
        return Explanation(
            base_value=float(base_values[0]),
            output_unit=self.output_unit,
            contributions=tuple(
                FeatureContribution(
                    feature=self.feature_names[position],
                    value=float(row.iloc[0, position]),
                    contribution=float(contributions[position]),
                )
                for position in largest
            ),
        )


def sample_for_explanation(
    features: pd.DataFrame, target: pd.Series, sample_size: int, random_seed: int
) -> pd.DataFrame:
    '''Return every fraud row plus random legitimate rows, up to sample_size.

    A plain random sample would hold only a handful of fraud cases, which hides
    the patterns the model uses to detect them. Importance computed on this
    sample is therefore not weighted by how often each class occurs.
    '''
    fraud = features[target == 1]
    legitimate = features[target == 0]
    legitimate_count = min(len(legitimate), max(sample_size - len(fraud), 0))
    sampled = legitimate.sample(n=legitimate_count, random_state=random_seed)
    return pd.concat([fraud, sampled])


def mean_absolute_shap(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    '''Rank features by their mean absolute SHAP value.'''
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    return importance.sort_values(ascending=False)


def compute_permutation_importance(
    classifier: Any, features: pd.DataFrame, target: pd.Series, repeats: int, random_seed: int
) -> pd.Series:
    '''Rank features by the drop in average precision when each one is shuffled.

    This is model-agnostic and measured on held-out data, so it serves as a
    cross-check on the SHAP ranking.
    '''
    result = permutation_importance(
        classifier,
        features,
        target,
        scoring='average_precision',
        n_repeats=repeats,
        random_state=random_seed,
    )
    importance = pd.Series(result.importances_mean, index=features.columns)
    return importance.sort_values(ascending=False)
