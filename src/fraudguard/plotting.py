'''Diagnostic figures for model evaluation and explainability.

Figures are built with the object-oriented Matplotlib API, so they need neither a
display backend nor global pyplot state. One palette is used throughout: blue for
the model under discussion, gray for context, and a blue-to-red scale only where
direction matters.
'''

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.typing import RcKeyType
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from fraudguard.config import ThresholdConfig
from fraudguard.evaluation import ThresholdMetrics, ThresholdSelection

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
NEUTRAL = '#c3c2b7'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'
RED = '#e34948'

_SEQUENTIAL_BLUES = LinearSegmentedColormap.from_list(
    'sequential_blues', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
)
_LOW_TO_HIGH = LinearSegmentedColormap.from_list('low_to_high', [BLUE, NEUTRAL, RED])

_STYLE: dict[RcKeyType, Any] = {
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'text.color': INK,
    'axes.titlesize': 11,
    'axes.titlelocation': 'left',
    'axes.titlepad': 10,
    'axes.labelcolor': INK_SECONDARY,
    'axes.edgecolor': NEUTRAL,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'lines.linewidth': 2.0,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
    'legend.frameon': False,
}
_REFERENCE_LINE: dict[str, Any] = {'color': NEUTRAL, 'linewidth': 1.0, 'linestyle': '--'}
_POINT_MARKER: dict[str, Any] = {
    'marker': 'o',
    'markersize': 9,
    'markeredgecolor': SURFACE,
    'markeredgewidth': 2,
}


@contextmanager
def _styled_figure(path: Path, width: float, height: float) -> Iterator[Figure]:
    '''Yield a figure with the shared style and save it to path on exit.'''
    with matplotlib.rc_context(_STYLE):
        figure = Figure(figsize=(width, height), layout='constrained')
        yield figure
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)


def plot_roc_curve(
    target: np.ndarray,
    scores: np.ndarray,
    operating_point: ThresholdMetrics,
    title: str,
    path: Path,
) -> Path:
    '''Plot the ROC curve and mark the chosen operating point.'''
    false_positive_rate, recall, _ = roc_curve(target, scores)
    with _styled_figure(path, 5.6, 4.6) as figure:
        axes = figure.subplots()
        axes.plot([0, 1], [0, 1], **_REFERENCE_LINE)
        axes.text(0.62, 0.56, 'chance', color=INK_MUTED, rotation=38, rotation_mode='anchor')
        axes.plot(false_positive_rate, recall, color=BLUE)
        _mark_operating_point(
            axes, operating_point.false_positive_rate, operating_point.recall, operating_point
        )
        axes.set(
            title=title,
            xlabel='False-positive rate',
            ylabel='Recall',
            xlim=(-0.01, 1),
            ylim=(0, 1.02),
        )
    return path


def plot_precision_recall_curve(
    target: np.ndarray,
    scores: np.ndarray,
    operating_point: ThresholdMetrics,
    title: str,
    path: Path,
) -> Path:
    '''Plot the precision-recall curve against the fraud-rate baseline.'''
    precision, recall, _ = precision_recall_curve(target, scores)
    fraud_rate = float(np.mean(target))
    with _styled_figure(path, 5.6, 4.6) as figure:
        axes = figure.subplots()
        axes.axhline(fraud_rate, **_REFERENCE_LINE)
        axes.text(0.99, fraud_rate + 0.015, 'fraud rate', color=INK_MUTED, ha='right')
        axes.step(recall, precision, where='post', color=BLUE)
        _mark_operating_point(
            axes, operating_point.recall, operating_point.precision, operating_point
        )
        axes.set(title=title, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    return path


def plot_precision_recall_comparison(
    target: np.ndarray, scores_by_model: Mapping[str, np.ndarray], path: Path
) -> Path:
    '''Plot one panel per model, each highlighting its curve against the others.'''
    curves = {
        name: precision_recall_curve(target, scores)[:2] for name, scores in scores_by_model.items()
    }
    column_count = min(2, len(curves))
    row_count = math.ceil(len(curves) / column_count)
    with _styled_figure(path, 3.8 * column_count, 3.1 * row_count) as figure:
        grid = figure.subplots(row_count, column_count, sharex=True, sharey=True, squeeze=False)
        panels: list[Axes] = list(grid.flat)
        for axes, name in zip(panels, curves, strict=False):
            for other_name, (precision, recall) in curves.items():
                if other_name != name:
                    axes.step(recall, precision, where='post', color=NEUTRAL, linewidth=1)
            precision, recall = curves[name]
            axes.step(recall, precision, where='post', color=BLUE)
            pr_auc = average_precision_score(target, scores_by_model[name])
            axes.set(title=f'{name}\nPR-AUC {pr_auc:.3f}', xlim=(0, 1), ylim=(0, 1.02))
        for unused in panels[len(curves) :]:
            unused.set_visible(False)
        figure.supxlabel('Recall', color=INK_SECONDARY, fontsize=10)
        figure.supylabel('Precision', color=INK_SECONDARY, fontsize=10)
    return path


def plot_confusion_matrix(metrics: ThresholdMetrics, title: str, path: Path) -> Path:
    '''Plot the confusion matrix, shaded by the share of each actual class.'''
    counts = np.array(
        [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ]
    )
    shares = counts / counts.sum(axis=1, keepdims=True)
    labels = ['legitimate', 'fraud']
    with _styled_figure(path, 4.6, 4.0) as figure:
        axes = figure.subplots()
        axes.imshow(shares, cmap=_SEQUENTIAL_BLUES, vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                colour = '#ffffff' if shares[row, column] > 0.5 else INK
                text = f'{counts[row, column]:,}\n{shares[row, column]:.2%}'
                axes.text(column, row, text, ha='center', va='center', color=colour)
        axes.set(title=title, xlabel='Predicted', ylabel='Actual')
        axes.set_xticks([0, 1], labels=labels)
        axes.set_yticks([0, 1], labels=labels)
        # A gap in the surface colour separates the cells instead of a drawn border
        axes.set_xticks([0.5], minor=True)
        axes.set_yticks([0.5], minor=True)
        axes.grid(visible=False, which='major')
        axes.grid(visible=True, which='minor', color=SURFACE, linewidth=3)
        axes.tick_params(which='both', length=0)
        axes.spines[:].set_visible(False)
    return path


def plot_calibration_curve(
    calibration: pd.DataFrame, brier_score: float, title: str, path: Path
) -> Path:
    '''Plot observed fraud rate against predicted probability, with bin sizes below.'''
    with _styled_figure(path, 5.6, 5.4) as figure:
        top, bottom = figure.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        top.plot([0, 1], [0, 1], **_REFERENCE_LINE)
        top.text(
            0.56, 0.5, 'perfect calibration', color=INK_MUTED, rotation=37, rotation_mode='anchor'
        )
        top.plot(
            calibration['mean_predicted'], calibration['observed_rate'], color=BLUE, **_POINT_MARKER
        )
        top.set(
            title=f'{title}\nBrier score {brier_score:.5f}',
            ylabel='Observed fraud rate',
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        # Dots rather than bars: a log axis has no meaningful zero baseline
        bottom.plot(
            calibration['mean_predicted'],
            calibration['count'],
            color=BLUE,
            linestyle='none',
            **_POINT_MARKER,
        )
        bottom.set(xlabel='Predicted fraud probability', ylabel='Rows in bin', yscale='log')
    return path


def plot_threshold_analysis(
    table: pd.DataFrame,
    selection: ThresholdSelection,
    config: ThresholdConfig,
    title: str,
    path: Path,
) -> Path:
    '''Plot detection quality, false-positive rate, and cost across thresholds.

    The three measures have different scales, so each gets its own panel on a
    shared threshold axis instead of a second y-axis.
    '''
    threshold = table['threshold']
    budget = config.max_false_positive_rate
    with _styled_figure(path, 6.8, 7.4) as figure:
        quality, false_alerts, cost = figure.subplots(3, 1, sharex=True, height_ratios=[3, 2, 2])

        for column, colour, label in (
            ('precision', BLUE, 'precision'),
            ('recall', ORANGE, 'recall'),
            ('f1', AQUA, 'F1'),
        ):
            quality.plot(threshold, table[column], color=colour, label=label)
        quality.set(title=title, ylabel='Score', ylim=(0, 1.02))

        false_alerts.plot(threshold, table['false_positive_rate'], color=BLUE)
        false_alerts.axhline(budget, **_REFERENCE_LINE)
        false_alerts.text(0.995, budget * 1.12, 'review budget', color=INK_MUTED, ha='right')
        # Zoom on the region around the budget; lower thresholds run off the top
        false_alerts.set(ylabel='False-positive rate', ylim=(0, budget * 4))
        false_alerts.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

        cost.plot(threshold, table['cost_per_10k'], color=BLUE)
        cost.set(xlabel='Decision threshold', ylabel='Cost per 10,000', xlim=(0, 1), ylim=(0, None))

        markers = (
            (selection.threshold, 'selected threshold', {'color': INK, 'linestyle': '--'}),
            (
                selection.max_f1_threshold,
                'highest F1 threshold',
                {'color': INK_MUTED, 'linestyle': ':'},
            ),
        )
        for value, label, style in markers:
            quality.axvline(value, linewidth=1, label=label, **style)
            for axes in (false_alerts, cost):
                axes.axvline(value, linewidth=1, **style)
        # One legend below the panels keeps the title and the curves clear
        figure.legend(*quality.get_legend_handles_labels(), loc='outside lower center', ncols=3)
    return path


def plot_feature_importance(
    importance: pd.Series, title: str, value_label: str, path: Path
) -> Path:
    '''Plot features as ranked horizontal bars, largest at the top.'''
    ordered = importance.sort_values()
    with _styled_figure(path, 6.0, 0.42 * len(ordered) + 1.2) as figure:
        axes = figure.subplots()
        axes.barh(ordered.index, ordered.to_numpy(), height=0.5, color=BLUE)
        for position, value in enumerate(ordered):
            axes.annotate(
                f'{value:.3f}',
                (value, position),
                xytext=(4, 0),
                textcoords='offset points',
                va='center',
                color=INK_SECONDARY,
                fontsize=9,
            )
        axes.set(title=title, xlabel=value_label, xlim=(0, float(ordered.max()) * 1.12))
        axes.grid(visible=False, axis='y')
        axes.tick_params(axis='y', length=0, labelcolor=INK_SECONDARY)
    return path


def plot_shap_summary(
    shap_values: np.ndarray, features: pd.DataFrame, top_features: Sequence[str], path: Path
) -> Path:
    '''Plot the SHAP value of every sampled row for the top features.

    Each dot is one transaction. Its colour is the rank of the feature value within
    the sample, so a few extreme values cannot wash out the colour scale.
    '''
    rng = np.random.default_rng(0)
    ordered = list(reversed(top_features))
    with _styled_figure(path, 6.6, 0.42 * len(ordered) + 1.4) as figure:
        axes = figure.subplots()
        axes.axvline(0, color=NEUTRAL, linewidth=1)
        for position, name in enumerate(ordered):
            column = features.columns.get_loc(name)
            jitter = rng.uniform(-0.28, 0.28, size=len(features))
            dots = axes.scatter(
                shap_values[:, column],
                position + jitter,
                c=features[name].rank(pct=True),
                cmap=_LOW_TO_HIGH,
                vmin=0,
                vmax=1,
                s=7,
                linewidths=0,
            )
        axes.set_yticks(range(len(ordered)), labels=ordered)
        axes.set(
            title='SHAP values of the most important features',
            xlabel='SHAP value (positive pushes towards fraud)',
        )
        axes.grid(visible=False, axis='y')
        axes.tick_params(axis='y', length=0, labelcolor=INK_SECONDARY)
        colour_bar = figure.colorbar(dots, ax=axes, ticks=[0, 1], aspect=30, pad=0.02)
        colour_bar.ax.set_yticklabels(['low', 'high'])
        colour_bar.set_label('Feature value', color=INK_SECONDARY)
        colour_bar.outline.set_visible(False)
    return path


def plot_local_explanation(contributions: pd.DataFrame, title: str, path: Path) -> Path:
    '''Plot the largest feature contributions for a single prediction.

    Args:
        contributions: Columns feature and contribution, ordered by absolute size.
        title: Figure title.
        path: Output file.
    '''
    ordered = contributions.iloc[::-1]
    values = ordered['contribution'].to_numpy()
    with _styled_figure(path, 6.0, 0.42 * len(ordered) + 1.5) as figure:
        axes = figure.subplots()
        axes.axvline(0, color=NEUTRAL, linewidth=1)
        for colour, mask, label in (
            (RED, values > 0, 'raises the fraud score'),
            (BLUE, values <= 0, 'lowers the fraud score'),
        ):
            axes.barh(np.flatnonzero(mask), values[mask], height=0.5, color=colour, label=label)
        axes.set_yticks(range(len(ordered)), labels=ordered['feature'])
        axes.set(title=title, xlabel='SHAP value')
        axes.grid(visible=False, axis='y')
        axes.tick_params(axis='y', length=0, labelcolor=INK_SECONDARY)
        figure.legend(loc='outside lower center', ncols=2)
    return path


def _mark_operating_point(
    axes: Axes, x: float, y: float, operating_point: ThresholdMetrics
) -> None:
    '''Mark the chosen threshold on a curve with a ringed dot and a label.'''
    axes.plot([x], [y], color=BLUE, linestyle='none', **_POINT_MARKER)
    # Place the label on whichever side of the point has room
    offset, alignment = ((10, -14), 'left') if x < 0.6 else ((-10, -14), 'right')
    axes.annotate(
        f'threshold {operating_point.threshold:.3f}',
        (x, y),
        xytext=offset,
        textcoords='offset points',
        ha=alignment,
        color=INK_SECONDARY,
    )
