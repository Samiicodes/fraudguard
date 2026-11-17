'''Training pipeline: prepare data, fit candidates, track runs, and register the champion.'''

import json
import logging
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ResamplingPipeline
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraudguard import plotting
from fraudguard.config import AppConfig, CandidateConfig, MlflowConfig, TrainingConfig
from fraudguard.data_ingestion import (
    INPUT_COLUMNS,
    TARGET_COLUMN,
    DatasetSplits,
    compute_file_sha256,
    load_raw_transactions,
    split_transactions,
)
from fraudguard.data_validation import (
    CleaningSummary,
    ValidationReport,
    assert_no_split_overlap,
    clean_transactions,
    validate_transactions,
)
from fraudguard.evaluation import (
    ThresholdMetrics,
    ThresholdSelection,
    bootstrap_confidence_intervals,
    build_threshold_table,
    compute_calibration_table,
    compute_ranking_metrics,
    compute_threshold_metrics,
    cost_without_model,
    select_threshold,
)
from fraudguard.explainability import (
    ShapExplainer,
    compute_permutation_importance,
    mean_absolute_shap,
    sample_for_explanation,
)
from fraudguard.feature_engineering import (
    MODEL_FEATURES,
    ModelData,
    ProcessedSplits,
    build_preprocessor,
    fit_preprocessor,
    save_processed_data,
)
from fraudguard.inference import CLASSIFIER_STEP, build_serving_pipeline
from fraudguard.monitoring import build_reference_profile
from fraudguard.reporting import (
    FIGURE_DIR_NAME,
    FIGURE_FILES,
    CandidateSummary,
    ExplainedExample,
    ReportData,
    SplitSummary,
    format_score,
    write_evaluation_report,
)

logger = logging.getLogger(__name__)

SYNTHETIC_LABEL = 'synthetic sample'

# Models are saved with skops, which only rebuilds the types listed here instead of
# unpickling arbitrary objects. MLflow stores the list next to the model.
SKOPS_TRUSTED_TYPES = [
    'fraudguard.feature_engineering.TransactionFeatureBuilder',
    'numpy.dtype',
    'sklearn.tree._tree.Tree',
    'xgboost.core.Booster',
    'xgboost.sklearn.XGBClassifier',
]


@dataclass(frozen=True)
class PreparedData:
    '''Validated, cleaned, split, and preprocessed data.'''

    dataset_label: str
    dataset_sha256: str
    is_synthetic: bool
    validation_report: ValidationReport
    cleaning: CleaningSummary
    splits: DatasetSplits
    processed: ProcessedSplits
    preprocessor: Pipeline


@dataclass(frozen=True)
class CandidateResult:
    '''A fitted candidate with its validation results and MLflow identifiers.'''

    config: CandidateConfig
    classifier: Any
    fit_seconds: float
    validation_scores: np.ndarray
    ranking: dict[str, float]
    threshold_table: pd.DataFrame
    selection: ThresholdSelection
    at_threshold: ThresholdMetrics
    run_id: str
    model_uri: str
    model_id: str


@dataclass(frozen=True)
class TestEvaluation:
    '''Champion results on the test split.'''

    scores: np.ndarray
    ranking: dict[str, float]
    at_threshold: ThresholdMetrics
    at_max_f1: ThresholdMetrics
    at_default: ThresholdMetrics
    intervals: dict[str, tuple[float, float]]
    no_model_cost: float


@dataclass(frozen=True)
class ChampionExplanation:
    '''Global and local explanations of the champion on validation data.'''

    sample_size: int
    shap_importance: pd.Series
    permutation_importance: pd.Series
    detected_example: ExplainedExample
    missed_example: ExplainedExample


@dataclass(frozen=True)
class TrainingSummary:
    '''Headline results of a training run.'''

    champion_name: str
    model_version: str
    decision_threshold: float
    validation_pr_auc: float
    test_pr_auc: float
    test_precision: float
    test_recall: float
    report_path: Path


def run_training_pipeline(config: AppConfig, use_sample_data: bool = False) -> TrainingSummary:
    '''Run the full pipeline from raw data to a registered champion and a report.

    Args:
        config: Project configuration.
        use_sample_data: Train on the synthetic sample instead of the real dataset.

    Returns:
        Headline results of the run.
    '''
    prepared = prepare_data(config, use_sample_data)
    experiment_id = configure_mlflow(config.mlflow)
    reference_profile = build_reference_profile(prepared.splits.train, config.monitoring.psi_bins)

    results = [
        train_candidate(candidate, prepared, config, experiment_id, reference_profile)
        for candidate in config.training.candidates
    ]

    logger.info('Selecting champion')
    champion = select_champion(results, config.training.selection_metric)
    logger.info('Champion is %s', champion.config.name)

    logger.info('Evaluating champion on the test split')
    test_evaluation = evaluate_on_test(champion, prepared.processed.test, config)

    logger.info('Explaining champion predictions')
    figure_dir = config.reports.output_dir / FIGURE_DIR_NAME
    explanation = explain_champion(champion, prepared.processed.validation, config, figure_dir)

    logger.info('Registering champion')
    model_version = register_champion(champion, config)

    logger.info('Saving artefacts')
    _plot_champion_figures(champion, results, test_evaluation, prepared, config, figure_dir)
    report_data = _build_report_data(
        prepared, results, champion, test_evaluation, explanation, model_version, config
    )
    report_path = write_evaluation_report(report_data, config.reports.output_dir)
    _write_result_files(report_data, champion, config.reports.output_dir)
    _log_champion_artifacts(champion, test_evaluation, config.reports.output_dir)

    return TrainingSummary(
        champion_name=champion.config.name,
        model_version=model_version,
        decision_threshold=champion.selection.threshold,
        validation_pr_auc=champion.ranking['pr_auc'],
        test_pr_auc=test_evaluation.ranking['pr_auc'],
        test_precision=test_evaluation.at_threshold.precision,
        test_recall=test_evaluation.at_threshold.recall,
        report_path=report_path,
    )


def prepare_data(config: AppConfig, use_sample_data: bool) -> PreparedData:
    '''Load, validate, clean, split, and preprocess the dataset.

    Raises:
        DataValidationError: If a blocking validation check fails or splits overlap.
    '''
    data_path = config.data.sample_path if use_sample_data else config.data.raw_path
    logger.info('Loading dataset')
    raw = load_raw_transactions(data_path)
    logger.info('Loaded %d rows from %s', len(raw), data_path.as_posix())

    logger.info('Validating schema')
    validation_report = validate_transactions(raw, config.validation)
    for check in validation_report.checks:
        logger.info('Check %s %s: %s', check.name, check.status, check.message)
    validation_report.raise_if_failed()

    logger.info('Cleaning and splitting data')
    cleaned, cleaning = clean_transactions(raw)
    splits = split_transactions(cleaned, config.split)
    # Guard against leakage: no input vector may appear in more than one split
    assert_no_split_overlap(splits)
    logger.info(
        'Split %d rows into train %d, validation %d, test %d',
        len(cleaned),
        len(splits.train),
        len(splits.validation),
        len(splits.test),
    )

    logger.info('Fitting preprocessor on the training split')
    preprocessor = build_preprocessor(config.features)
    processed = fit_preprocessor(preprocessor, splits)
    save_processed_data(processed, preprocessor, config.data.processed_dir)

    return PreparedData(
        dataset_label=SYNTHETIC_LABEL if use_sample_data else data_path.name,
        dataset_sha256=compute_file_sha256(data_path),
        is_synthetic=use_sample_data,
        validation_report=validation_report,
        cleaning=cleaning,
        splits=splits,
        processed=processed,
        preprocessor=preprocessor,
    )


def configure_mlflow(config: MlflowConfig) -> str:
    '''Point MLflow at the tracking store and return the experiment id.'''
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        return str(experiment.experiment_id)
    if config.tracking_uri.startswith(('http://', 'https://')):
        # A tracking server decides where its artefacts live
        return str(mlflow.create_experiment(config.experiment_name))
    artifact_location = config.artifact_dir.resolve().as_uri()
    return str(
        mlflow.create_experiment(config.experiment_name, artifact_location=artifact_location)
    )


def build_classifier(
    candidate: CandidateConfig, training: TrainingConfig, scale_pos_weight: float
) -> Any:
    '''Build an unfitted estimator for a candidate, including its imbalance handling.

    Args:
        candidate: Algorithm, imbalance strategy, and hyperparameters.
        training: Shared training settings.
        scale_pos_weight: Ratio of legitimate to fraud rows in the training split.
    '''
    strategy, seed = candidate.imbalance_strategy, training.random_seed
    uses_class_weight = strategy == 'class_weight'
    classifier: Any
    if candidate.algorithm == 'logistic_regression':
        classifier = LogisticRegression(
            class_weight='balanced' if uses_class_weight else None,
            random_state=seed,
            **candidate.params,
        )
    elif candidate.algorithm == 'random_forest':
        classifier = RandomForestClassifier(
            class_weight='balanced_subsample' if uses_class_weight else None,
            random_state=seed,
            n_jobs=-1,
            **candidate.params,
        )
    else:
        classifier = XGBClassifier(
            scale_pos_weight=scale_pos_weight if strategy == 'scale_pos_weight' else 1.0,
            tree_method='hist',
            random_state=seed,
            n_jobs=-1,
            **candidate.params,
        )

    if strategy == 'smote':
        # The imblearn pipeline resamples during fit only, never when predicting
        sampler = SMOTE(sampling_strategy=training.smote_sampling_ratio, random_state=seed)
        return ResamplingPipeline([('smote', sampler), (CLASSIFIER_STEP, classifier)])
    return classifier


def train_candidate(
    candidate: CandidateConfig,
    prepared: PreparedData,
    config: AppConfig,
    experiment_id: str,
    reference_profile: dict[str, Any],
) -> CandidateResult:
    '''Fit one candidate, evaluate it on validation data, and log an MLflow run.'''
    logger.info('Training %s', candidate.name)
    train, validation = prepared.processed.train, prepared.processed.validation
    fraud_count = int(train.target.sum())
    estimator = build_classifier(
        candidate, config.training, scale_pos_weight=(len(train.target) - fraud_count) / fraud_count
    )
    started = time.perf_counter()
    estimator.fit(train.features, train.target)
    fit_seconds = time.perf_counter() - started
    # Serving needs the classifier alone, without the sampler
    is_resampled = isinstance(estimator, ResamplingPipeline)
    classifier = estimator.named_steps[CLASSIFIER_STEP] if is_resampled else estimator

    target = validation.target.to_numpy()
    scores = classifier.predict_proba(validation.features)[:, 1]
    threshold_table = build_threshold_table(target, scores, config.threshold)
    selection = select_threshold(threshold_table, config.threshold)
    result = CandidateResult(
        config=candidate,
        classifier=classifier,
        fit_seconds=fit_seconds,
        validation_scores=scores,
        ranking=compute_ranking_metrics(target, scores),
        threshold_table=threshold_table,
        selection=selection,
        at_threshold=compute_threshold_metrics(
            target, scores, selection.threshold, config.threshold
        ),
        run_id='',
        model_uri='',
        model_id='',
    )
    run_id, model_info = _log_candidate_run(
        result, prepared, config, experiment_id, reference_profile
    )
    logger.info(
        'Validation PR-AUC %.4f, ROC-AUC %.4f, threshold %.3f',
        result.ranking['pr_auc'],
        result.ranking['roc_auc'],
        selection.threshold,
    )
    return replace(
        result, run_id=run_id, model_uri=model_info.model_uri, model_id=model_info.model_id
    )


def select_champion(results: list[CandidateResult], selection_metric: str) -> CandidateResult:
    '''Return the candidate with the best validation score on the selection metric.'''
    return max(results, key=lambda result: result.ranking[selection_metric])


def evaluate_on_test(
    champion: CandidateResult, test: ModelData, config: AppConfig
) -> TestEvaluation:
    '''Score the test split once, with the model and threshold already fixed.'''
    target = test.target.to_numpy()
    scores = champion.classifier.predict_proba(test.features)[:, 1]
    threshold = champion.selection.threshold

    def at(value: float) -> ThresholdMetrics:
        return compute_threshold_metrics(target, scores, value, config.threshold)

    return TestEvaluation(
        scores=scores,
        ranking=compute_ranking_metrics(target, scores),
        at_threshold=at(threshold),
        at_max_f1=at(champion.selection.max_f1_threshold),
        at_default=at(0.5),
        intervals=bootstrap_confidence_intervals(
            target,
            scores,
            threshold,
            config.threshold,
            sample_count=config.evaluation.bootstrap_samples,
            random_seed=config.training.random_seed,
        ),
        no_model_cost=cost_without_model(target, config.threshold),
    )


def explain_champion(
    champion: CandidateResult, validation: ModelData, config: AppConfig, figure_dir: Path
) -> ChampionExplanation:
    '''Compute global importance and two local explanations on validation data.'''
    settings, seed = config.explainability, config.training.random_seed
    sample = sample_for_explanation(
        validation.features, validation.target, settings.sample_size, seed
    )
    explainer = ShapExplainer(champion.classifier, MODEL_FEATURES)
    shap_values = explainer.shap_values(sample)
    shap_importance = mean_absolute_shap(shap_values, MODEL_FEATURES)
    top_features = list(shap_importance.index[: settings.top_features])

    plotting.plot_feature_importance(
        shap_importance.head(settings.top_features),
        'Most important features by mean absolute SHAP value',
        'Mean absolute SHAP value',
        figure_dir / FIGURE_FILES['shap_importance'],
    )
    plotting.plot_shap_summary(
        shap_values, sample, top_features, figure_dir / FIGURE_FILES['shap_summary']
    )

    # Explain the best-scored and the worst-scored fraud case in the validation split
    fraud_rows = np.flatnonzero(validation.target.to_numpy() == 1)
    fraud_scores = champion.validation_scores[fraud_rows]
    examples = {}
    for key, row in (
        ('detected_example', int(fraud_rows[np.argmax(fraud_scores)])),
        ('missed_example', int(fraud_rows[np.argmin(fraud_scores)])),
    ):
        explanation = explainer.explain_row(validation.features.iloc[[row]], settings.top_features)
        score = float(champion.validation_scores[row])
        plotting.plot_local_explanation(
            explanation.to_frame(),
            f'Largest contributions, fraud scored {format_score(score)}',
            figure_dir / FIGURE_FILES[key],
        )
        examples[key] = ExplainedExample(fraud_probability=score, explanation=explanation)

    return ChampionExplanation(
        sample_size=len(sample),
        shap_importance=shap_importance,
        permutation_importance=compute_permutation_importance(
            champion.classifier,
            validation.features,
            validation.target,
            settings.permutation_repeats,
            seed,
        ),
        detected_example=examples['detected_example'],
        missed_example=examples['missed_example'],
    )


def register_champion(champion: CandidateResult, config: AppConfig) -> str:
    '''Register the champion as a new model version and move the champion alias to it.'''
    name = config.mlflow.registered_model_name
    # The client API is used because mlflow.register_model prints to the console
    client = MlflowClient()
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)
    version = client.create_model_version(
        name=name, source=champion.model_uri, run_id=champion.run_id, model_id=champion.model_id
    ).version
    client.set_registered_model_alias(name, config.mlflow.champion_alias, version)
    metric = config.training.selection_metric
    tags = {
        'candidate_name': champion.config.name,
        'decision_threshold': f'{champion.selection.threshold:.3f}',
        f'validation_{metric}': f'{champion.ranking[metric]:.4f}',
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    logger.info(
        'Registered %s version %s with alias %s', name, version, config.mlflow.champion_alias
    )
    return str(version)


def _log_candidate_run(
    result: CandidateResult,
    prepared: PreparedData,
    config: AppConfig,
    experiment_id: str,
    reference_profile: dict[str, Any],
) -> tuple[str, ModelInfo]:
    '''Log parameters, validation metrics, figures, and the serving pipeline.'''
    candidate, validation = result.config, prepared.processed.validation
    serving_pipeline = build_serving_pipeline(prepared.preprocessor, result.classifier)
    input_example = prepared.splits.train[INPUT_COLUMNS].head(5)
    signature = infer_signature(input_example, serving_pipeline.predict_proba(input_example)[:, 1])
    title = f'{candidate.name}, validation split'

    with (
        mlflow.start_run(experiment_id=experiment_id, run_name=candidate.name) as run,
        tempfile.TemporaryDirectory() as scratch,
    ):
        mlflow.set_tags(
            {
                'dataset_label': prepared.dataset_label,
                'dataset_sha256': prepared.dataset_sha256,
                'split': 'validation',
            }
        )
        mlflow.log_params(
            {
                'algorithm': candidate.algorithm,
                'imbalance_strategy': candidate.imbalance_strategy,
                'random_seed': config.training.random_seed,
                'train_rows': len(prepared.processed.train.target),
                'train_fraud_rows': int(prepared.processed.train.target.sum()),
                'false_positive_cost': config.threshold.false_positive_cost,
                'false_negative_cost': config.threshold.false_negative_cost,
                'max_false_positive_rate': config.threshold.max_false_positive_rate,
                **candidate.params,
            }
        )
        mlflow.log_metrics(
            {
                'fit_seconds': result.fit_seconds,
                **_prefixed('validation', result.ranking),
                **_prefixed('validation', result.at_threshold.to_dict()),
            }
        )
        target = validation.target.to_numpy()
        _plot_evaluation_figures(
            target, result.validation_scores, result.at_threshold, config, title, Path(scratch)
        )
        plotting.plot_threshold_analysis(
            result.threshold_table,
            result.selection,
            config.threshold,
            f'Threshold analysis, {title}',
            Path(scratch) / FIGURE_FILES['threshold_analysis'],
        )
        result.threshold_table.to_csv(Path(scratch) / 'threshold_analysis.csv', index=False)
        mlflow.log_artifacts(scratch, artifact_path='validation')

        model_info = mlflow.sklearn.log_model(
            serving_pipeline,
            name='model',
            signature=signature,
            input_example=input_example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            pyfunc_predict_fn='predict_proba',
            # The service reads its threshold and drift baseline from the model itself
            metadata={
                'candidate_name': candidate.name,
                'decision_threshold': result.selection.threshold,
                'reference_profile': reference_profile,
            },
        )
    return run.info.run_id, model_info


def _plot_evaluation_figures(
    target: np.ndarray,
    scores: np.ndarray,
    at_threshold: ThresholdMetrics,
    config: AppConfig,
    title: str,
    figure_dir: Path,
) -> None:
    '''Write the ROC, precision-recall, confusion matrix, and calibration figures.'''
    ranking = compute_ranking_metrics(target, scores)
    roc_auc, pr_auc = ranking['roc_auc'], ranking['pr_auc']
    plotting.plot_roc_curve(
        target, scores, at_threshold, f'ROC curve, {title}\nROC-AUC {roc_auc:.3f}',
        figure_dir / FIGURE_FILES['roc_curve'],
    )  # fmt: skip
    plotting.plot_precision_recall_curve(
        target, scores, at_threshold, f'Precision-recall curve, {title}\nPR-AUC {pr_auc:.3f}',
        figure_dir / FIGURE_FILES['pr_curve'],
    )  # fmt: skip
    plotting.plot_confusion_matrix(
        at_threshold,
        f'Confusion matrix at threshold {at_threshold.threshold:.3f}\n{title}',
        figure_dir / FIGURE_FILES['confusion_matrix'],
    )
    plotting.plot_calibration_curve(
        compute_calibration_table(target, scores, config.evaluation.calibration_bins),
        ranking['brier_score'],
        f'Calibration, {title}',
        figure_dir / FIGURE_FILES['calibration'],
    )


def _plot_champion_figures(
    champion: CandidateResult,
    results: list[CandidateResult],
    test_evaluation: TestEvaluation,
    prepared: PreparedData,
    config: AppConfig,
    figure_dir: Path,
) -> None:
    '''Write the report figures: test-split diagnostics and validation comparisons.'''
    name = champion.config.name
    processed = prepared.processed
    _plot_evaluation_figures(
        processed.test.target.to_numpy(),
        test_evaluation.scores,
        test_evaluation.at_threshold,
        config,
        f'{name}, test split',
        figure_dir,
    )
    plotting.plot_threshold_analysis(
        champion.threshold_table,
        champion.selection,
        config.threshold,
        f'Threshold analysis, {name}, validation split',
        figure_dir / FIGURE_FILES['threshold_analysis'],
    )
    plotting.plot_precision_recall_comparison(
        processed.validation.target.to_numpy(),
        {result.config.name: result.validation_scores for result in results},
        figure_dir / FIGURE_FILES['pr_comparison'],
    )


def _build_report_data(
    prepared: PreparedData,
    results: list[CandidateResult],
    champion: CandidateResult,
    test_evaluation: TestEvaluation,
    explanation: ChampionExplanation,
    model_version: str,
    config: AppConfig,
) -> ReportData:
    candidates = tuple(
        CandidateSummary(
            name=result.config.name,
            imbalance_strategy=result.config.imbalance_strategy,
            pr_auc=result.ranking['pr_auc'],
            roc_auc=result.ranking['roc_auc'],
            brier_score=result.ranking['brier_score'],
            at_threshold=result.at_threshold,
            fit_seconds=result.fit_seconds,
        )
        for result in results
    )
    named_splits = {
        'train': prepared.splits.train,
        'validation': prepared.splits.validation,
        'test': prepared.splits.test,
    }
    splits = tuple(
        SplitSummary(name=name, rows=len(split), fraud_rows=int(split[TARGET_COLUMN].sum()))
        for name, split in named_splits.items()
    )
    return ReportData(
        dataset_label=prepared.dataset_label,
        dataset_sha256=prepared.dataset_sha256,
        is_synthetic=prepared.is_synthetic,
        validation_report=prepared.validation_report,
        cleaning=prepared.cleaning,
        splits=splits,
        candidates=candidates,
        selection_metric=config.training.selection_metric,
        champion_name=champion.config.name,
        champion_imbalance_strategy=champion.config.imbalance_strategy,
        model_version=model_version,
        threshold_config=config.threshold,
        selection=champion.selection,
        threshold_table=champion.threshold_table,
        test_ranking=test_evaluation.ranking,
        test_metrics=test_evaluation.at_threshold,
        test_at_max_f1=test_evaluation.at_max_f1,
        test_at_default=test_evaluation.at_default,
        test_intervals=test_evaluation.intervals,
        no_model_cost=test_evaluation.no_model_cost,
        explanation_sample_size=explanation.sample_size,
        shap_importance=explanation.shap_importance,
        permutation_importance=explanation.permutation_importance,
        top_feature_count=config.explainability.top_features,
        detected_example=explanation.detected_example,
        missed_example=explanation.missed_example,
    )


def _write_result_files(data: ReportData, champion: CandidateResult, output_dir: Path) -> None:
    '''Write machine-readable results next to the Markdown report.'''
    comparison = pd.DataFrame([candidate.to_dict() for candidate in data.candidates])
    comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
    data.threshold_table.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    validation_summary = json.dumps(data.validation_report.to_dict(), indent=2)
    (output_dir / 'validation_report.json').write_text(validation_summary + '\n', encoding='utf-8')
    metrics = {
        'dataset': {'label': data.dataset_label, 'sha256': data.dataset_sha256},
        'champion': {
            'name': data.champion_name,
            'model_version': data.model_version,
            'decision_threshold': data.selection.threshold,
            'review_budget_met': data.selection.budget_met,
        },
        'validation': {**champion.ranking, **champion.at_threshold.to_dict()},
        'test': {
            **data.test_ranking,
            **data.test_metrics.to_dict(),
            'intervals': {name: list(bounds) for name, bounds in data.test_intervals.items()},
        },
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')


def _log_champion_artifacts(
    champion: CandidateResult, test_evaluation: TestEvaluation, output_dir: Path
) -> None:
    '''Add test metrics and the report files to the MLflow run of the champion.'''
    with mlflow.start_run(run_id=champion.run_id):
        mlflow.log_metrics(
            {
                **_prefixed('test', test_evaluation.ranking),
                **_prefixed('test', test_evaluation.at_threshold.to_dict()),
            }
        )
        mlflow.log_artifacts(str(output_dir), artifact_path='report')


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f'{prefix}_{name}': float(value) for name, value in values.items()}
