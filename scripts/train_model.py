'''Train all candidates, register the champion, and write the evaluation report.'''

import argparse
import logging
import shutil
from pathlib import Path

from fraudguard.config import EnvironmentSettings, configure_logging, load_config
from fraudguard.data_validation import DataValidationError
from fraudguard.reporting import FIGURE_DIR_NAME
from fraudguard.training import TrainingSummary, run_training_pipeline

logger = logging.getLogger('train_model')

DOCS_DIR = Path('docs')


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument(
        '--use_sample_data',
        action='store_true',
        help='Train on the synthetic sample instead of creditcard.csv.',
    )
    parser.add_argument(
        '--publish_docs',
        action='store_true',
        help='Copy the evaluation report and its figures into docs/.',
    )
    return parser.parse_args()


def publish_docs(summary: TrainingSummary) -> None:
    '''Copy the generated report and figures into the versioned docs directory.'''
    report_dir = summary.report_path.parent
    shutil.copy2(summary.report_path, DOCS_DIR / summary.report_path.name)
    shutil.copytree(report_dir / FIGURE_DIR_NAME, DOCS_DIR / FIGURE_DIR_NAME, dirs_exist_ok=True)
    logger.info('Published report to %s', DOCS_DIR.as_posix())


def main() -> None:
    '''Run the training pipeline and print its headline results.'''
    arguments = parse_arguments()
    settings = EnvironmentSettings()
    configure_logging(settings.log_level)
    config = load_config(arguments.config_path, settings)

    try:
        summary = run_training_pipeline(config, use_sample_data=arguments.use_sample_data)
    except (FileNotFoundError, DataValidationError) as error:
        logger.error('%s', error)
        raise SystemExit(1) from error

    logger.info(
        'Champion %s registered as version %s', summary.champion_name, summary.model_version
    )
    logger.info('Decision threshold %.3f', summary.decision_threshold)
    logger.info(
        'Test PR-AUC %.4f, precision %.4f, recall %.4f',
        summary.test_pr_auc,
        summary.test_precision,
        summary.test_recall,
    )
    logger.info('Report written to %s', summary.report_path.as_posix())
    if arguments.publish_docs:
        publish_docs(summary)
    logger.info('Done')


if __name__ == '__main__':
    main()
