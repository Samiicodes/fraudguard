'''Export the champion model and demo transactions to a bundle the API can serve without MLflow.

Point the service at the bundle with FRAUDGUARD_MODEL_BUNDLE. The demo transactions come from
the test split of the dataset the champion was trained on, rebuilt with the same cleaning and
seeded split as training, so none of them was seen during training or model selection.
'''

import argparse
import logging
import shutil
from pathlib import Path

from fraudguard.bundle import (
    MANIFEST_FILE,
    choose_demo_examples,
    export_champion,
    load_model_bundle,
    write_demo_examples,
)
from fraudguard.config import EnvironmentSettings, configure_logging, load_config
from fraudguard.data_ingestion import (
    compute_file_sha256,
    load_raw_transactions,
    split_transactions,
)
from fraudguard.data_validation import clean_transactions

logger = logging.getLogger('export_model')

DEFAULT_OUTPUT_DIR = Path('dist/model_bundle')


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Bundle directory. A previous bundle there is replaced.',
    )
    parser.add_argument(
        '--use_sample_data',
        action='store_true',
        help='Take demo transactions from the synthetic sample instead of creditcard.csv.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Seed for picking demo transactions.')
    return parser.parse_args()


def clear_previous_bundle(output_dir: Path) -> None:
    '''Remove an earlier bundle, refusing to touch a directory that is not one.'''
    if not output_dir.exists():
        return
    if not (output_dir / MANIFEST_FILE).is_file():
        raise SystemExit(f'{output_dir} exists and is not a model bundle; choose another path')
    shutil.rmtree(output_dir)


def main() -> None:
    '''Export the bundle and print what it contains.'''
    arguments = parse_arguments()
    settings = EnvironmentSettings()
    configure_logging(settings.log_level)
    config = load_config(arguments.config_path, settings)
    data_path = config.data.sample_path if arguments.use_sample_data else config.data.raw_path
    output_dir: Path = arguments.output_dir

    clear_previous_bundle(output_dir)
    logger.info(
        'Exporting %s@%s', config.mlflow.registered_model_name, config.mlflow.champion_alias
    )
    manifest = export_champion(config.mlflow, output_dir)
    logger.info('Exported version %s to %s', manifest.model_version, output_dir.as_posix())

    # Examples from another file would not be the model's test split, and could be training rows
    if manifest.dataset_sha256 != compute_file_sha256(data_path):
        shutil.rmtree(output_dir)
        raise SystemExit(
            f'{data_path.as_posix()} is not the dataset the champion was trained on. '
            'Retrain, or pass --use_sample_data if it was trained on the sample.'
        )

    logger.info('Rebuilding the test split of %s', data_path.as_posix())
    cleaned, _ = clean_transactions(load_raw_transactions(data_path))
    test = split_transactions(cleaned, config.split).test
    examples = choose_demo_examples(test, load_model_bundle(output_dir), arguments.seed)
    write_demo_examples(examples, output_dir)
    logger.info('Picked %d demo transactions from %s test rows', len(examples), f'{len(test):,}')
    logger.info('Done')


if __name__ == '__main__':
    main()
