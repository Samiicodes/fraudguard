'''Generate a synthetic transaction file with the same columns as creditcard.csv.'''

import argparse
import logging
from pathlib import Path

from fraudguard.config import configure_logging, load_config
from fraudguard.data_ingestion import generate_synthetic_transactions

logger = logging.getLogger('generate_sample_data')


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument(
        '--output_path', type=Path, help='Output CSV. Defaults to data.sample_path in the config.'
    )
    parser.add_argument('--n_rows', type=int, default=50_000, help='Number of rows to generate.')
    parser.add_argument('--fraud_rate', type=float, default=0.005, help='Share of fraud rows.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def main() -> None:
    '''Generate the sample file and report where it was written.'''
    arguments = parse_arguments()
    configure_logging()
    output_path = arguments.output_path or load_config(arguments.config_path).data.sample_path

    logger.info('Generating %s synthetic transactions', f'{arguments.n_rows:,}')
    frame = generate_synthetic_transactions(
        n_rows=arguments.n_rows,
        fraud_rate=arguments.fraud_rate,
        random_seed=arguments.random_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info('Saved %s', output_path.as_posix())
    logger.info('Done')


if __name__ == '__main__':
    main()
