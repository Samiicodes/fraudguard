'''Start the inference API with uvicorn.'''

import argparse
from pathlib import Path

import uvicorn

from fraudguard.api import create_app
from fraudguard.config import (
    SERVICE_LOG_FORMAT,
    EnvironmentSettings,
    configure_logging,
    load_config,
)


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument('--host', help='Interface to bind. Defaults to api.host in the config.')
    parser.add_argument(
        '--port', type=int, help='Port to bind. Defaults to api.port in the config.'
    )
    return parser.parse_args()


def main() -> None:
    '''Load the configuration and serve the application.'''
    arguments = parse_arguments()
    settings = EnvironmentSettings()
    configure_logging(settings.log_level, SERVICE_LOG_FORMAT)
    config = load_config(arguments.config_path, settings)
    # Uvicorn keeps the logging set up above; the API middleware logs each request
    uvicorn.run(
        create_app(config),
        host=arguments.host or config.api.host,
        port=arguments.port or config.api.port,
        log_config=None,
        access_log=False,
    )


if __name__ == '__main__':
    main()
