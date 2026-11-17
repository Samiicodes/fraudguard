'''Credit card fraud detection: training pipeline, evaluation, and inference service.'''

import os

# MLflow sends usage telemetry by default. This project is local-first, so telemetry
# stays off unless the variable is set explicitly before the package is imported.
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
# Progress bars and run links printed to stdout only clutter the pipeline output
os.environ.setdefault('MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR', 'false')
os.environ.setdefault('MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT', 'true')

__version__ = '0.1.0'
