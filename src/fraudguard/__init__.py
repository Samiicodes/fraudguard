'''Credit card fraud detection: training pipeline, evaluation, and inference service.'''

import os

# MLflow sends usage telemetry by default. This project is local-first, so telemetry
# stays off unless the variable is set explicitly before the package is imported.
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
# Progress bars for local artefact copies only clutter the pipeline output
os.environ.setdefault('MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR', 'false')

__version__ = '0.1.0'
