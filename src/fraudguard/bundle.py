'''A self-contained copy of the champion model, for serving without the registry.

A bundle is a directory holding the MLflow model, a manifest with its registry name, version,
and test metrics, and optionally a few example transactions for the demo page. It lets the
service run where the local MLflow registry is not available, such as a container in the cloud.
'''

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME

from fraudguard.config import MlflowConfig
from fraudguard.data_ingestion import INPUT_COLUMNS, TARGET_COLUMN
from fraudguard.inference import FraudModel, model_from_artifacts

MODEL_DIR = 'model'
MANIFEST_FILE = 'manifest.json'
EXAMPLES_FILE = 'examples.json'
_TEST_METRIC_PREFIX = 'test_'

# Demo groups in display order: what happened, whether the transaction was fraud, whether the
# model flagged it, and how many examples to show. The errors are there on purpose: a demo of
# only correct answers would hide the trade-off that the decision threshold makes.
DEMO_GROUPS: tuple[tuple[str, bool, bool, int], ...] = (
    ('Fraud the model catches', True, True, 2),
    ('Fraud the model misses', True, False, 1),
    ('Legitimate payment, approved', False, False, 2),
    ('Legitimate payment, flagged by mistake', False, True, 1),
)


@dataclass(frozen=True)
class BundleManifest:
    '''Where a bundled model came from and how it scored on the test split.'''

    model_name: str
    model_version: str
    algorithm: str | None
    dataset_sha256: str | None
    exported_at: str
    test_metrics: dict[str, float]


@dataclass(frozen=True)
class DemoExample:
    '''A held-out transaction shown on the demo page.'''

    group: str
    actual_label: str
    transaction: dict[str, float]


def export_champion(config: MlflowConfig, bundle_dir: Path) -> BundleManifest:
    '''Copy the model behind the champion alias, with its test metrics, into a new bundle.

    Raises:
        FileExistsError: If the bundle directory already holds a model.
        mlflow.exceptions.MlflowException: If the model or the alias does not exist.
    '''
    model_dir = bundle_dir / MODEL_DIR
    if model_dir.exists():
        raise FileExistsError(f'{model_dir} already exists')
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(config.registered_model_name, config.champion_alias)
    with tempfile.TemporaryDirectory() as download_dir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f'models:/{config.registered_model_name}/{version.version}',
            dst_path=download_dir,
        )
        shutil.copytree(local_path, model_dir)
    # The download directory is private to this user (mode 0700) and copytree keeps that mode,
    # but images run the service as another user, so the bundle must be readable by everyone
    for path in [model_dir, *model_dir.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    # MLflow records the absolute artefact path on the exporting machine. A bundle is loaded
    # from wherever it is copied to, so only its own relative location is kept.
    descriptor_path = model_dir / MLMODEL_FILE_NAME
    descriptor = Model.load(descriptor_path)
    descriptor.artifact_path = MODEL_DIR
    descriptor.save(descriptor_path)

    run = client.get_run(version.run_id).data if version.run_id else None
    metrics = run.metrics if run else {}
    manifest = BundleManifest(
        model_name=config.registered_model_name,
        model_version=str(version.version),
        algorithm=run.params.get('algorithm') if run else None,
        dataset_sha256=run.tags.get('dataset_sha256') if run else None,
        exported_at=datetime.now(UTC).isoformat(timespec='seconds'),
        test_metrics={
            name.removeprefix(_TEST_METRIC_PREFIX): float(value)
            for name, value in sorted(metrics.items())
            if name.startswith(_TEST_METRIC_PREFIX)
        },
    )
    _write_json(bundle_dir / MANIFEST_FILE, asdict(manifest))
    return manifest


def read_manifest(bundle_dir: Path) -> BundleManifest:
    '''Read the manifest of a bundle written by export_champion.

    Raises:
        FileNotFoundError: If the directory is not a bundle.
    '''
    path = bundle_dir / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f'No model bundle manifest at {path}')
    return BundleManifest(**json.loads(path.read_text(encoding='utf-8')))


def load_model_bundle(bundle_dir: Path) -> FraudModel:
    '''Load the model in a bundle written by export_champion.

    Raises:
        FileNotFoundError: If the directory is not a bundle.
    '''
    manifest = read_manifest(bundle_dir)
    return model_from_artifacts(
        str(bundle_dir / MODEL_DIR), manifest.model_name, manifest.model_version
    )


def choose_demo_examples(
    transactions: pd.DataFrame, model: FraudModel, seed: int = 0
) -> list[DemoExample]:
    '''Pick held-out transactions that show each kind of outcome, right and wrong.

    Args:
        transactions: Labelled transactions in the canonical schema, from the test split.
        model: The model whose decisions define the groups.
        seed: Seed for the choice within each group.

    Returns:
        Examples in the order of DEMO_GROUPS. A group with no matching transaction is skipped.
    '''
    inputs = transactions[INPUT_COLUMNS]
    predictions = model.predict(inputs)
    flagged = np.array([prediction.is_fraud for prediction in predictions])
    is_fraud = transactions[TARGET_COLUMN].to_numpy().astype(bool)
    values = inputs.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    examples: list[DemoExample] = []
    for group, fraud, flag, count in DEMO_GROUPS:
        candidates = np.flatnonzero((is_fraud == fraud) & (flagged == flag))
        if len(candidates) == 0:
            continue
        chosen = np.sort(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
        examples.extend(
            DemoExample(
                group=group,
                actual_label='fraud' if fraud else 'legitimate',
                transaction=dict(zip(INPUT_COLUMNS, values[row].tolist(), strict=True)),
            )
            for row in chosen
        )
    return examples


def write_demo_examples(examples: list[DemoExample], bundle_dir: Path) -> Path:
    '''Store demo examples in the bundle and return the file path.'''
    path = bundle_dir / EXAMPLES_FILE
    _write_json(path, [asdict(example) for example in examples])
    return path


def read_demo_examples(bundle_dir: Path) -> list[DemoExample]:
    '''Return the demo examples stored in a bundle, or none if it has no examples file.'''
    path = bundle_dir / EXAMPLES_FILE
    if not path.is_file():
        return []
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding='utf-8'))
    return [DemoExample(**item) for item in raw]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
