'''Deploy the API with a model bundle to Google Cloud Run.

Run scripts/export_model.py first. Deploying needs the gcloud CLI, logged in, with a default
project that has billing and the Cloud Run, Cloud Build, and Artifact Registry APIs enabled;
docs/deployment.md lists the one-off setup. The script stages the files the image needs, and
`gcloud run deploy --source` builds them with Cloud Build and replaces the running revision.
'''

import argparse
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from fraudguard.bundle import MANIFEST_FILE, read_manifest
from fraudguard.config import configure_logging

logger = logging.getLogger('deploy_cloud_run')

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / 'deploy' / 'cloud_run' / 'Dockerfile'
DEFAULT_BUNDLE_DIR = Path('dist/model_bundle')
DEFAULT_STAGING_DIR = Path('dist/cloud_run')
STAGED_BUNDLE_DIR = 'model_bundle'
# Copied from the repository as they are; the image builds from nothing else
REPOSITORY_FILES = (
    'pyproject.toml',
    'uv.lock',
    'README.md',
    'LICENSE',
    'config',
    'src',
    'scripts/run_api.py',
)
# One instance keeps the worst-case bill bounded; the demo never needs more
SERVICE_SETTINGS = (
    '--allow-unauthenticated',
    '--port=8080',
    '--cpu=1',
    '--memory=1Gi',
    '--min-instances=0',
    '--max-instances=1',
    '--cpu-boost',
)
HEALTH_ATTEMPTS = 12
HEALTH_DELAY_SECONDS = 10


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', default='fraudguard', help='Cloud Run service name.')
    parser.add_argument('--region', default='europe-west1', help='Cloud Run region.')
    parser.add_argument('--project', help='Google Cloud project. Defaults to the gcloud one.')
    parser.add_argument('--bundle_dir', type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument('--staging_dir', type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument(
        '--stage_only', action='store_true', help='Stage the files without deploying them.'
    )
    parser.add_argument(
        '--allow_dirty',
        action='store_true',
        help='Deploy even if tracked files have uncommitted changes.',
    )
    return parser.parse_args()


def git_revision() -> tuple[str, bool]:
    '''Return the current commit and whether tracked files have uncommitted changes.'''

    def git(*arguments: str) -> str:
        return subprocess.run(
            ['git', *arguments], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return git('rev-parse', '--short', 'HEAD'), bool(
        git('status', '--porcelain', '--untracked-files=no')
    )


def stage(bundle_dir: Path, staging_dir: Path) -> None:
    '''Copy the code, the Dockerfile, and the bundle into a fresh staging directory.'''
    if staging_dir.exists():
        if not (staging_dir / STAGED_BUNDLE_DIR / MANIFEST_FILE).is_file():
            raise SystemExit(f'{staging_dir} exists and is not a staged build; choose another path')
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in REPOSITORY_FILES:
        source, target = ROOT / name, staging_dir / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(DOCKERFILE, staging_dir / 'Dockerfile')
    shutil.copytree(bundle_dir, staging_dir / STAGED_BUNDLE_DIR)


def gcloud(*arguments: str, project: str | None) -> str:
    '''Run a gcloud command and return its output.'''
    # On Windows gcloud is a .cmd file, which only resolves through its full path
    executable = shutil.which('gcloud')
    if executable is None:
        raise SystemExit('gcloud not found; install the Google Cloud CLI first')
    command = [executable, *arguments, '--quiet']
    if project:
        command.append(f'--project={project}')
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def wait_for_health(url: str) -> str:
    '''Return the /health body once the service answers, or stop after a few attempts.'''
    for attempt in range(1, HEALTH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(f'{url}/health', timeout=60) as response:
                body: str = response.read().decode('utf-8')
                return body
        except (urllib.error.URLError, TimeoutError) as error:
            logger.info('Health check %d failed: %s', attempt, error)
            time.sleep(HEALTH_DELAY_SECONDS)
    raise SystemExit(f'{url}/health did not answer; check the Cloud Run logs')


def main() -> None:
    '''Stage, deploy, and check the service.'''
    arguments = parse_arguments()
    configure_logging()
    bundle_dir: Path = arguments.bundle_dir
    if not (bundle_dir / MANIFEST_FILE).is_file():
        raise SystemExit(f'No model bundle at {bundle_dir}; run scripts/export_model.py first')
    manifest = read_manifest(bundle_dir)

    commit, dirty = git_revision()
    if dirty and not arguments.allow_dirty:
        raise SystemExit(
            'Tracked files have uncommitted changes; commit them or pass --allow_dirty'
        )

    logger.info('Staging commit %s with model version %s', commit, manifest.model_version)
    stage(bundle_dir, arguments.staging_dir)
    if arguments.stage_only:
        logger.info('Staged files in %s', arguments.staging_dir.as_posix())
        return

    logger.info('Building and deploying %s to %s', arguments.service, arguments.region)
    # Label values allow lowercase letters, digits, dashes, and underscores only
    labels = f'commit={commit}{"-dirty" if dirty else ""},model-version={manifest.model_version}'
    gcloud(
        'run',
        'deploy',
        arguments.service,
        f'--source={arguments.staging_dir}',
        f'--region={arguments.region}',
        *SERVICE_SETTINGS,
        f'--labels={labels}',
        project=arguments.project,
    )
    url = gcloud(
        'run',
        'services',
        'describe',
        arguments.service,
        f'--region={arguments.region}',
        '--format=value(status.url)',
        project=arguments.project,
    )
    logger.info('Health %s', wait_for_health(url))
    logger.info('Live at %s', url)
    logger.info('Done')


if __name__ == '__main__':
    main()
