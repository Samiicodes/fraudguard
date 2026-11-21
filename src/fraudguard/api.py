'''FastAPI service that scores card transactions with the registered champion model.'''

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fraudguard import __version__
from fraudguard.bundle import load_model_bundle, read_demo_examples, read_manifest
from fraudguard.config import AppConfig, load_config
from fraudguard.inference import FraudModel, load_champion_model
from fraudguard.monitoring import PredictionMonitor

logger = logging.getLogger(__name__)

ModelLoader = Callable[[AppConfig], FraudModel]

# Client-supplied request ids are echoed into logs, so only plain tokens are accepted
_REQUEST_ID_PATTERN = re.compile(r'[A-Za-z0-9._-]{1,64}')
_DEMO_PAGE = resources.files('fraudguard').joinpath('demo.html').read_text(encoding='utf-8')

# First row of creditcard.csv, a legitimate transaction
_EXAMPLE_TRANSACTION: dict[str, Any] = {
    'time_seconds': 0.0,
    'v1': -1.359807, 'v2': -0.072781, 'v3': 2.536347, 'v4': 1.378155, 'v5': -0.338321,
    'v6': 0.462388, 'v7': 0.239599, 'v8': 0.098698, 'v9': 0.363787, 'v10': 0.090794,
    'v11': -0.5516, 'v12': -0.617801, 'v13': -0.99139, 'v14': -0.311169, 'v15': 1.468177,
    'v16': -0.470401, 'v17': 0.207971, 'v18': 0.025791, 'v19': 0.403993, 'v20': 0.251412,
    'v21': -0.018307, 'v22': 0.277838, 'v23': -0.110474, 'v24': 0.066928, 'v25': 0.128539,
    'v26': -0.189115, 'v27': 0.133558, 'v28': -0.021053,
    'amount': 149.62,
}  # fmt: skip


class TransactionRequest(BaseModel):
    '''One card transaction in the units of creditcard.csv.'''

    model_config = ConfigDict(
        extra='forbid', allow_inf_nan=False, json_schema_extra={'examples': [_EXAMPLE_TRANSACTION]}
    )

    time_seconds: float = Field(
        ge=0, description='Seconds since the first transaction of the reference period.'
    )
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float = Field(ge=0, description='Transaction amount.')


class FeatureContributionResponse(BaseModel):
    '''Contribution of one model feature to the score.'''

    feature: str
    value: float = Field(description='Standardised model input.')
    contribution: float = Field(description='SHAP value. Positive values raise the score.')


class ExplanationResponse(BaseModel):
    '''Largest feature contributions behind a score.'''

    base_value: float
    output_unit: Literal['log_odds', 'probability']
    contributions: list[FeatureContributionResponse]


class PredictionResponse(BaseModel):
    '''Score and decision for one transaction.'''

    request_id: str
    fraud_probability: float = Field(
        description='Model score between 0 and 1. Class weighting can push it above the '
        'true probability, so use predicted_label for decisions.'
    )
    predicted_label: Literal['fraud', 'legitimate']
    decision_threshold: float
    model_name: str
    model_version: str
    latency_ms: float
    explanation: ExplanationResponse | None


class HealthResponse(BaseModel):
    '''Service and model status.'''

    status: Literal['ok', 'unavailable']
    model_loaded: bool
    model_name: str | None
    model_version: str | None
    service_version: str


class DemoExampleResponse(BaseModel):
    '''A transaction from the held-out test split, with its true label.'''

    group: str
    actual_label: Literal['fraud', 'legitimate']
    transaction: TransactionRequest


class DemoResponse(BaseModel):
    '''Content for the demo page.'''

    algorithm: str | None
    test_metrics: dict[str, float] = Field(
        description='Metrics of the served model on the test split, at its decision threshold.'
    )
    examples: list[DemoExampleResponse]


router = APIRouter()


def get_model(request: Request) -> FraudModel:
    '''Return the loaded model, or answer 503 when none is available.'''
    model: FraudModel | None = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Model is not loaded. Check /health and the service logs.',
        )
    return model


@router.get('/health', response_model=HealthResponse, responses={503: {'model': HealthResponse}})
def health(request: Request, response: Response) -> HealthResponse:
    '''Report whether the service has a model to serve.'''
    model: FraudModel | None = request.app.state.model
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status='ok' if model else 'unavailable',
        model_loaded=model is not None,
        model_name=model.model_name if model else None,
        model_version=model.model_version if model else None,
        service_version=__version__,
    )


@router.post('/predict', response_model=PredictionResponse)
def predict(
    transaction: TransactionRequest,
    request: Request,
    model: Annotated[FraudModel, Depends(get_model)],
    explain: bool = True,
) -> PredictionResponse:
    '''Score one transaction and, unless explain is false, list its top contributions.'''
    started = time.perf_counter()
    inputs = transaction.model_dump()
    config: AppConfig = request.app.state.config
    top_k = config.api.explanation_top_k if explain else 0
    prediction = model.predict(pd.DataFrame([inputs]), explanation_top_k=top_k)[0]
    latency_ms = (time.perf_counter() - started) * 1000

    request_id: str = request.state.request_id
    monitor: PredictionMonitor = request.app.state.monitor
    monitor.record(
        request_id=request_id,
        inputs=inputs,
        fraud_probability=prediction.fraud_probability,
        is_flagged=prediction.is_fraud,
        latency_ms=latency_ms,
        model_version=model.model_version,
    )
    explanation = prediction.explanation
    return PredictionResponse(
        request_id=request_id,
        fraud_probability=prediction.fraud_probability,
        predicted_label='fraud' if prediction.is_fraud else 'legitimate',
        decision_threshold=model.decision_threshold,
        model_name=model.model_name,
        model_version=model.model_version,
        latency_ms=round(latency_ms, 3),
        explanation=ExplanationResponse.model_validate(explanation, from_attributes=True)
        if explanation
        else None,
    )


@router.get('/', include_in_schema=False)
def demo_page() -> HTMLResponse:
    '''Serve the demo page, which scores test transactions through /predict.'''
    return HTMLResponse(_DEMO_PAGE)


@router.get('/demo/examples', response_model=DemoResponse)
def demo_examples(request: Request) -> DemoResponse:
    '''Return the test transactions and metrics for the demo page.

    Both come from the model bundle, so the list is empty when the service loads its model
    from the registry.
    '''
    demo: DemoResponse = request.app.state.demo
    return demo


@router.get('/metrics')
def metrics(request: Request) -> dict[str, Any]:
    '''Return prediction counts, latency, input statistics, and drift status.'''
    model: FraudModel | None = request.app.state.model
    monitor: PredictionMonitor = request.app.state.monitor
    return {
        'model_name': model.model_name if model else None,
        'model_version': model.model_version if model else None,
        'decision_threshold': model.decision_threshold if model else None,
        **monitor.summary(),
    }


async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    '''Return field-level validation errors without echoing the submitted values.'''
    # The default handler echoes each input value, which cannot be serialised when it is NaN
    detail = [
        {'loc': list(item['loc']), 'msg': item['msg'], 'type': item['type']}
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={'detail': detail, 'request_id': request.state.request_id},
    )


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    '''Attach a request id, log every request, and turn unhandled errors into a 500.'''
    supplied_id = request.headers.get('x-request-id', '')
    request_id = supplied_id if _REQUEST_ID_PATTERN.fullmatch(supplied_id) else uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Keep the traceback in the logs and out of the response
        logger.exception('Unhandled error request_id=%s', request_id)
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'detail': 'Internal server error', 'request_id': request_id},
        )
    response.headers['X-Request-ID'] = request_id
    logger.info(
        'request_id=%s method=%s path=%s status=%d duration_ms=%.1f',
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


def load_configured_model(config: AppConfig) -> FraudModel:
    '''Load the model bundle when api.model_bundle is set, otherwise the registry champion.'''
    if config.api.model_bundle is not None:
        return load_model_bundle(config.api.model_bundle)
    return load_champion_model(config.mlflow)


def create_app(
    config: AppConfig | None = None, model_loader: ModelLoader = load_configured_model
) -> FastAPI:
    '''Build the application.

    Args:
        config: Project configuration. Read from the default location when omitted.
        model_loader: Returns the model to serve. Tests replace the default loader.
    '''

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_config = config or load_config()
        model = _load_model(app_config, model_loader)
        app.state.config = app_config
        app.state.model = model
        app.state.demo = _load_demo(app_config.api.model_bundle if model else None)
        app.state.monitor = PredictionMonitor(
            app_config.monitoring, model.reference_profile if model else None
        )
        yield

    app = FastAPI(
        title='fraudguard',
        summary='Scores card transactions with the registered champion fraud model.',
        version=__version__,
        lifespan=lifespan,
    )
    app.middleware('http')(log_requests)
    app.exception_handler(RequestValidationError)(validation_error_handler)
    app.include_router(router)
    return app


def _load_model(config: AppConfig, model_loader: ModelLoader) -> FraudModel | None:
    '''Load the model, or log the failure and return None so /health can report it.'''
    try:
        model = model_loader(config)
    except Exception:
        if config.api.model_bundle is not None:
            logger.exception('Could not load the model bundle at %s', config.api.model_bundle)
        else:
            logger.exception(
                'Could not load model %s with alias %s from %s',
                config.mlflow.registered_model_name,
                config.mlflow.champion_alias,
                config.mlflow.tracking_uri,
            )
        return None
    logger.info(
        'Loaded model %s version %s with decision threshold %.3f',
        model.model_name,
        model.model_version,
        model.decision_threshold,
    )
    return model


def _load_demo(bundle_dir: Path | None) -> DemoResponse:
    '''Read the demo page content from the bundle the model was loaded from, if any.'''
    if bundle_dir is None:
        return DemoResponse(algorithm=None, test_metrics={}, examples=[])
    manifest = read_manifest(bundle_dir)
    return DemoResponse.model_validate(
        {
            'algorithm': manifest.algorithm,
            'test_metrics': manifest.test_metrics,
            'examples': [asdict(example) for example in read_demo_examples(bundle_dir)],
        }
    )
