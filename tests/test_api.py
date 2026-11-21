'''Tests for request validation, responses, and error handling of the API.'''

import json
from collections.abc import Callable, Iterator
from typing import Any, cast

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudguard.api import create_app
from fraudguard.config import AppConfig
from fraudguard.data_ingestion import INPUT_COLUMNS
from fraudguard.inference import FraudModel, FraudPrediction
from fraudguard.training import TrainingSummary

TrainedPipeline = tuple[AppConfig, TrainingSummary]
Payload = dict[str, Any]

INVALID_PAYLOADS: dict[str, Callable[[Payload], Payload]] = {
    'missing_field': lambda body: {name: value for name, value in body.items() if name != 'v14'},
    'unknown_field': lambda body: {**body, 'card_number': 4000123412341234},
    'negative_amount': lambda body: {**body, 'amount': -1.0},
    'negative_time': lambda body: {**body, 'time_seconds': -10.0},
    'text_value': lambda body: {**body, 'v1': 'high'},
    'null_value': lambda body: {**body, 'v2': None},
}


class BrokenModel:
    '''Stands in for a model whose prediction fails unexpectedly.'''

    model_name = 'broken'
    model_version = '0'
    decision_threshold = 0.5
    reference_profile = None

    def predict(
        self, transactions: pd.DataFrame, explanation_top_k: int = 0
    ) -> list[FraudPrediction]:
        raise RuntimeError('database password is hunter2')


def unavailable_registry(config: AppConfig) -> FraudModel:
    raise RuntimeError('registry unreachable')


@pytest.fixture(scope='module')
def client(trained_pipeline: TrainedPipeline) -> Iterator[TestClient]:
    config, _ = trained_pipeline
    with TestClient(create_app(config)) as test_client:
        yield test_client


@pytest.fixture
def payload(transactions: pd.DataFrame) -> Payload:
    row = transactions[INPUT_COLUMNS].iloc[0]
    return {str(name): float(value) for name, value in row.items()}


def test_health_reports_the_loaded_model(
    client: TestClient, trained_pipeline: TrainedPipeline
) -> None:
    _, summary = trained_pipeline

    response = client.get('/health')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['model_loaded'] is True
    assert body['model_version'] == summary.model_version


def test_predict_returns_score_label_threshold_version_and_latency(
    client: TestClient, payload: Payload
) -> None:
    response = client.post('/predict', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body['fraud_probability'] <= 1.0
    is_fraud = body['fraud_probability'] >= body['decision_threshold']
    assert body['predicted_label'] == ('fraud' if is_fraud else 'legitimate')
    assert body['model_name'] == 'fraudguard'
    assert body['latency_ms'] > 0
    assert len(body['explanation']['contributions']) == 5
    assert response.headers['x-request-id'] == body['request_id']


def test_explanation_can_be_skipped(client: TestClient, payload: Payload) -> None:
    response = client.post('/predict', params={'explain': 'false'}, json=payload)

    assert response.status_code == 200
    assert response.json()['explanation'] is None


def test_client_request_id_is_echoed(client: TestClient, payload: Payload) -> None:
    response = client.post('/predict', json=payload, headers={'X-Request-ID': 'checkout-42'})

    assert response.headers['x-request-id'] == 'checkout-42'
    assert response.json()['request_id'] == 'checkout-42'


@pytest.mark.parametrize('supplied_id', ['spaces are not allowed', 'x' * 65])
def test_unsafe_request_ids_are_replaced(
    client: TestClient, payload: Payload, supplied_id: str
) -> None:
    response = client.post('/predict', json=payload, headers={'X-Request-ID': supplied_id})

    assert response.headers['x-request-id'] != supplied_id
    assert len(response.headers['x-request-id']) == 32


@pytest.mark.parametrize(
    'make_invalid', INVALID_PAYLOADS.values(), ids=list(INVALID_PAYLOADS.keys())
)
def test_invalid_payloads_are_rejected(
    client: TestClient, payload: Payload, make_invalid: Callable[[Payload], Payload]
) -> None:
    response = client.post('/predict', json=make_invalid(payload))

    assert response.status_code == 422
    assert response.json()['detail']


def test_non_finite_values_are_rejected(client: TestClient, payload: Payload) -> None:
    # json.dumps writes NaN, which Python's JSON parser accepts but the schema must not
    body = json.dumps({**payload, 'v3': float('nan')})

    response = client.post('/predict', content=body, headers={'Content-Type': 'application/json'})

    assert response.status_code == 422
    assert response.json()['detail'][0]['loc'] == ['body', 'v3']


def test_malformed_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/predict', content='{amount: 12', headers={'Content-Type': 'application/json'}
    )

    assert response.status_code == 422


def test_metrics_count_predictions(client: TestClient, payload: Payload) -> None:
    before = client.get('/metrics').json()['prediction_count']

    client.post('/predict', json=payload)
    metrics = client.get('/metrics').json()

    assert metrics['prediction_count'] == before + 1
    assert metrics['model_name'] == 'fraudguard'
    assert metrics['latency_ms']['p95'] > 0
    assert 'drift' in metrics


def test_openapi_schema_documents_every_input_field(client: TestClient) -> None:
    schemas = client.get('/openapi.json').json()['components']['schemas']

    assert set(schemas['TransactionRequest']['properties']) == set(INPUT_COLUMNS)


def test_service_without_a_model_reports_unavailable(
    app_config: AppConfig, payload: Payload
) -> None:
    with TestClient(create_app(app_config, model_loader=unavailable_registry)) as test_client:
        health = test_client.get('/health')
        prediction = test_client.post('/predict', json=payload)

    assert health.status_code == 503
    assert health.json()['model_loaded'] is False
    assert prediction.status_code == 503


def test_internal_errors_return_a_generic_500(app_config: AppConfig, payload: Payload) -> None:
    app = create_app(app_config, model_loader=lambda _: cast(FraudModel, BrokenModel()))

    with TestClient(app) as test_client:
        response = test_client.post('/predict', json=payload)

    assert response.status_code == 500
    assert response.json()['detail'] == 'Internal server error'
    assert 'hunter2' not in response.text
    assert response.json()['request_id'] == response.headers['x-request-id']
