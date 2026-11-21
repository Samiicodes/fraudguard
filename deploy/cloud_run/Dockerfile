# Image for the public demo on Google Cloud Run. scripts/deploy_cloud_run.py stages this file
# with the code and a model bundle, so the container needs no MLflow registry. It differs from
# the main Dockerfile in three ways: no BuildKit cache mounts, which Cloud Build does not
# enable, the bundle is baked in, and the service listens on 8080, the Cloud Run default.

FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-trixie

# XGBoost needs the OpenMP runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 fraudguard

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY config ./config
COPY scripts/run_api.py ./scripts/run_api.py
COPY model_bundle ./model_bundle

RUN mkdir -p data/monitoring \
    && chown fraudguard:fraudguard /app \
    && chown -R fraudguard:fraudguard data

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    FRAUDGUARD_API_HOST=0.0.0.0 \
    FRAUDGUARD_API_PORT=8080 \
    FRAUDGUARD_MODEL_BUNDLE=/app/model_bundle

USER fraudguard
EXPOSE 8080

CMD ["python", "scripts/run_api.py"]
