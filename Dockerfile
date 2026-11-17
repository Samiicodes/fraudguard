# Build stage: resolve dependencies from uv.lock into a virtual environment
FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies change less often than the code, so they get their own layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


# Runtime stage: the virtual environment, configuration, and entry scripts only
FROM python:3.12-slim-trixie

# XGBoost needs the OpenMP runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 fraudguard

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY config ./config
COPY scripts ./scripts

# Writable locations for a local tracking store, data, reports, and prediction logs
RUN mkdir -p data/monitoring reports /mlflow \
    && chown fraudguard:fraudguard /app /mlflow \
    && chown -R fraudguard:fraudguard data reports

# The image has no git, so GitPython should not complain when MLflow imports it
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    FRAUDGUARD_API_HOST=0.0.0.0

USER fraudguard
EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
