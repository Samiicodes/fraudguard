# Development tasks. Every target runs through uv, so no virtual environment
# needs to be activated first. Python tools start with python -m instead of their
# .exe launchers, which Windows application control policies can block.

UV ?= uv
IMAGE ?= fraudguard:latest

.PHONY: help install lint format test sample-data train train-sample serve mlflow-ui notebook bundle deploy docker-build docker-up docker-down

help:
	@echo install       Install dependencies and the pre-commit hooks
	@echo lint          Run ruff and mypy
	@echo format        Apply ruff fixes and formatting
	@echo test          Run the test suite with coverage
	@echo sample-data   Generate the synthetic sample dataset
	@echo train         Train on data/raw/creditcard.csv
	@echo train-sample  Train on the synthetic sample dataset
	@echo serve         Start the API on http://127.0.0.1:8000
	@echo mlflow-ui     Browse runs and registered models on http://127.0.0.1:5000
	@echo notebook      Open the exploratory analysis in JupyterLab
	@echo bundle        Export the champion and demo transactions to dist/model_bundle
	@echo deploy        Deploy the bundle and the API to Google Cloud Run
	@echo docker-build  Build the Docker image
	@echo docker-up     Start MLflow, a sample training job, and the API with Docker Compose
	@echo docker-down   Stop the Docker Compose stack

install:
	$(UV) sync --frozen
	$(UV) run python -m pre_commit install

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run python -m mypy

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run python -m pytest --cov --cov-report=term-missing

sample-data:
	$(UV) run python scripts/generate_sample_data.py

train:
	$(UV) run python scripts/train_model.py

train-sample: sample-data
	$(UV) run python scripts/train_model.py --use_sample_data

serve:
	$(UV) run python scripts/run_api.py

mlflow-ui:
	$(UV) run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

notebook:
	$(UV) run --group notebook python -m jupyterlab notebooks/exploratory_analysis.ipynb

bundle:
	$(UV) run python scripts/export_model.py

deploy:
	$(UV) run python scripts/deploy_cloud_run.py

docker-build:
	docker build --tag $(IMAGE) .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
