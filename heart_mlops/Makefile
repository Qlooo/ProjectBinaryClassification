.PHONY: setup lint test format all pipeline

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

lint:
	flake8 src tests
	isort --check-only src tests
	black --check src tests

format:
	isort src tests
	black src tests

test:
	pytest -q

train:
	python -m heart_pred.train --params params.yaml --artifacts artifacts

evaluate:
	python -m heart_pred.evaluate --model artifacts/model.joblib --params params.yaml

all: format lint test train evaluate

pipeline:
	dvc repro
