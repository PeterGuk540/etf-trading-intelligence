# ETF Trading Intelligence System - Makefile

.PHONY: help install dev-install test lint format clean docker-up docker-down train backtest

help:
	@echo "ETF Trading Intelligence System - Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make dev-install   - Install development dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean up generated files"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make train        - Train models"
	@echo "  make backtest     - Run backtesting"

install:
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev,research]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check src/ tests/ --fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

train:
	python -m src.training.train --config config/model_configs.yaml

backtest:
	python -m src.evaluation.backtest --config config/trading_configs.yaml

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

monitoring:
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "API: http://localhost:8000"

db-init:
	python scripts/init_db.py

db-migrate:
	alembic upgrade head

download-data:
	python scripts/download_data.py --start 2018-01-01

validate-config:
	python scripts/validate_config.py