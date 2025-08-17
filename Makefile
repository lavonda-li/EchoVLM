.PHONY: install install-dev fmt lint test run clean help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package in development mode
	uv pip install -e . || pip install -e .

install-dev: ## Install package with development dependencies
	uv pip install -e .[dev] || pip install -e .[dev]

fmt: ## Format code with ruff and black
	ruff check --fix .
	black src/ tests/

lint: ## Run linting checks
	ruff check .
	mypy src/

test: ## Run tests
	pytest -q

test-verbose: ## Run tests with verbose output
	pytest -v

test-cov: ## Run tests with coverage
	pytest --cov=src/echo_infer --cov-report=html

run: ## Run inference with default config
	python -m echo_infer.cli run --config configs/default.yaml

run-dev: ## Run inference with debug logging
	python -m echo_infer.cli run --config configs/default.yaml --verbose

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: lint test ## Run all checks (lint + test)

pre-commit: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files
