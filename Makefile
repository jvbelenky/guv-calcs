SHELL := /bin/bash

.PHONY: all test test-all test-cov test-fast clean format lint publish release interactive local

all: sync test

sync:
	uv sync --group local

test:
	uv run pytest tests/ -v

test-all:
	uv run pytest tests/ -v -m ""

test-cov:
	uv run pytest tests/ --cov=src/guv_calcs --cov-report=term-missing

test-fast:
	uv run pytest tests/ -v -x --ignore=tests/test_e2e.py -m "not slow"

local:
	uv sync --group local

interactive:
	uv run jupyter notebook notebooks/guv-calcs_test.ipynb --ip=0.0.0.0 --no-browser

format:
	uv run black src/guv_calcs/*

lint: format
	uv run flake8 --ignore=E114,E116,E117,E203,E231,E266,E302,E303,E501,E722,W293,W291,W503 src/guv_calcs/*

clean:
	@echo "Cleaning directory..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*~" -delete
	@find . -type f -name "*.kate-swp" -delete
	@echo "Done"

publish:
	rm -rf dist build */*.egg-info *.egg-info
	uv build
	uv publish

release:
	@bash scripts/release.sh $(VERSION)
	$(MAKE) publish
