SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)

 .PHONY: all test clean build release interactive
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

all: install test

## Install package (editable)
local:
	uv pip install -e .

install:
	uv pip install .

## Lint using flake8 and black
format: 
	black src/guv_calcs/*
	
lint: format
	flake8 --ignore=E114,E116,E117,E203,E231,E266,E302,E303,E501,E722,W293,W291,W503 src/guv_calcs/*

## Remove compiled python files
clean:
	@echo "Cleaning directory..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*~" -delete
	@find . -type f -name "*.kate-swp" -delete
	@echo "Done"

test:
	$(PYTHON_INTERPRETER) -m pytest tests/ -v

test-all:
	$(PYTHON_INTERPRETER) -m pytest tests/ -v -m ""

test-cov:
	$(PYTHON_INTERPRETER) -m pytest tests/ --cov=src/guv_calcs --cov-report=term-missing

test-fast:
	$(PYTHON_INTERPRETER) -m pytest tests/ -v -x --ignore=tests/test_e2e.py -m "not slow"
	
interactive:
	.venv/bin/jupyter notebook notebooks/guv-calcs_test.ipynb --ip=0.0.0.0 --no-browser

# ----- PyPI upload only -----
publish-pypi:
	rm -rf dist build */*.egg-info *.egg-info
	uv build
	twine upload dist/*

# ----- Full release pipeline -----
# release.sh bumps version, updates changelog, commits, tags, and pushes.
# The tag push triggers the GitHub Action to create a GitHub Release.
release:
	@bash scripts/release.sh $(VERSION)
	make publish-pypi