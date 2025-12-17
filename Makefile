 .PHONY: test clean build
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install package
local:
	rm -rf dist build */*.egg-info *.egg-info
	$(PYTHON_INTERPRETER) setup.py sdist
	pip install -e . --no-cache-dir
	
install:
	pip install -r requirements.txt
	rm -rf dist build */*.egg-info *.egg-info
	$(PYTHON_INTERPRETER) setup.py sdist
	pip install . --no-cache-dir

publish:
	rm -rf dist build */*.egg-info *.egg-info
	$(PYTHON_INTERPRETER) setup.py sdist bdist_wheel
	twine upload dist/*

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
	$(PYTHON_INTERPRETER) tests/test.py
	
all: install lint test