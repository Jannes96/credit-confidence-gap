PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help setup data test lint typecheck reproduce all clean

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -e ".[dev]"

setup: $(BIN)/python  ## Create the virtualenv and install the package

data: setup  ## Download and checksum-verify the raw dataset
	$(BIN)/python -m credit_trust_lab.data.download

test: setup  ## Run the test suite
	$(BIN)/pytest

lint: setup  ## Check formatting and lint rules
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

typecheck: setup  ## Run mypy
	$(BIN)/mypy

reproduce: data  ## Regenerate every figure and metric in reports/
	$(BIN)/python scripts/reproduce_all.py

all: lint typecheck test reproduce  ## Everything, in the order CI would run it

clean:  ## Remove generated artefacts (raw data is kept; it is checksum-verified)
	rm -rf reports/figures/*.png reports/metrics/*.json reports/metrics/*.txt
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
