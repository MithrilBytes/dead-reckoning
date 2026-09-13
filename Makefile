.PHONY: install lock lock-check demo test lint typecheck check record clean

# The pins are resolved on the requires-python floor, and make lock writes the venv's
# version into constraints.txt. A newer interpreter still installs them: PYTHON=python3.14.
PYTHON ?= python3.13

# constraints.txt holds one exact version for every package in .[dev]. make lock
# rewrites it from a fresh resolution, and make lock-check fails when pyproject.toml
# no longer resolves to it. Both consult the package index.
install:
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --constraint constraints.txt --editable '.[dev]'

lock:
	./.venv/bin/python -m scripts.lock

lock-check:
	./.venv/bin/python -m scripts.lock --check

demo:
	./.venv/bin/python -m demo.scenario_outage

test:
	./.venv/bin/python -m pytest

lint:
	./.venv/bin/ruff check .
	./.venv/bin/ruff format --check .

typecheck:
	./.venv/bin/pyright

check: lint typecheck test

record:
	./demo/record_demo.sh

clean:
	rm -rf data .pytest_cache .ruff_cache .hypothesis
