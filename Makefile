.PHONY: install demo test lint typecheck check record clean

install:
	python -m venv .venv
	./.venv/bin/pip install -e '.[dev]'

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
