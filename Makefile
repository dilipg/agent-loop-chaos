# agent-loop-chaos — developer targets.
#
# `make check` is the gate. It is exactly docs/07-TESTING.md §8 jobs 1-3: lint,
# types, tests, and the schema job. Do not weaken it to get a green run.

PY := python3
PKG := src/agent_loop_chaos

.DEFAULT_GOAL := help
.PHONY: help check lint fmt typecheck test cov schema golden-update demo build clean

help:  ## list targets
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

check: lint typecheck test schema  ## the gate: lint + types + tests + schemas

lint:  ## ruff check + format check
	ruff check .
	ruff format --check .

fmt:  ## apply ruff format and fix what is auto-fixable
	ruff format .
	ruff check --fix .

typecheck:  ## mypy --strict on the package
	mypy --strict $(PKG)

test:  ## pytest, default run (no network, excludes -m live)
	pytest -q

cov:  ## pytest with the 85% coverage gate enforced
	pytest -q --cov --cov-report=term-missing --cov-fail-under=85

# Job 3 of docs/07 §8. The dataclass<->schema field-parity assertion joins this
# target in M4, when report.py first has dataclasses to compare against.
schema:  ## validate the packaged schemas and every shipped example
	pytest -q tests/test_schemas.py
	$(PY) tools/verify_pack.py --strict

golden-update:  ## refresh tests/golden/*.json — review the diff in the commit
	ALC_UPDATE_GOLDEN=1 pytest -q tests -k golden

demo:  ## run the demo suite against the buggy example agent (M8+)
	alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-demo

build:  ## build the sdist and wheel, then twine check
	$(PY) -m build
	twine check dist/*

clean:  ## remove build, cache and coverage artefacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
