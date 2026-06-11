.PHONY: all test run lint

all: test run

test:
	python3 -m pytest -q

run:
	python3 run_all.py

lint:
	ruff check src tests run_all.py scripts
