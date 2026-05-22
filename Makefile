PYTHON := python3.10
VENV   := env
BIN    := $(VENV)/bin

.PHONY: all env install test lint fmt typecheck clean

all: install

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

env: $(VENV)/bin/activate

install: env
	$(BIN)/pip install -q -e ".[dev]"

test: install
	$(BIN)/pytest $(ARGS)

lint: install
	$(BIN)/ruff check .

fmt: install
	$(BIN)/ruff format .

typecheck: install
	$(BIN)/mypy pr_checker

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
