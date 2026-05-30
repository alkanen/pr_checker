PYTHON    := python3.10
VENV      := env
BIN       := $(VENV)/bin
IMAGE     := pr-checker
CONTAINER := pr-checker
TAG       ?= latest
PORT      ?= 8000
ROOT_PATH ?=

# Pass .env to the container if one exists
ENV_FILE_FLAG := $(shell [ -f .env ] && echo "--env-file .env")

# PORT always has a Make-level default (8000) so it is always forwarded.
APP_ENV_REQUIRED := -e PORT=$(PORT)

# Optional vars — passed by name so Docker inherits the value from the host shell,
# avoiding all quoting/escaping issues (including values containing quotes or globs).
# Only included when non-empty so ServerConfig defaults (and .env values) are preserved.
APP_ENV_OPT := \
	$(if $(ROOT_PATH),             -e ROOT_PATH) \
	$(if $(GITHUB_TOKEN),          -e GITHUB_TOKEN) \
	$(if $(GITHUB_WEBHOOK_SECRET), -e GITHUB_WEBHOOK_SECRET) \
	$(if $(OPENAI_API_KEY),        -e OPENAI_API_KEY) \
	$(if $(OPENAI_BASE_URL),       -e OPENAI_BASE_URL) \
	$(if $(LLM_MODEL),             -e LLM_MODEL) \
	$(if $(LLM_TIMEOUT),           -e LLM_TIMEOUT) \
	$(if $(DATABASE_URL),          -e DATABASE_URL) \
	$(if $(FORWARDED_ALLOW_IPS),   -e FORWARDED_ALLOW_IPS) \
	$(if $(LOG_LEVEL),             -e LOG_LEVEL) \
	$(if $(LOG_FORMAT),            -e LOG_FORMAT)

APP_ENV = $(APP_ENV_REQUIRED) $(APP_ENV_OPT)

.PHONY: all env install test lint fmt typecheck clean \
        docker-build docker-run docker-start docker-stop docker-logs

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

# --- Docker ---

docker-build:
	docker build -t $(IMAGE):$(TAG) .

# Foreground — logs stream to the terminal; Ctrl-C stops and removes the container
docker-run:
	docker rm -f $(CONTAINER) 2>/dev/null || true
	docker run --rm --name $(CONTAINER) \
		-p 127.0.0.1:$(PORT):$(PORT) \
		$(APP_ENV) \
		$(ENV_FILE_FLAG) \
		$(IMAGE):$(TAG)

# Detached — use docker-logs / docker-stop to manage
docker-start:
	docker rm -f $(CONTAINER) 2>/dev/null || true
	docker run -d --name $(CONTAINER) \
		-p 127.0.0.1:$(PORT):$(PORT) \
		$(APP_ENV) \
		$(ENV_FILE_FLAG) \
		$(IMAGE):$(TAG)

docker-stop:
	docker stop $(CONTAINER) 2>/dev/null || true
	docker rm   $(CONTAINER) 2>/dev/null || true

docker-logs:
	docker logs -f $(CONTAINER)
