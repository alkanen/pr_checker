# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pr_checker` is a GitHub webhook server that receives PR events and performs in-depth automated code review using an LLM. It posts review comments back to GitHub covering:

- Code quality and style
- Security vulnerabilities
- Test coverage
- Conformance to linked GitHub issues

## Stack

- **Python 3.10** with virtual environment at `./env/`
- **FastAPI** as the webhook server framework
- **LLM**: Configurable via OpenAI-compatible API (supports LM Studio locally and Anthropic via `OPENAI_BASE_URL` + `OPENAI_API_KEY`)
- **GitHub API**: REST calls to fetch PR diffs, issue details, and post review comments

## Dev Setup

```bash
make install   # create ./env/ and install all deps from pyproject.toml
make test      # run pytest (ARGS="-k test_name" to filter)
make lint      # ruff check
make fmt       # ruff format
make typecheck # mypy
make clean     # delete ./env/ and all caches
```

Start the dev server:

```bash
./env/bin/uvicorn pr_checker.main:app --reload --port 8000
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `GITHUB_WEBHOOK_SECRET` | Validates incoming webhook payloads |
| `GITHUB_TOKEN` | Personal access token for GitHub API calls |
| `OPENAI_API_KEY` | API key for the LLM provider (inference/completions) |
| `OPENAI_BASE_URL` | Override to point at LM Studio (`http://localhost:1234/v1`) or Anthropic proxy |
| `LLM_MODEL` | Model name to use (e.g. `claude-sonnet-4-6`, `llama-3.3-70b-instruct`) |
| `LLM_TIMEOUT` | Max seconds to wait between any two received chunks (default `600`); covers slow time-to-first-token on CPU-only backends |
| `LM_STUDIO_URL` | Base URL for the LM Studio **model-management** API (e.g. `http://localhost:1234/api/v1`); must include `/api/v1` — distinct from `OPENAI_BASE_URL` which ends in `/v1` |
| `LM_STUDIO_API_KEY` | Bearer token for the LM Studio management API (same key as `OPENAI_API_KEY` on most setups) |

## Code Conventions

- Use `async def` for all FastAPI route handlers and I/O-bound functions
- Type-annotate all function signatures; mypy runs on every edit
- Pydantic models for all request/response bodies and config
- Validate GitHub webhook signatures before processing any payload

## Tooling (already configured in `.claude/settings.json`)

After every file edit, the hooks automatically run:
- `ruff format` — auto-formats the file
- `ruff check` — lints and reports issues
- `mypy` — type-checks the file

Do not add noqa or type: ignore comments unless there is no correct alternative.

## Testing

Run tests with:

```bash
pytest
```

Use real HTTP test clients (e.g. `httpx.AsyncClient` with FastAPI's `app`) rather than mocking the framework layer. Mock external calls (GitHub API, LLM API) at the HTTP boundary.
