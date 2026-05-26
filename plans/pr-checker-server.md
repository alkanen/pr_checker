# Plan: LLM-powered GitHub PR Review Server

> Source PRD: [GitHub issue #1](https://github.com/alkanen/pr_checker/issues/1)

## Architectural decisions

Durable decisions that apply across all phases:

- **Routes**: `POST /webhook`, `GET /health`, `GET /admin/reviews`, `GET /admin/queue`, `GET /admin/models`, `POST /admin/models/load`, `POST /admin/models/unload`, `GET /admin/config`, `POST /admin/config`, `POST /admin/index`
- **Schema**: Three SQLite tables — `review_jobs` (job lifecycle + PR metadata), `findings` (per-finding rows FK'd to job), `indexed_branches` (repo + branch + last indexed commit SHA)
- **Key models**: All inter-module data uses typed dataclasses or `NamedTuple`. Core types: `PRJob`, `DiffHunk`, `CodeSnippet`, `LinkedIssue`, `ProjectStandards`, `ReviewContext`, `Finding`, `ReviewResult`. Enums: `Severity` (critical/high/medium/low/info), `FindingCategory`, `ReviewVerdict` (approve/request_changes/comment), `JobStatus` (pending/in_progress/completed/failed), `ReviewTrigger` (opened/updated/review_requested/on_demand)
- **Auth**: `AuthProvider` protocol with `PATAuthProvider` implemented in v1. `GitHubAppAuthProvider` slot reserved — switching auth must not require changes to review logic. Note: GitHub Checks API requires a GitHub App token; v1 uses commit statuses (`POST /repos/{repo}/statuses/{sha}`) which work with PATs.
- **Service boundaries**: GitHub REST API via `httpx` (wrapped in `GitHubClient`); LM Studio / any OpenAI-compatible provider via `openai` SDK with configurable `base_url`; Qdrant via `qdrant-client`
- **Qdrant naming**: One collection per repo, named `{owner}__{repo}`. Every point payload includes a `branch` field; all queries filter by branch to isolate base vs feature branch results.
- **Config layering**: Server-level YAML is the base. Per-repo `.pr-checker.yml` is fetched from the repo's **default branch** at review time and deep-merged over server defaults. Repo config cannot escalate permissions beyond what the server allows.
- **Logging**: Structured JSON-lines to a log file. Every entry includes `timestamp` (ISO 8601), `level`, `github_account`, `repository`, `message`, and an open `extra` dict. `github_account` and `repository` propagate via Python `contextvars` — not threaded through function signatures.

---

## Phase 1: End-to-end pipe skeleton

**User stories**: 1, 4, 9, 17

### What to build

A thin vertical slice that proves every integration layer is wired up before any real logic is written. The server receives a GitHub `pull_request` webhook, validates the HMAC-SHA256 signature, creates a `PRJob` record in SQLite with status `pending`, posts a GitHub Checks API run with status `in_progress`, runs a no-op "review" that immediately produces a hardcoded placeholder comment on the PR, marks the job `completed`, and updates the Checks run to `success`. If anything throws, the job is marked `failed` and a brief error comment is posted to the PR.

The `ReviewQueue` processes one job at a time. All config (GitHub token, webhook secret, server port) comes from environment variables with a simple loader — no YAML yet.

### Acceptance criteria

- [ ] `POST /webhook` with a valid `pull_request` `opened` payload and correct HMAC signature returns HTTP 202
- [ ] A `review_jobs` row is created in SQLite with `status = pending`, then transitions to `in_progress`, then `completed`
- [ ] A GitHub commit status appears on the PR as `pending` when the job starts and `success` when it finishes (PAT-compatible commit status API; Checks API requires a GitHub App and is deferred to when App auth is implemented)
- [ ] A placeholder comment is posted to the PR by the bot account
- [ ] `POST /webhook` with an invalid signature returns HTTP 401 and no job is created
- [ ] If the placeholder posting raises an exception, the job is marked `failed` and an error comment is posted to the PR
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] Unit tests cover: signature validation (valid, invalid, missing), job state transitions, webhook event routing
- [ ] Integration test: full webhook → queue → placeholder review → GitHub comment flow with mocked GitHub HTTP

---

## Phase 2: GitHub data layer

**User stories**: 10, 11

### What to build

Replace the hardcoded PR data in Phase 1 with real fetched data. `GitHubClient` gains methods to fetch the PR diff (as a list of `DiffHunk` dataclasses), the full content of changed files, and linked issues. `IssueResolver` parses issue references from four sources in priority order: GitHub closing keywords (`closes #N`, `fixes #N`, `resolves #N`, etc.) in the PR description; raw issue URLs in the PR body; issue number embedded in the branch name (patterns like `issue-N`, `feat/N-slug`, `bugfix/N`); and as a last resort, an LLM inference call that reads the PR title and description and guesses the most likely issue. Each resolved reference is fetched from the GitHub API and returned as a `LinkedIssue` dataclass.

### Acceptance criteria

- [ ] Given a PR number, `GitHubClient` returns a parsed list of `DiffHunk` objects with file path, hunk header, added/removed lines, and line numbers
- [ ] `GitHubClient` fetches the full content of each changed file (for files below a size threshold) and the raw diff hunk for larger files
- [ ] `IssueResolver` correctly extracts issue numbers from: `closes #42` in PR body, a full GitHub issue URL, a branch named `feature/42-add-login`, and falls back to LLM inference when none of the above match
- [ ] Fetched `LinkedIssue` dataclasses include issue title, body, labels, and assignees
- [ ] Unit tests cover all four issue linking patterns, including absent links and malformed references
- [ ] Unit tests use mocked GitHub HTTP responses via `pytest-httpx`

---

## Phase 3: Config system + project standards detection

**User stories**: 12, 13, 18, 19, 29

### What to build

`ConfigManager` loads a server-level YAML config file on startup, validates it with Pydantic, and returns typed config dataclasses. At review time it fetches `.pr-checker.yml` from the target repo's default branch via `GitHubClient` and deep-merges it over the server defaults. The merged config controls: which review checks are enabled, which model is mapped to each task type, VRAM budget, embedding model, Qdrant connection, and which output types (inline comments, summary, formal review) are active.

`StandardsDetector` fetches known config files from the repo (`pyproject.toml`, `ruff.toml`, `mypy.ini`, `.eslintrc*`, `.prettierrc*`, etc.) via `GitHubClient`, extracts the tool-specific settings sections, and returns a `ProjectStandards` dataclass ready to be injected into the review prompt.

### Acceptance criteria

- [ ] Server starts with a missing config file and falls back to compiled-in defaults without crashing
- [ ] A `.pr-checker.yml` in a repo correctly overrides specific server-level settings while leaving unmentioned settings at their server defaults
- [ ] A repo config cannot set values that exceed server-level permission limits (e.g. cannot enable a check the server has disabled globally)
- [ ] `StandardsDetector` correctly parses ruff and mypy sections from `pyproject.toml` into `ProjectStandards`
- [ ] `StandardsDetector` returns an empty `ProjectStandards` without error when no standard config files exist in the repo
- [ ] Unit tests use real YAML/TOML fixture files, not mocked parsers
- [ ] Unit tests cover deep-merge edge cases: nested override, list replacement, null/missing key

---

## Phase 4: Model manager

**User stories**: 18, 19, 20, 21

### What to build

`ModelManager` interfaces with the LM Studio OpenAI-compatible API to: list all available models and their metadata (context window size, estimated VRAM from parameter count × quantisation multiplier); query which model is currently loaded and how much VRAM is in use; and load or unload models. Model selection for a given task follows two steps: (1) look up the configured model for the task type from config; (2) if the estimated prompt token count exceeds that model's context limit, walk available models sorted by context size and pick the smallest that fits. Before loading a new model, the manager checks whether both the currently loaded model and the needed model fit within the configured total VRAM budget — if yes, keep both; if no, unload the current model first.

### Acceptance criteria

- [ ] Given a task type and estimated token count, `ModelManager.get_model_for_task()` returns the configured model when it fits, or the smallest sufficient fallback model when it does not
- [ ] When a model switch is needed and both models fit in VRAM, neither is unloaded
- [ ] When a model switch is needed and they do not both fit, the current model is unloaded before loading the new one
- [ ] VRAM estimation uses: configured total VRAM, LM Studio's reported loaded model size, and a configurable bytes-per-parameter multiplier for unloaded models
- [ ] Unit tests cover: task mapping, context-size fallback, VRAM-fits-both, VRAM-must-swap, no-model-fits (raises an exception)
- [ ] System test (skipped if `LM_STUDIO_URL` env var is absent): load a model, verify it appears in VRAM state, unload it, verify cleared

---

## Phase 5: Real LLM review

**User stories**: 5, 6, 7, 8, 15

### What to build

Replace the Phase 1 placeholder with a real LLM review. `LLMClient` sends a structured prompt to the selected model containing: the PR diff hunks, project standards, linked issue context, and task instructions. It implements the tool-call loop: if the model emits a `get_code_snippet(file_path, symbol_name)` or `search_code(query)` tool call, the server resolves it (via `GitHubClient` for exact lookup; Qdrant stub returning empty results until Phase 7) and continues the conversation. The loop runs until the model produces a final structured response or a maximum-turn limit is reached.

`ReviewFormatter` takes the `ReviewResult` and produces the GitHub API payloads: inline `PullRequestReviewComment` objects anchored to diff positions, a PR-level summary comment, and a formal review submission with verdict (approve / request_changes / comment). Each of the three output types is individually gated by config flags. Findings include `severity` and `confidence` on every entry.

### Acceptance criteria

- [ ] A real LLM produces a `ReviewResult` with at least one `Finding` for a test diff containing a deliberate issue
- [ ] Each `Finding` has a valid `Severity`, a `confidence` float between 0 and 1, a `category`, a `message`, and an optional `suggestion`
- [ ] Inline comments are posted to the correct diff positions on the PR
- [ ] The summary comment appears as a top-level PR comment
- [ ] A formal GitHub review is submitted with a verdict
- [ ] Disabling inline comments in config suppresses them without affecting the summary or formal review
- [ ] The tool-call loop correctly resolves a `get_code_snippet` call and feeds the result back to the model
- [ ] A maximum tool-call turn limit is enforced; the review completes (possibly with a warning finding) rather than looping forever
- [ ] Unit tests for `ReviewFormatter` are pure: findings in → GitHub API payloads out, no HTTP calls
- [ ] Integration test: full review orchestration with mocked GitHub HTTP and mocked LLM HTTP

---

## Phase 6: Qdrant indexing pipeline

**User stories**: 22, 23, 24

### What to build

`CodeIndexer` fetches the file tree for a repo branch via `GitHubClient`, downloads each source file, and splits it into semantic chunks: top-level functions, classes, and module-level blocks (using language-specific parsing — at minimum Python via the `ast` module; other languages chunked by heuristic line counts). Each chunk is passed to `EmbeddingService`, which calls the configured embedding model's embeddings endpoint and returns a vector. `QdrantClient` upserts each chunk as a Qdrant point with payload `{repo_full_name, branch, file_path, chunk_type, content, start_line, end_line}`.

Indexing is triggered automatically when `POST /webhook` receives a `push` event for a tracked branch. A `POST /admin/index` endpoint (and a matching CLI command) triggers a full re-index of a specified repo + branch on demand. When a branch is deleted (GitHub `delete` event), all Qdrant points for that branch are removed.

### Acceptance criteria

- [ ] After indexing a Python repo branch, Qdrant contains one point per top-level function and class in each `.py` file, tagged with the correct branch
- [ ] A push webhook for a branch triggers incremental re-indexing of only the files changed in that push
- [ ] `POST /admin/index?repo=owner/repo&branch=main` triggers a full re-index and returns job status
- [ ] A `delete` branch webhook removes all Qdrant points for that branch from the collection
- [ ] `indexed_branches` SQLite table is updated with the latest indexed commit SHA after each successful index run
- [ ] Unit tests mock `EmbeddingService` and `QdrantClient`; test chunking logic independently with real Python source fixtures
- [ ] System test (skipped if `QDRANT_URL` and `LM_STUDIO_URL` absent): index a small real repo branch, verify point count in Qdrant

---

## Phase 7: Semantic context retrieval (RAG)

**User stories**: 14, 15

### What to build

`ContextBuilder` takes a PR's `DiffHunk` list, extracts identifiers and code patterns from changed lines, and issues semantic searches against Qdrant for both the base branch and the feature branch. Results are deduplicated (same content from both branches → keep one), re-ranked by relevance score, and trimmed to a configurable maximum snippet count and total token budget. The assembled `ReviewContext` now contains real `CodeSnippet` entries alongside the diff.

The `search_code(query)` tool call exposed to the LLM in Phase 5 is wired to live Qdrant instead of returning empty results. When the LLM calls it mid-review, `ContextBuilder.search(query, branch)` is invoked and the top results are injected as the next assistant turn.

### Acceptance criteria

- [ ] Given a diff that modifies a function that calls another function defined elsewhere in the repo, `ContextBuilder` returns the called function's implementation as a relevant snippet
- [ ] Snippets from the base branch and feature branch are both retrieved and correctly labelled in the `ReviewContext`
- [ ] Duplicate snippets (identical content from both branches) are deduplicated to a single entry
- [ ] A `search_code` tool call mid-review returns Qdrant results and the model continues the review using them
- [ ] The total token count of injected snippets is bounded by the configurable limit
- [ ] Unit tests mock `QdrantClient`; test deduplication, ranking, and token-budget trimming independently
- [ ] Integration test: review a test PR in a repo that has been indexed in Phase 6; verify that retrieved snippets appear in the structured review context

---

## Phase 8: Large PR handling

**User stories**: 16

### What to build

`ReviewOrchestrator` gains awareness of prompt size before sending to the LLM. It estimates the total token count of the assembled `ReviewContext` (diff + snippets + standards + issue context + instructions). If this exceeds the selected model's context limit, it first tries to escalate to a larger-context model via `ModelManager`. If no single model can fit the full context, it falls back to a chunked review: processes files (or logical groups of files) sequentially, collects per-chunk `Finding` lists, then runs a final synthesis pass with a lightweight model to produce a coherent overall `ReviewResult` and summary. The PR-level comment notes when chunked review was used.

### Acceptance criteria

- [ ] A PR whose diff fits within the configured model's context is reviewed in a single pass (no chunking)
- [ ] A PR whose diff exceeds the configured model's context but fits a larger available model is reviewed in a single pass with the larger model
- [ ] A PR whose diff exceeds all available model context sizes is reviewed in chunks; all findings are collected and a synthesis summary is produced
- [ ] The PR summary comment indicates when chunked review was used and which files were covered
- [ ] No findings are silently dropped: every chunk's findings appear in the final `ReviewResult`
- [ ] Unit tests use synthetic `ReviewContext` objects with controlled token counts to trigger each code path without a real LLM

---

## Phase 9: Structured logging

**User stories**: 26

### What to build

Replace all `print` statements and ad-hoc logging with a `StructuredLogger` that writes JSON-lines to a configurable log file path. Every log entry is a single JSON object: `{"timestamp": "<ISO8601>", "level": "INFO", "github_account": "...", "repository": "...", "message": "...", "extra": {...}}`. `github_account` and `repository` are stored in Python `contextvars` at the start of each webhook handler and webhook queue worker, so they propagate automatically to every log call inside that async context without being threaded through every function signature. Sensitive values (tokens, secrets) must never appear in log entries.

### Acceptance criteria

- [ ] Every significant event (webhook received, job enqueued, model loaded, review started, finding count, review posted, error) produces a log entry
- [ ] Every log entry is valid JSON on a single line
- [ ] Every log entry contains `timestamp`, `level`, `github_account`, `repository`, and `message`
- [ ] `github_account` and `repository` are correctly populated for log calls deep inside `LLMClient` and `GitHubClient` without those classes accepting them as parameters
- [ ] `GITHUB_TOKEN` and `GITHUB_WEBHOOK_SECRET` values never appear in any log entry
- [ ] Unit tests capture log output and assert on JSON structure and required fields
- [ ] Log file path is configurable; defaults to `pr_checker.log` in the working directory

---

## Phase 10: Admin UI

**User stories**: 27

### What to build

A set of FastAPI routes serving Jinja2 HTML templates, reachable at `/admin/`. Four pages:

- **Review history** — paginated table of past `review_jobs` rows with PR link, repo, trigger, status, verdict, finding count, and duration. Searchable/filterable by repo and status.
- **Queue** — live list of pending and in-progress jobs with enqueue time and current status. Auto-refreshes every few seconds via a simple meta-refresh or fetch poll.
- **Model management** — shows all models available in LM Studio, which is currently loaded, estimated and reported VRAM usage as a gauge, and buttons to manually load or unload a specific model (calls `ModelManager` via the backend).
- **Config editor** — renders the current merged server config as an editable YAML text area. Submitting posts the updated YAML, validates it via `ConfigManager`, and writes it to the config file if valid. Shows validation errors inline without saving.

No authentication on the admin UI in v1 (single-operator tool, assumed to be on a trusted network).

### Acceptance criteria

- [ ] `GET /admin/reviews` renders a table with at least the last 50 completed jobs from SQLite
- [ ] `GET /admin/queue` renders all pending and in-progress jobs; page reflects current state within a few seconds
- [ ] `GET /admin/models` shows all LM Studio models, marks the loaded one, and displays a VRAM gauge
- [ ] `POST /admin/models/load` with a model name loads that model via `ModelManager` and redirects back to the models page
- [ ] `POST /admin/models/unload` unloads the current model and redirects back
- [ ] `GET /admin/config` renders the current server config as editable YAML
- [ ] `POST /admin/config` with valid YAML saves the config and shows a success message; invalid YAML shows errors and does not save
- [ ] All admin pages return HTTP 200 and valid HTML when the database is empty
- [ ] Integration tests use the FastAPI test client to assert on HTML response content
