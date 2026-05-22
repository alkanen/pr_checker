---
name: start-server
description: Start the FastAPI dev server with uvicorn and hot-reload. Use when you want to run pr_checker locally or verify a change works end-to-end.
disable-model-invocation: true
---

Run the FastAPI development server:

```bash
source ./env/bin/activate && uvicorn pr_checker.main:app --reload --port 8000
```

The server listens on http://localhost:8000. GitHub webhooks should be tunneled to this port (e.g. via `ngrok http 8000`).

After starting, confirm the server is up by checking http://localhost:8000/docs for the auto-generated API docs.
