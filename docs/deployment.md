# Deployment

## Prerequisites

- Docker
- GNU Make
- A GitHub personal access token (PAT) with `repo` scope (recommended — the server starts without it but cannot post statuses or comments)
- A publicly reachable URL for the webhook endpoint (ngrok works for local testing)

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GITHUB_WEBHOOK_SECRET` | Yes | — | Secret configured in the GitHub webhook settings |
| `GITHUB_TOKEN` | Recommended | — | PAT with `repo` scope for GitHub API calls. The server starts without it but logs a warning and cannot post statuses or comments. |
| `OPENAI_API_KEY` | For LLM review | — | API key for the LLM provider |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Override to point at LM Studio or an Anthropic proxy |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name |
| `PORT` | No | `8000` | Port the server listens on |
| `ROOT_PATH` | No | `` | Path prefix when running behind a reverse proxy (see nginx section) |
| `FORWARDED_ALLOW_IPS` | No | `127.0.0.1` | Comma-separated IPs/CIDRs uvicorn trusts for `X-Forwarded-*` headers. Set to `*` to trust all (e.g. inside a container network). |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./pr_checker.db` | SQLAlchemy async DB URL |

Put secrets in a `.env` file (gitignored) at the project root:

```
GITHUB_WEBHOOK_SECRET=your-secret
GITHUB_TOKEN=ghp_...
OPENAI_API_KEY=sk-...
```

Both `make docker-run` and `make docker-start` automatically pass `.env` to the
container if the file exists. The documented variables in the table above are also
forwarded from the host shell when exported, so exporting them is an equally valid
alternative to a `.env` file.

---

## Building the Docker image

```bash
make docker-build
```

Override the tag:

```bash
make docker-build TAG=1.2.3
```

---

## Running the container

### Foreground (logs stream to the terminal; Ctrl-C stops and removes the container)

```bash
make docker-run
```

With a non-default port and path prefix:

```bash
make docker-run PORT=9000 ROOT_PATH=/pr-checker
```

### Detached (background daemon)

```bash
make docker-start
```

View logs:

```bash
make docker-logs
```

Stop and remove the container:

```bash
make docker-stop
```

---

## Running locally without Docker

```bash
make install
./env/bin/uvicorn pr_checker.main:app --reload --port 8000 --env-file .env
```

With a path prefix:

```bash
./env/bin/uvicorn pr_checker.main:app --reload --port 8000 --env-file .env \
    --root-path /pr-checker --proxy-headers --forwarded-allow-ips 127.0.0.1
```

---

## nginx configuration

### Why `ROOT_PATH` and `FORWARDED_ALLOW_IPS` matter

When the service sits at `/pr-checker/` on your domain, every URL FastAPI generates
(redirects, `/docs`, `/openapi.json`) must include that prefix. `ROOT_PATH` tells
uvicorn to inject the prefix into the ASGI scope so FastAPI produces correct URLs
automatically.

nginx forwards the original scheme in the `X-Forwarded-Proto` header. Uvicorn only
trusts that header from IP addresses listed in `FORWARDED_ALLOW_IPS` (default
`127.0.0.1`). Without this, generated URLs will show `http://` instead of `https://`
even on a TLS-terminating proxy.

**When running in Docker** (the typical case), nginx reaches the container via the
Docker bridge network. From the container's perspective the request arrives from the
bridge gateway (e.g. `172.17.0.1`), *not* `127.0.0.1`, so the default will silently
ignore `X-Forwarded-Proto`. Set `FORWARDED_ALLOW_IPS=*` (or the specific gateway CIDR)
when starting the container:

```bash
make docker-start FORWARDED_ALLOW_IPS='*'
```

> **Security note**: `FORWARDED_ALLOW_IPS=*` trusts `X-Forwarded-*` headers from any
> source. This is safe only when the container port is not reachable by untrusted
> clients. The Makefile binds the port to `127.0.0.1` by default
> (`-p 127.0.0.1:PORT:PORT`), so only processes on the host can connect directly.
> If you override the binding to expose the port on all interfaces, restrict
> `FORWARDED_ALLOW_IPS` to the specific proxy IP or CIDR instead of using `*`.

**When running without Docker** (uvicorn directly on the host), nginx and uvicorn share
the loopback interface and the default `127.0.0.1` is correct.

The value of `ROOT_PATH` must exactly match the nginx `location` prefix (without the
trailing slash).

### Path-prefix setup (e.g. `https://example.com/pr-checker/webhook`)

```nginx
server {
    listen 443 ssl;
    server_name example.com;

    # ... TLS config ...

    location /pr-checker/ {
        proxy_pass          http://127.0.0.1:8000/;
        proxy_set_header    Host              $host;
        proxy_set_header    X-Real-IP         $remote_addr;
        proxy_set_header    X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header    X-Forwarded-Proto $scheme;

        # GitHub sends raw JSON bodies — disable buffering so the
        # payload reaches the app unmodified and HMAC validation passes.
        proxy_request_buffering off;
    }
}
```

Start the container to match:

```bash
make docker-start ROOT_PATH=/pr-checker
```

The webhook URL to register in GitHub: `https://example.com/pr-checker/webhook`

> **Trailing slash on `proxy_pass`**: `proxy_pass http://127.0.0.1:8000/` (with
> slash) strips the `/pr-checker` prefix before forwarding to the app. Without
> the trailing slash nginx would forward the full `/pr-checker/webhook` path and
> every route would 404.

### Root setup (e.g. `https://example.com/webhook`)

```nginx
server {
    listen 443 ssl;
    server_name example.com;

    # ... TLS config ...

    location / {
        proxy_pass          http://127.0.0.1:8000;
        proxy_set_header    Host              $host;
        proxy_set_header    X-Real-IP         $remote_addr;
        proxy_set_header    X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header    X-Forwarded-Proto $scheme;
        proxy_request_buffering off;
    }
}
```

No `ROOT_PATH` is needed in this case.

### HTTP-only (development / internal network)

```nginx
server {
    listen 80;
    server_name example.com;

    location /pr-checker/ {
        proxy_pass          http://127.0.0.1:8000/;
        proxy_set_header    Host              $host;
        proxy_set_header    X-Real-IP         $remote_addr;
        proxy_set_header    X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header    X-Forwarded-Proto $scheme;
        proxy_request_buffering off;
    }
}
```

---

## Testing locally with ngrok

1. Install ngrok and authenticate.
2. Start the server (Docker or local).
3. Expose port 8000:

   ```bash
   ngrok http 8000
   ```

4. Note the HTTPS forwarding URL (e.g. `https://abc123.ngrok.io`).
5. Register the webhook in GitHub (see next section) using that URL.

No nginx or `ROOT_PATH` is needed for ngrok — it proxies at the root.

---

## Registering the GitHub webhook

1. Go to your repository → **Settings** → **Webhooks** → **Add webhook**.
2. **Payload URL**: `https://example.com/pr-checker/webhook` (or your ngrok URL + `/webhook`).
3. **Content type**: `application/json`.
4. **Secret**: the value of `GITHUB_WEBHOOK_SECRET`.
5. **Events**: select **Let me select individual events** → tick **Pull requests**.
6. Ensure **Active** is checked, then click **Add webhook**.

---

## Verifying the setup

After a webhook is registered, open a pull request (or re-open an existing one).

1. **GitHub webhook deliveries** (Settings → Webhooks → your webhook → Recent Deliveries) should show a `202` response code from the server (empty body).
2. On the PR page, a commit status labelled **pr-checker** should appear as **pending**, then transition to **success** within a few seconds.
3. A placeholder comment from the bot account should appear on the PR.

If any step fails, check `make docker-logs` (detached) or the terminal output (foreground) for the server-side error.
