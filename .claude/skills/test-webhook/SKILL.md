---
name: test-webhook
description: Send a sample GitHub PR webhook payload to the local server. Use to manually test the PR review flow without a real GitHub event.
disable-model-invocation: true
---

Send a simulated `pull_request` opened event to the local server:

```bash
curl -s -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: sha256=REPLACE_WITH_VALID_SIG" \
  -d '{
    "action": "opened",
    "number": 1,
    "pull_request": {
      "number": 1,
      "title": "Test PR",
      "body": "Test PR body",
      "html_url": "https://github.com/owner/repo/pull/1",
      "head": {"sha": "abc123", "ref": "feature-branch"},
      "base": {"sha": "def456", "ref": "main"},
      "user": {"login": "testuser"}
    },
    "repository": {
      "full_name": "owner/repo"
    }
  }'
```

Note: If webhook signature validation is enabled, compute the real HMAC-SHA256 signature using `GITHUB_WEBHOOK_SECRET`, or temporarily disable signature checking in dev mode.
